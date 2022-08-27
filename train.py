import copy
import logging
import os
import time
import math
from shutil import copyfile
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp

from lip_convnets import LipConvNet
from utils import *

logger = logging.getLogger(__name__)


def iso_l1_loss(data1, data2):
    return -(data1 - data2).mean(0)  # the loss sign shouldn't matter since either is ok. (x, x' are just symbols)


def iso_l2_loss(data1, data2):
    return -(((data1 - data2).mean(0))**2)


def init_log(args, log_name='output.log'):
    args.out_dir += '_' + str(args.dataset)
    args.out_dir += '_' + str(args.l)
    args.out_dir += '_' + str(args.block_size)
    args.out_dir += '_' + str(args.conv_layer)
    args.out_dir += '_' + str(args.init_channels)
    args.out_dir += '_' + str(args.activation)
    if args.lln:
        args.out_dir += '_lln'

    os.makedirs(args.out_dir, exist_ok=True)
    code_dir = os.path.join(args.out_dir, 'code')
    os.makedirs(code_dir, exist_ok=True)
    for f in os.listdir('./'):
        src = os.path.join('./', f)
        dst = os.path.join(code_dir, f)
        if os.path.isfile(src):
            if f[-3:] == '.py' or f[-3:] == '.sh':
                copyfile(src, dst)

    logfile = os.path.join(args.out_dir, log_name)
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='%(message)s',
        level=logging.INFO,
        filename=os.path.join(args.out_dir, log_name))
    wandb.init(project="iso", entity="pbb")


def init_model(args):
    model = LipConvNet(args.conv_layer, args.activation, init_channels=args.init_channels,
                       block_size=args.block_size, num_classes=args.num_classes,
                       lln=args.lln)
    return model


def main():
    args = get_args()

    if args.conv_layer == 'cayley' and args.opt_level == 'O2':
        raise ValueError('O2 optimization level is incompatible with Cayley Convolution')

    init_log(args, log_name='output.log')
    logger.info(args)
    args.num_classes = 1
    assert args.n == 15000 and args.n % args.batch_size == 0, 'n must be 15000 and divisible by batch size'

    init_random(args.seed)
    train_loader_1, train_loader_2, _, _ = get_loaders(
        args.data_dir, args.batch_size, args.n, args.dataset, args.workers)

    model = init_model(args).cuda()
    model.train()

    conv_params, activation_params, other_params = parameter_lists(model)
    if args.conv_layer == 'soc':
        opt = torch.optim.SGD([
            {'params': activation_params, 'weight_decay': 0.},
            {'params': (conv_params + other_params), 'weight_decay': args.weight_decay}
        ], lr=args.lr_max, momentum=args.momentum)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum,
                              weight_decay=0.)

    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = True
    model, opt = amp.initialize(model, opt, **amp_args)

    if args.l == 'l1':
        criterion = iso_l1_loss
    elif args.l == 'l2':
        criterion = iso_l2_loss
    else:
        raise Exception('Unknown loss')

    lr_steps = args.epochs * len(train_loader_1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps // 2,
                                                                      (3 * lr_steps) // 4], gamma=0.1)

    wandb.config = {
        "learning_rate": args.lr_max,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }

    last_model_path = os.path.join(args.out_dir, 'last.pth')
    last_opt_path = os.path.join(args.out_dir, 'last_opt.pth')

    # Training
    start_train_time = time.time()

    # only need train and test loss
    logger.info('Epoch \t Seconds \t LR \t Train Loss')
    for epoch in range(args.epochs):
        model.train()
        start_epoch_time = time.time()
        train_loss = 0
        train_n = 0

        for i, (X_1, X_2) in enumerate(zip(train_loader_1, train_loader_2)):
            X_1, X_2 = X_1[0], X_2[0]
            X_1, X_2 = X_1.cuda(), X_2.cuda()

            output_1, output_2 = model(X_1), model(X_2)
            # if args.lln:
            #     curr_cert = lln_certificates(output, y, model.last_layer, L)
            # else:
            #     curr_cert = ortho_certificates(output, y, L)

            # ce_loss = criterion(output_1, output_2)
            # loss = ce_loss - args.gamma * F.relu(curr_cert).mean()

            loss = criterion(output_1, output_2)

            opt.zero_grad()
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            opt.step()

            train_loss += loss.item()
            train_n += 1
            scheduler.step()

        epoch_time = time.time()
        lr = scheduler.get_last_lr()[0]
        logger.info('%d \t %.1f \t %.4f \t %.4f',
                    epoch, epoch_time - start_epoch_time, lr, train_loss/train_n)

        wandb.log({"loss": train_loss/train_n})

        torch.save(model.state_dict(), last_model_path)

        trainer_state_dict = {'epoch': epoch, 'optimizer_state_dict': opt.state_dict()}
        torch.save(trainer_state_dict, last_opt_path)

    train_time = time.time()

    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)


if __name__ == "__main__":
    main()
