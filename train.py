import logging
from multiprocessing.util import get_logger
import os
import time
import math
from shutil import copyfile
import wandb

import numpy as np
import torch
from apex import amp
from utils import *
from lip_convnets import LipConvNet


def init_model(args):
    args.in_planes = 1 if args.dataset == 'mnist' else 3
    model = LipConvNet(args.conv_layer, args.activation, init_channels=args.init_channels,
                       block_size=args.block_size, num_classes=args.num_classes,
                       lln=args.lln, in_planes=args.in_planes)
    # summary(model, input_size=(128, 3, 32, 32))
    # summary(model, input_size=(128, 1, 28, 28))
    return model


def main():
    args = get_args()
    args = process_args(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not args.debug:
        wandb.init(
            project='Isoperimetry',
            job_type='train',
            name='({}x) ID={}{}{}'.format(args.duplicated, args.intrinsic_dim,
                                          ' P' if args.rand_perm else '', ' NN1e-2' if args.rand_noisy > 0 else ''),
            config=vars(args)
        )

    train_loader_1, train_loader_2, test_loader = get_synthetic_loaders(
        batch_size=args.batch_size,
        generate=args.syn_func,
        dim=args.dim,
        intrinsic_dim=args.intrinsic_dim,
        rand_perm=args.rand_perm,
        rand_noisy=args.rand_noisy,
        train_size=args.train_size,
    ) if args.dataset == 'gaussian' else get_loaders(
        args.data_dir,
        args.batch_size,
        args.dataset,
        train_size=args.train_size,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    code_dir = os.path.join(args.out_dir, 'code')
    os.makedirs(code_dir, exist_ok=True)
    for f in os.listdir('./'):
        src = os.path.join('./', f)
        dst = os.path.join(code_dir, f)
        if os.path.isfile(src):
            if f[-3:] == '.py' or f[-3:] == '.sh':
                copyfile(src, dst)

    train_logfile = os.path.join(args.out_dir, 'train.log')
    if os.path.exists(train_logfile):
        os.remove(train_logfile)
    train_logger = setup_logger('train_logger', train_logfile)
    train_logger.info(args)

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

    criterion = isoLoss()

    lr_steps = args.epochs
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[lr_steps // 2, (3 * lr_steps) // 4, (7 * lr_steps) // 8], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=20, min_lr=args.lr_min, factor=0.6)

    last_model_path = os.path.join(args.out_dir, 'last.pth')
    last_opt_path = os.path.join(args.out_dir, 'last_opt.pth')

    # Training
    start_train_time = time.time()
    train_logger.info('Epoch \t Seconds \t LR \t Train Loss')
    for epoch in range(args.epochs):
        if epoch in epoch_store_list:
            model_path = os.path.join(args.out_dir, 'epoch' + str(epoch) + '.pth')
            torch.save(model.state_dict(), model_path)

        model.train()
        start_epoch_time = time.time()
        train_loss = 0
        train_n = 0

        for _, (X_1, X_2) in enumerate(zip(train_loader_1, train_loader_2)):
            if args.dataset != 'gaussian':
                X_1, X_2 = X_1[0], X_2[0]

            X_1, X_2 = X_1.cuda(), X_2.cuda()

            output1, output2 = model(X_1), model(X_2)

            # print(output1.size())

            ce_loss = criterion(output1, output2)
            loss = ce_loss

            opt.zero_grad()
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            opt.step()

            train_loss += ce_loss * X_1.size(0)
            train_n += X_1.size(0)

        epoch_time = time.time()
        train_loss /= train_n

        # # reduce on plateau scheduler
        # scheduler.step(train_loss)
        # lr = scheduler._last_lr[0]

        # multistep scheduler
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        train_logger.info('%d \t %.1f \t %.4f \t %.4f',
                          epoch, epoch_time - start_epoch_time, lr, train_loss)

        if not args.debug:
            wandb.log({"loss": train_loss})
            if epoch % 50 == 0:
                wandb.watch(model)

        torch.save(model.state_dict(), last_model_path)
        trainer_state_dict = {'epoch': epoch, 'optimizer_state_dict': opt.state_dict()}
        torch.save(trainer_state_dict, last_opt_path)

    model_path = os.path.join(args.out_dir, 'epoch' + str(args.epochs) + '.pth')
    torch.save(model.state_dict(), model_path)
    train_time = time.time()
    train_logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)


if __name__ == "__main__":
    main()
