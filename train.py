import argparse
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

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def get_args():
    parser = argparse.ArgumentParser()

    # isoperimetry arguments
    parser.add_argument('--train-size', default=1000, type=int)
    parser.add_argument('--val-size', default=1000, type=int)
    parser.add_argument('--loss', default='l1', type=str, choices=['l1', 'l2'])
    parser.add_argument('--synthetic', default=False, type=bool)
    parser.add_argument('--syn-data', default='gaussian', type=str, choices=['gaussian'])

    # Training specifications
    parser.add_argument('--batch-size', default=500, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.01, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O2'],
                        help='O0 is FP32 training and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
                        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')

    # Model architecture specifications
    parser.add_argument('--conv-layer', default='soc', type=str, choices=['bcop', 'cayley', 'soc'],
                        help='BCOP, Cayley, SOC convolution')
    parser.add_argument('--init-channels', default=32, type=int)
    parser.add_argument('--activation', default='maxmin', choices=['maxmin', 'hh1', 'hh2'],
                        help='Activation function')
    parser.add_argument('--block-size', default=1, type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8],
                        help='size of each block')
    parser.add_argument('--lln', action='store_true', help='set last linear to be linear and normalized')

    # Dataset specifications
    parser.add_argument('--data-dir', default='./cifar-data', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'],
                        help='dataset to use for training')

    # Other specifications
    parser.add_argument('--epsilon', default=36, type=int)
    parser.add_argument('--out-dir', default='LNNIso', type=str, help='Output directory')
    parser.add_argument('--workers', default=4, type=int, help='Number of workers used in data-loading')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def init_model(args):
    model = LipConvNet(args.conv_layer, args.activation, init_channels=args.init_channels,
                       block_size=args.block_size, num_classes=args.num_classes,
                       lln=args.lln)
    return model

def eval(args, epoch, model_path, test_loader, logger):
    # Evaluate on different test sample sizes
    logger.info('Epoch : {}'.format(epoch))
    logger.info('Test sample size \t Avg test loss \t Total time')
    for test_size in [50, 100, 250, 500, 1000, 5000, 8000, 10000]:
        # logger.info('Test sample size = %d', test_size)
        # test_sample = np.split(np.random.choice(len(test_loader.dataset), size=test_size*2, replace=False), 2)
        # Evaluation at best model (early stopping)
        model_test = init_model(args).cuda()
        model_test.load_state_dict(torch.load(model_path))
        model_test.float()
        model_test.eval()
            
        start_test_time = time.time()
        losses_arr = random_evaluate(args.synthetic, test_loader, model_test, test_size, 1000, args.loss)
        total_time = time.time() - start_test_time
        test_loss = np.mean(losses_arr)
        histogram = wandb.plot.histogram(wandb.Table(
            data=losses_arr, columns=["step", "loss"]), value='loss', title='n={}'.format(test_size))
        wandb.log({'n={}'.format(test_size): histogram})
        
        logger.info('%d \t %.4f \t %.4f', test_size, test_loss, total_time)


def main():
    args = get_args()
    if args.conv_layer == 'cayley' and args.opt_level == 'O2':
        raise ValueError('O2 optimization level is incompatible with Cayley Convolution')

    if args.synthetic:
        args.out_dir += '_' + str(args.syn_data)
    else:
        args.out_dir += '_' + str(args.dataset)

    args.out_dir += '_train_size=' + str(args.train_size)
    # args.out_dir += '_val_size=' + str(args.val_size)
    args.out_dir += '_loss=' + str(args.loss)
    args.out_dir += '_mom=' + str(args.momentum)
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

    train_logfile = os.path.join(args.out_dir, 'train.log')
    if os.path.exists(train_logfile):
        os.remove(train_logfile)
    
    eval_logfile = os.path.join(args.out_dir, 'eval.log')
    if os.path.exists(train_logfile):
        os.remove(eval_logfile)

    train_logger = setup_logger('train_logger', train_logfile)
    eval_logger = setup_logger('train_logger', eval_logfile)
    
    train_logger.info(args)
    eval_logger.info(args)

    args.num_classes = 1

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.synthetic:
        train_loader_1, train_loader_2, test_loader = get_synthetic_loaders(
            batch_size=args.batch_size,
            dataset_name=args.syn_data,
            dim=[3, 32, 32],
            train_size=args.train_size,
            test_size=40000
        )
    else:
        train_loader_1, train_loader_2, test_loader = get_loaders(
            args.data_dir, 
            args.batch_size, 
            args.dataset, 
            train_size = args.sample_size, 
        )
    
    std = cifar10_std if args.dataset == "cifar10" else cifar100_std

    # Only need R^d -> R lipschitz functions
    args.num_classes = 1

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

    criterion = isoLoss(args.loss)

    lr_steps = args.epochs * len(train_loader_1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[lr_steps // 2, (3 * lr_steps) // 4], gamma=0.1)

    last_model_path = os.path.join(args.out_dir, 'last.pth')
    last_opt_path = os.path.join(args.out_dir, 'last_opt.pth')

    # Training
    start_train_time = time.time()

    wandb.init(project="isoperimetry")
    wandb.config = {
        "learning_rate": args.lr_max,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }

    
    train_logger.info('Epoch \t Seconds \t LR \t Train Loss')
    for epoch in range(args.epochs):
        model.train()
        start_epoch_time = time.time()
        train_loss = 0
        train_n = 0

        for _, (X_1, X_2) in enumerate(zip(train_loader_1, train_loader_2)):
            if args.synthetic == False:
                X_1, X_2 = X_1[0], X_2[0]

            X_1, X_2 = X_1.cuda(), X_2.cuda()

            output1, output2 = model(X_1), model(X_2)
        
            ce_loss = criterion(output1, output2)
            loss = ce_loss

            opt.zero_grad()
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            opt.step()

            if args.loss == 'l1': 
                train_loss += ce_loss * X_1.size(0)
            else:
                train_loss += -torch.sqrt(-ce_loss) * X_1.size(0)
            train_n += X_1.size(0)
            scheduler.step()

        epoch_time = time.time()
        lr = scheduler.get_last_lr()[0]
        train_logger.info('%d \t %.1f \t %.4f \t %.4f',
                    epoch, epoch_time - start_epoch_time, lr, train_loss/train_n)
        
        wandb.log({"loss": train_loss, "lr": lr})
        wandb.watch(model)

        if epoch % 50 == 0:
            model_path = os.path.join(args.out_dir, 'epoch' + str(epoch) + '.pth')
            torch.save(model.state_dict(), model_path)
            eval(args, epoch, model_path, test_loader, eval_logger)

        torch.save(model.state_dict(), last_model_path)

        trainer_state_dict = {'epoch': epoch, 'optimizer_state_dict': opt.state_dict()}
        torch.save(trainer_state_dict, last_opt_path)

    train_time = time.time()

    train_logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

if __name__ == "__main__":
    main()
