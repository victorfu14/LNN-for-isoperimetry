import argparse
import copy
from inspect import ArgSpec
import logging
from multiprocessing.util import get_logger
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

from utils import *
from lip_convnets import LipConvNet

from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

test_size_list = [256, 512, 1024, 2048, 4096]

def init_model(args):
    args.in_planes = 1 if args.dataset == 'mnist' else 3
    model = LipConvNet(args.conv_layer, args.activation, init_channels=args.init_channels,
                       block_size=args.block_size, num_classes=args.num_classes,
                       lln=args.lln, syn=args.synthetic, in_planes=args.in_planes)
    return model

def eval(args, epoch, model_path, test_loader):
    # Evaluate on different test sample sizes
    logger = logging.getLogger('eval_logger')
    logger.info('Before epoch {}'.format(epoch))
    logger.info('Test sample size \t Avg abs test loss \t Total time')
    loss = {}
    for test_size in test_size_list:
        model_test = init_model(args).cuda()
        model_test.load_state_dict(torch.load(model_path))
        model_test.float()
        model_test.eval() if epoch != 0 else model_test.train()
            
        start_test_time = time.time()
        losses_arr = random_evaluate(args.synthetic, test_loader, model_test, test_size, 20)
        total_time = time.time() - start_test_time
        test_loss = np.mean(np.abs(losses_arr))
        loss[test_size] = [[val, epoch] for val in losses_arr]
        
        logger.info('%d \t %.4f \t %.4f', test_size, test_loss, total_time)
    
    return loss

def evaluate_model(args, test_loader):
    eval_logfile = os.path.join(args.out_dir, 'eval.log')
    if os.path.exists(eval_logfile):
        os.remove(eval_logfile)
    eval_logger = setup_logger('eval_logger', eval_logfile)
    eval_logger.info(args)

    start = time.time()
    loss = {}
    mean_loss_aggregate = []
    loss_reg = []
    # epoch_list = epoch_eval_list if args.epochs in epoch_eval_list else epoch_eval_list + [args.epochs]
    epoch_list = [50]
    for i in test_size_list:
        loss[i] = []
    for epoch in epoch_list:
        mean_loss = []
        model_path = os.path.join('LNNIso_mnist_batch_size=128_8_0.01_train_size=10000', 'epoch' + str(epoch) + '.pth')
        loss_this_epoch = eval(args, epoch, model_path, test_loader)
        for test_size in loss_this_epoch:
            loss[test_size] += loss_this_epoch[test_size]
            mean_loss_info = [np.mean(np.abs(np.transpose(loss_this_epoch[test_size])[0])), test_size, epoch]
            mean_loss.append(mean_loss_info[:2])
            mean_loss_aggregate.append(mean_loss_info)
            table = wandb.Table(data=loss[test_size], columns=['loss', 'epoch'])
            wandb.log({'n={}'.format(test_size): table})

        # regression
        X = 1 / np.sqrt([[size] for size in np.transpose(mean_loss)[1]])
        y = np.transpose(mean_loss)[0]

        reg = LinearRegression(fit_intercept=False).fit(X, y)
        eval_logger.info("Regression: loss = {} * 1 / sqrt(n), score = {}".format(reg.coef_, reg.score(X, y)))
        loss_reg.append([epoch, reg.coef_[0], reg.score(X, y)])
        table = wandb.Table(data=loss_reg, columns=['epoch', 'coefficient', 'fit score'])
        wandb.log({'regression result for each epoch': table})

        table = wandb.Table(data=mean_loss, columns=['avg loss', 'sample size'])
        wandb.log({'loss vs n, epoch={}'.format(epoch): table})
        table = wandb.Table(data=mean_loss_aggregate, columns=['avg loss', 'sample size', 'epoch'])
        wandb.log({'loss vs n': table})

    end = time.time()
    eval_logger.info('Total eval time: %.4f minutes', (end - start)/60)
    return

def load_mnist_per_class(dir_, batch_size, label):
    dataset_func = datasets.MNIST
    mean = (0.1307)
    std = (0.3081)

    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    dataset = dataset_func(
        dir_, train=True, transform=transform, download=True)
    dataset_2 = dataset_func(
        dir_, train=False, transform=transform, download=True)

    dataset.data = torch.cat((dataset.data, dataset_2.data), dim=0)
    dataset.targets = torch.cat((dataset.targets, dataset_2.targets), dim=0)

    index = np.where(dataset.targets == label)
    dataset.data = dataset.data[index]
    dataset.targets = dataset.targets[index]

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=len(dataset.data),
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

    return loader


def main():
    args = get_args()
    args = process_args(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.run_name += ' label=7'

    wandb.init(
        project='Isoperimetry',
        job_type='eval',
        name=args.run_name,
        config=vars(args)
    )
    
    test_loader = load_mnist_per_class(
        args.data_dir, 
        args.batch_size, 
        label=7
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
        
    evaluate_model(args, test_loader)
        
    
if __name__ == "__main__":
    main()
