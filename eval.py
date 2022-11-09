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

test_size_list = [5000, 10000]

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
    # logger.info('Test sample size \t Avg abs test loss \t Total time')
    data = []
    for test_size in test_size_list:
        model_test = init_model(args).cuda()
        model_test.load_state_dict(torch.load(model_path))
        model_test.float()
        model_test.eval() if epoch != 0 else model_test.train()
            
        start_test_time = time.time()
        moments_dic = moment_evaluate(args.synthetic, test_loader, model_test, test_size, 5)
        total_time = time.time() - start_test_time
        
        logger.info('test sample size: %d', test_size)
        logger.info('p \t p-th moment^{1/p}')
        for p in moments_dic:
            logger.info('%d \t %.4f', p, np.mean(moments_dic[p]))

        for p in moments_dic:
            for moment in moments_dic[p]:
                data.append([epoch, test_size, p, moment])

    return data

def evaluate_model(args, test_loader):
    eval_logfile = os.path.join(args.out_dir, 'eval_moment_all.log')
    if os.path.exists(eval_logfile):
        os.remove(eval_logfile)
    eval_logger = setup_logger('eval_logger', eval_logfile)
    eval_logger.info(args)

    start = time.time()
    data = []
    # mean_loss_aggregate = []
    # loss_reg = []
    epoch_list = epoch_eval_list if args.epochs in epoch_eval_list else epoch_eval_list + [args.epochs]
    # for i in test_size_list:
    #     loss[i] = []
    for epoch in epoch_list:
        mean_loss = []
        model_path = os.path.join(args.out_dir, 'epoch' + str(epoch) + '.pth')
        data_this_epoch = eval(args, epoch, model_path, test_loader)
        data += data_this_epoch
        table = wandb.Table(data=data, columns=['epoch', 'sample size', 'p-th moment', 'moment^{1/p}'])
        wandb.log({'moment table': table})
        # for test_size in loss_this_epoch:
        #     loss[test_size] += loss_this_epoch[test_size]
        #     mean_loss_info = [np.mean(np.abs(np.transpose(loss_this_epoch[test_size])[0])), test_size, epoch]
        #     mean_loss.append(mean_loss_info[:2])
        #     mean_loss_aggregate.append(mean_loss_info)
        #     table = wandb.Table(data=loss[test_size], columns=['loss', 'epoch'])
        #     wandb.log({'n={}'.format(test_size): table})

        # # regression
        # X = 1 / np.sqrt([[size] for size in np.transpose(mean_loss)[1]])
        # y = np.transpose(mean_loss)[0]

        # reg = LinearRegression(fit_intercept=False).fit(X, y)
        # eval_logger.info("Regression: loss = {} * 1 / sqrt(n), score = {}".format(reg.coef_, reg.score(X, y)))
        # loss_reg.append([epoch, reg.coef_[0], reg.score(X, y)])
        # table = wandb.Table(data=loss_reg, columns=['epoch', 'coefficient', 'fit score'])
        # wandb.log({'regression result for each epoch': table})

        # table = wandb.Table(data=mean_loss, columns=['avg loss', 'sample size'])
        # wandb.log({'loss vs n, epoch={}'.format(epoch): table})
        # table = wandb.Table(data=mean_loss_aggregate, columns=['avg loss', 'sample size', 'epoch'])
        # wandb.log({'loss vs n': table})

    end = time.time()
    eval_logger.info('Total eval time: %.4f minutes', (end - start)/60)
    return

def main():
    args = get_args()
    args = process_args(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.run_name += ' moment eval'

    if not args.debug:
        wandb.init(
            project='Isoperimetry',
            job_type='eval',
            name=args.run_name,
            config=vars(args)
        )
    
    _, _, test_loader = get_loaders(
        args.data_dir, 
        args.batch_size, 
        args.dataset, 
        train_size=args.train_size, 
        label=args.cifar5m_label
    ) if args.synthetic == False else get_synthetic_loaders(
        batch_size=args.batch_size,
        generate=args.syn_func,
        dim=args.dim,
        train_size=args.train_size,
    )
        
    evaluate_model(args, test_loader)
        
    
if __name__ == "__main__":
    main()
