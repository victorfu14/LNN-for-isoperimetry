import logging
import os
import time
from shutil import copyfile
from train import init_model, get_args

import numpy as np
import torch
import wandb

from utils import *

logger = logging.getLogger(__name__)

def eval(args, epoch, model_path, test_loader):
    # Evaluate on different test sample sizes
    logger = logging.getLogger('eval_logger')
    logger.info('Epoch : {}'.format(epoch))
    logger.info('Test sample size \t Avg test loss \t Total time')
    for test_size in [50, 100, 250, 500, 1000, 5000, 8000, 10000]:
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