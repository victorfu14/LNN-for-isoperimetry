import logging
import os
import time
from shutil import copyfile
from train import iso_loss, init_model, get_args

import numpy as np
import torch

from utils import *

logger = logging.getLogger(__name__)


def main():
    args = get_args()

    if args.conv_layer == 'cayley' and args.opt_level == 'O2':
        raise ValueError('O2 optimization level is incompatible with Cayley Convolution')

    args.out_dir += '_' + str(args.dataset)
    args.out_dir += '_n=' + str(args.n)
    args.out_dir += '_' + str(args.block_size)
    args.out_dir += '_' + str(args.conv_layer)
    args.out_dir += '_' + str(args.init_channels)
    args.out_dir += '_' + str(args.activation)
    args.out_dir += '_cr' + str(args.gamma)
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

    output = str(args.n) + '_on_' + str(args.n_eval) + '.log'
    logfile = os.path.join(args.out_dir, output)
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='%(message)s',
        level=logging.INFO,
        filename=os.path.join(args.out_dir, output))
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    assert args.n <= 60000/4 and args.n_eval in [10000, 8000, 6000,
                                                 4000, 2000, 1000, 500, 100]  # Make sure that n is not too large

    _, _, test_loader_1, test_loader_2 = get_loaders(
        args.data_dir, args.batch_size, args.n, args.dataset, eval=True, n_eval=args.n_eval)

    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        raise Exception('Unknown dataset')

    criterion = iso_loss

    best_model_path = os.path.join(args.out_dir, 'best.pth')
    last_model_path = os.path.join(args.out_dir, 'last.pth')

    logger.info('Model \t Test Loss \t Test Time')

    # Evaluation at best model
    model = init_model(args).cuda()
    model.load_state_dict(torch.load(best_model_path))
    model.float()
    model.eval()

    start_test_time = time.time()
    test_loss = torch.abs(evaluate_certificates(
        test_loader_1, test_loader_2, model, criterion, eval=True))
    total_time = time.time() - start_test_time

    logger.info('%s \t %.4f \t %.4f', 'Best', test_loss, total_time)

    # Evaluation at last model
    model.load_state_dict(torch.load(last_model_path))
    model.float()
    model.eval()

    start_test_time = time.time()
    test_loss = torch.abs(evaluate_certificates(
        test_loader_1, test_loader_2, model, criterion, eval=True))
    total_time = time.time() - start_test_time

    logger.info('Last Epoch \t Test Loss \t Test Time')
    logger.info('%s \t %.4f \t %.4f', 'Last', test_loss, total_time)


if __name__ == "__main__":
    main()
