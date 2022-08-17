import logging
import os
import time
from shutil import copyfile
from train import iso_l1_loss, init_model, get_args

import numpy as np
import torch

from utils import *

logger = logging.getLogger(__name__)


def main():
    args = get_args()

    if args.conv_layer == 'cayley' and args.opt_level == 'O2':
        raise ValueError('O2 optimization level is incompatible with Cayley Convolution')

    args.out_dir += '_' + str(args.dataset)
    args.out_dir += '_' + str(args.l)
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

    output = 'eval.log'
    logfile = os.path.join(args.out_dir, output)
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='%(message)s',
        level=logging.INFO,
        filename=os.path.join(args.out_dir, output))
    # logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    eval_list = [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000,
                 2000, 1000, 500, 100]
    assert args.n in eval_list  # Make sure that n is not too large

    args.num_classes = 1

    # TODO we should still look at L_1 loss when evaluating
    criterion = iso_l1_loss

    logger.info('Model \t n \t Train Loss \t Valid Loss \t Test Loss \t Test Time')
    model = init_model(args).cuda()
    best_model_path = os.path.join(args.out_dir, 'best.pth')
    last_model_path = os.path.join(args.out_dir, 'last.pth')

    for n_eval in eval_list:
        train_loader_1, train_loader_2, valid_loader_1, valid_loader_2, test_loader_1, test_loader_2 = get_eval_loaders(
            args.data_dir, n_eval, args.n, args.dataset)
        # Evaluation at best model
        model = init_model(args).cuda()
        model.load_state_dict(torch.load(best_model_path))
        model.float()
        model.eval()

        start_test_time = time.time()
        train_loss = evaluate(train_loader_1, train_loader_2, model, criterion)
        valid_loss = evaluate(valid_loader_1, valid_loader_2, model, criterion)
        test_loss = evaluate(test_loader_1, test_loader_2, model, criterion)
        total_time = time.time() - start_test_time

        logger.info('%s \t %d \t %.4f \t %.4f \t %.4f \t %.4f', 'Best',
                    n_eval, train_loss, valid_loss, test_loss, total_time)

    for n_eval in eval_list:
        train_loader_1, train_loader_2, valid_loader_1, valid_loader_2, test_loader_1, test_loader_2 = get_eval_loaders(
            args.data_dir, n_eval, args.n, args.dataset)
        # Evaluation at last model

        model = init_model(args).cuda()
        model.load_state_dict(torch.load(best_model_path))
        model.float()
        model.eval()

        start_test_time = time.time()
        train_loss = evaluate(train_loader_1, train_loader_2, model, criterion)
        valid_loss = evaluate(valid_loader_1, valid_loader_2, model, criterion)
        test_loss = evaluate(test_loader_1, test_loader_2, model, criterion)
        total_time = time.time() - start_test_time

        logger.info('%s \t %d \t %.4f \t %.4f \t %.4f \t %.4f', 'Last',
                    n_eval, train_loss, valid_loss, test_loss, total_time)


if __name__ == "__main__":
    main()
