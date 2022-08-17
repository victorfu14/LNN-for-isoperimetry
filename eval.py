import logging
import os
import time
from shutil import copyfile
from train import iso_l1_loss, init_model, get_args, init_log

import numpy as np
import torch

from utils import *

logger = logging.getLogger(__name__)


def main():
    args = get_args()

    if args.conv_layer == 'cayley' and args.opt_level == 'O2':
        raise ValueError('O2 optimization level is incompatible with Cayley Convolution')

    init_log(args, log_name='eval.log')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.num_classes = 1

    # TODO we should still look at L_1 loss when evaluating
    criterion = iso_l1_loss

    eval_list = [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000, 500, 100]
    assert args.n == 10000

    logger.info('-------------------------Best Model--------------------------')
    logger.info('n \t Train Loss \t Valid Loss \t Test Loss \t Test Time')
    model = init_model(args).cuda()
    best_model_path = os.path.join(args.out_dir, 'best.pth')
    last_model_path = os.path.join(args.out_dir, 'last.pth')

    for n_eval in eval_list:
        train_loader_1, train_loader_2, valid_loader_1, valid_loader_2, test_loader_1, test_loader_2 = get_eval_loaders(
            args.data_dir, n_eval, args.n, args.dataset, args.workers)

        model = init_model(args).cuda()
        model.load_state_dict(torch.load(best_model_path))
        model.float()
        model.eval()

        start_test_time = time.time()
        train_loss, _ = evaluate(train_loader_1, train_loader_2, model, criterion)
        valid_loss, _ = evaluate(valid_loader_1, valid_loader_2, model, criterion)
        test_loss, test_loss_list = evaluate(test_loader_1, test_loader_2, model, criterion)

        if n_eval <= 2000:
            print('n = {}: Test Loss: mean: {} | all: {}'.format(
                n_eval, np.abs(np.mean(test_loss_list)),  np.abs(test_loss_list)))

        total_time = time.time() - start_test_time

        logger.info('%d \t %.4f \t %.4f \t %.4f \t %.4f',
                    n_eval, np.abs(train_loss), np.abs(valid_loss), np.abs(test_loss), total_time)

    logger.info('-------------------------Last Model--------------------------')
    for n_eval in eval_list:
        train_loader_1, train_loader_2, valid_loader_1, valid_loader_2, test_loader_1, test_loader_2 = get_eval_loaders(
            args.data_dir, n_eval, args.n, args.dataset, args.workers)

        model = init_model(args).cuda()
        model.load_state_dict(torch.load(last_model_path))
        model.float()
        model.eval()

        start_test_time = time.time()
        train_loss, _ = evaluate(train_loader_1, train_loader_2, model, criterion)
        valid_loss, _ = evaluate(valid_loader_1, valid_loader_2, model, criterion)
        test_loss, test_loss_list = evaluate(test_loader_1, test_loader_2, model, criterion)
        total_time = time.time() - start_test_time

        if n_eval <= 2000:
            print('n = {}: Test Loss: mean: {} | all: {}'.format(
                n_eval, np.abs(np.mean(test_loss_list)),  np.abs(test_loss_list)))

        logger.info('%d \t %.4f \t %.4f \t %.4f \t %.4f',
                    n_eval, np.abs(train_loss), np.abs(valid_loss), np.abs(test_loss), total_time)


if __name__ == "__main__":
    main()
