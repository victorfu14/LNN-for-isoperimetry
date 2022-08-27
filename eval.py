import logging
import os
import time
from train import iso_l1_loss, init_model, init_log
import matplotlib.pyplot as plt

import numpy as np
import torch

from utils import *

logger = logging.getLogger(__name__)


def evaluate(loader_1, loader_2, n_eval, num, model, criterion):
    losses_list = []
    model.eval()

    with torch.no_grad():
        for i, (X_1, X_2) in enumerate(zip(loader_1, loader_2)):
            X_1, X_2 = X_1[0], X_2[0]
            X_1, X_2 = X_1.cuda(), X_2.cuda()

            batch_size = num
            weights = torch.ones(n_eval).expand(batch_size, -1)
            id_X_1 = torch.multinomial(weights, num_samples=n_eval, replacement=False)
            id_X_2 = torch.multinomial(weights, num_samples=n_eval, replacement=False)
            for j, (id_1, id_2) in enumerate(zip(id_X_1, id_X_2)):
                output_1, output_2 = model(X_1[id_1]), model(X_2[id_2])
                loss = criterion(output_1, output_2)
                losses_list.append(loss.item())

        losses_array = torch.stack(losses_list).cpu().numpy()
    return losses_array.median(), losses_array


def main():
    args = get_args()

    if args.conv_layer == 'cayley' and args.opt_level == 'O2':
        raise ValueError('O2 optimization level is incompatible with Cayley Convolution')

    init_log(args, log_name='eval.log')

    args.num_classes = 1

    # TODO we should still look at L_1 loss when evaluating
    criterion = iso_l1_loss

    eval_list = [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000, 500, 100]
    assert args.n == 15000

    logger.info('-------------------------Last Model--------------------------')
    logger.info('n \t Test Loss \t Test Time')
    model = init_model(args).cuda()
    last_model_path = os.path.join(args.out_dir, 'last.pth')

    _, a = plt.subplots(nrows=4, ncols=3)
    a = a.ravel()

    init_random(args.seed)
    _, _, test_loader_1, test_loader_2 = get_loaders(
        args.data_dir, n=args.n, dataset_name=args.dataset, num_workers=args.workers)

    for i, (n_eval, ax) in enumerate(zip(eval_list, a)):
        model = init_model(args).cuda()
        model.load_state_dict(torch.load(last_model_path))
        model.float()
        model.eval()

        start_test_time = time.time()
        test_loss, test_loss_list = evaluate(test_loader_1, test_loader_2, n_eval, 100, model, criterion)

        wandb.log({"loss": test_loss})

        print('n = {}: Test Loss: median: {} | all: {}'.format(
            n_eval, np.abs(test_loss),  np.abs(test_loss_list)))

        # TODO plot the histogram of the test loss
        ax.hist(test_loss_list, bins='auto')
        ax.set_title('n = {}'.format(n_eval))
        ax.set_xlabel('Test Loss')
        ax.set_ylabel('Count')

        total_time = time.time() - start_test_time

        logger.info('%d \t %.4f \t %.4f \t %.4f \t %.4f',
                    np.abs(test_loss), total_time)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'hist.png'))


if __name__ == "__main__":
    main()
