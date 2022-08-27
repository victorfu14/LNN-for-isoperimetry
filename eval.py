import logging
import os
import time
from train import iso_l1_loss, init_model, init_log
import matplotlib.pyplot as plt
import wandb
import numpy as np
import torch

from utils import *

logger = logging.getLogger(__name__)


def evaluate(loader_1, loader_2, n_eval, num, model, criterion):
    losses_list = []
    hist = []
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
                losses_list.append(loss)
                hist.append([j, loss])

        losses_array = torch.stack(losses_list).cpu().numpy()
    return np.median(losses_array), losses_array, hist


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
    wandb.init(project="iso", entity="pbb", name=args.dataset+" b={}".format(args.block_size)+" eval")
    model = init_model(args).cuda()
    last_model_path = os.path.join(args.out_dir, 'last.pth')

    _, a = plt.subplots(nrows=4, ncols=3)
    a = a.ravel()

    init_random(args.seed)
    _, _, test_loader_1, test_loader_2 = get_loaders(
        args.data_dir, n=args.n, dataset_name=args.dataset, num_workers=args.workers)

    hist_all = []
    for i, (n_eval, ax) in enumerate(zip(eval_list, a)):
        model = init_model(args).cuda()
        model.load_state_dict(torch.load(last_model_path))
        model.float()
        model.eval()

        start_test_time = time.time()
        test_loss, test_loss_list, hist = evaluate(
            test_loader_1, test_loader_2, n_eval, args.eval_num, model, criterion)
        hist_all.append([n_eval, test_loss])

        histogram = wandb.plot.histogram(wandb.Table(
            data=hist, columns=["step", "loss"]), value='loss', title='n={}'.format(n_eval))
        wandb.log({'n={}'.format(n_eval): histogram})

        # TODO plot the histogram of the test loss
        ax.hist(test_loss_list, bins='auto')
        ax.set_title('n = {}'.format(n_eval))
        ax.set_xlabel('Test Loss')
        ax.set_ylabel('Count')

        total_time = time.time() - start_test_time

        logger.info('%d \t %.4f \t %.4f',
                    n_eval, np.abs(test_loss), total_time)

    histogram_all = wandb.plot.histogram(wandb.Table(
        data=hist_all, columns=["n", "loss"]), value='loss', title='histogram')
    wandb.log({'histogram': histogram_all})
    plt.tight_layout()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    plt.savefig(os.path.join(args.out_dir, 'hist.png'))


if __name__ == "__main__":
    main()
