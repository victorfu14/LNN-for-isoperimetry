import logging
import os
import time
from train import iso_l1_loss, init_model, init_log
import wandb
import numpy as np
import torch

from utils import *


def evaluate(loader_1, loader_2, n_eval, num, model, criterion):
    losses_list = []
    hist = []
    model.eval()

    with torch.no_grad():
        for i, (X_1, X_2) in enumerate(zip(loader_1, loader_2)):
            X_1, X_2 = X_1[0].cuda(), X_2[0].cuda()

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

    criterion = iso_l1_loss

    eval_list = [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000, 500, 100]
    eval_iter = [100 * i for i in range(args.epochs//100 + 1)]
    assert args.n == 15000

    wandb.init(project="iso", entity="pbb", name=args.dataset+" b={} eval".format(args.block_size))

    init_random(args.seed)
    _, _, test_loader_1, test_loader_2 = get_loaders(
        args.data_dir, n=args.n, dataset_name=args.dataset, num_workers=args.workers)

    for i in eval_iter:
        model = init_model(args).cuda()
        model.load_state_dict(torch.load(os.path.join(args.out_dir, 'iter_{}.pth'.format(i))))
        hist_all = []
        for n_eval in eval_list:
            model.float()
            model.eval()

            test_loss, test_loss_list, hist = evaluate(
                test_loader_1, test_loader_2, n_eval, args.eval_num, model, criterion)
            hist_all.append([n_eval, test_loss])
            histogram = wandb.plot.histogram(wandb.Table(
                data=hist, columns=["step", "loss"]), value='loss', title='iter={} n={}'.format(i, n_eval))
            wandb.log({'iter={} n={}'.format(i, n_eval): histogram})

        histogram_all = wandb.plot.histogram(wandb.Table(
            data=hist_all, columns=["n", "loss"]), value='loss', title='histogram')
        wandb.log({'histogram iter{}'.format(i): histogram_all})


if __name__ == "__main__":
    main()
