import argparse
import numpy as np
import ot

from scipy.spatial.distance import cdist
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from utils import *

def main():
    args = get_args()

    torch.manual_seed(0)
    np.random.seed(0)

    logger = setup_logger('logger', args.dataset + '_noise=' + str(args.noise) + '_out.log')
    logger.info(args)

    num_samples = [50, 100, 200, 500, 1000, 5000, 10000, 20000]
    dist = []
    data = get_loaders(dataset_name=args.dataset, add_noise=args.noise)

    C = 0.5
    if args.dataset == 'mnist':
        C = 0.05

    for num in num_samples:
        idx = np.random.choice(len(data), num * 2, replace=False)
        A = [data[i].flatten().numpy() for i in idx[:num]]
        B = [data[i].flatten().numpy() for i in idx[num:]]
        M = cdist(A, B)
        
        a = np.ones(num)/num
        b = np.ones(num)/num
        dist.append(ot.sinkhorn2(a, b, M, reg=C))

    logger.info('Sample Size \t Wasserstein Dist' )
    for (n, d) in zip(num_samples, dist):
        logger.info('%d \t %.4f', n, d)

    X = np.log([[elem] for elem in dist])
    y = np.log(num_samples)

    reg = LinearRegression().fit(X, y)
    logger.info('Regression: log n = {} * log d + {} , score = {}'.format(reg.coef_[0], reg.intercept_, reg.score(X, y)))

if __name__ == "__main__":
    main()