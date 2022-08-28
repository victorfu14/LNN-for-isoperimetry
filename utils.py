from custom_activations import MaxMin, HouseHolder, HouseHolder_Order_2
from skew_ortho_conv import SOC
from block_ortho_conv import BCOP
from cayley_ortho_conv import Cayley, CayleyLinear
import os
from shutil import copyfile
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import argparse
import math

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2507, 0.2507, 0.2507)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()

upper_limit = ((1 - mu) / std)
lower_limit = ((0 - mu) / std)


def init_random(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_args():
    parser = argparse.ArgumentParser()

    # isoperimetry arguments
    parser.add_argument('--n', default=15000, type=int, help='n for number of samples training on')
    parser.add_argument('--loss', default='l1', choices=['l1', 'l2'], type=str, help='Choose the loss function')
    parser.add_argument('--eval-num', default=100, type=int, help='Number of evaluation samples for each n')

    # Training specifications
    parser.add_argument('--batch-size', default=500, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--lr-min', default=1e-4, type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-drop', default=200, type=int)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O2'],
                        help='O0 is FP32 training and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
                        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')

    # Model architecture specifications
    parser.add_argument('--conv-layer', default='soc', type=str, choices=['bcop', 'cayley', 'soc'],
                        help='BCOP, Cayley, SOC convolution')
    parser.add_argument('--init-channels', default=32, type=int)
    parser.add_argument('--activation', default='maxmin', choices=['maxmin', 'hh1', 'hh2'],
                        help='Activation function')
    parser.add_argument('--block-size', default=1, type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8],
                        help='size of each block')
    parser.add_argument('--lln', action='store_true', help='set last linear to be linear and normalized')

    # Dataset specifications
    parser.add_argument('--data-dir', default='./cifar-data', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'],
                        help='dataset to use for training')

    # Other specifications
    parser.add_argument('--out-dir', default='ISO', type=str, help='Output directory')
    parser.add_argument('--workers', default=4, type=int, help='Number of workers used in data-loading')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()

# We partition our dataset into 10000+10000/10000+10000/10000+10000


def init_dataset(dir_, dataset_name='cifar10', normalize=True):
    if dataset_name == 'cifar10':
        dataset_func = datasets.CIFAR10
        mean = cifar10_mean
        std = cifar10_std
    elif dataset_name == 'cifar100':
        dataset_func = datasets.CIFAR100
        mean = cifar100_mean
        std = cifar100_std

    if normalize:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    train_dataset = dataset_func(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = dataset_func(
        dir_, train=False, transform=test_transform, download=True)
    return torch.utils.data.ConcatDataset([train_dataset, test_dataset])


def init_log(args, log_name='output.log'):
    args.out_dir += '_' + str(args.dataset)
    args.out_dir += '_' + str(args.loss)
    args.out_dir += '_' + str(args.block_size)
    args.out_dir += '_' + str(args.conv_layer)
    args.out_dir += '_' + str(args.init_channels)
    args.out_dir += '_' + str(args.activation)
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

    logfile = os.path.join(args.out_dir, log_name)
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='%(message)s',
        level=logging.INFO,
        filename=os.path.join(args.out_dir, log_name))


def get_loaders(dir_, batch_size=128, n=15000, dataset_name='cifar10', normalize=True, num_workers=4):
    train_dataset_1, train_dataset_2, test_dataset_1, test_dataset_2, _ = torch.utils.data.random_split(
        init_dataset(dir_, dataset_name, normalize), [n, n, n, n, 60000 - 4 * n])

    train_loader_1 = torch.utils.data.DataLoader(
        dataset=train_dataset_1,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    train_loader_2 = torch.utils.data.DataLoader(
        dataset=train_dataset_2,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    test_loader_1 = torch.utils.data.DataLoader(
        dataset=test_dataset_1,
        batch_size=n,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader_2 = torch.utils.data.DataLoader(
        dataset=test_dataset_2,
        batch_size=n,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    return train_loader_1, train_loader_2, test_loader_1, test_loader_2


conv_mapping = {
    'standard': nn.Conv2d,
    'soc': SOC,
    'bcop': BCOP,
    'cayley': Cayley
}


activation_dict = {
    'relu': F.relu,
    'swish': F.silu,
    'sigmoid': F.sigmoid,
    'tanh': F.tanh,
    'softplus': F.softplus,
    'maxmin': MaxMin()
}


def activation_mapping(activation_name, channels=None):
    if activation_name == 'hh1':
        assert channels is not None, channels
        activation_func = HouseHolder(channels=channels)
    elif activation_name == 'hh2':
        assert channels is not None, channels
        activation_func = HouseHolder_Order_2(channels=channels)
    else:
        activation_func = activation_dict[activation_name]
    return activation_func


def parameter_lists(model):
    conv_params = []
    activation_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'activation' in name:
                activation_params.append(param)
            elif 'conv' in name:
                conv_params.append(param)
            else:
                other_params.append(param)
    return conv_params, activation_params, other_params
