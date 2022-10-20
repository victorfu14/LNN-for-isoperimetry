from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import logging
import argparse

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2507, 0.2507, 0.2507)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

mnist_mean = (0.1307)
mnist_std = (0.3081)

cifar10_maxpool_mean = (0.54904723, 0.5385685, 0.5022309)
cifar10_maxpool_std = (0.24201128, 0.23731293, 0.257864)
cifar10_avgpool_mean = (0.4914006, 0.48215854, 0.4465299)
cifar10_avgpool_std = (0.23807627, 0.23468511, 0.2535857)

cifar100_maxpool_mean = (0.5631373, 0.54179263, 0.4953446)
cifar100_maxpool_std = (0.26223433, 0.25095224, 0.27351803)
cifar100_avgpool_mean = (0.5070756, 0.48654792, 0.44091725)
cifar100_avgpool_std = (0.25941512, 0.24840887, 0.26879147)

mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)

formatter = logging.Formatter('%(message)s')

# epoch_store_list = [3]
epoch_store_list = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 50, 35, 45, 50, 75, 100, 150] 
epoch_eval_list = [0, 5, 10, 25, 50, 75, 100, 150]
# epoch_store_list = [0, 1, 2, 3, 4, 5, 7, 10, 15, 25, 35] # cifar10
# epoch_store_list = [0, 1, 4, 7, 10, 15, 25, 35] # cifar10
# epoch_store_list = [0, 1, 2, 3, 5, 7, 10, 15, 25, 50, 75] # cifar100

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def get_args():
    parser = argparse.ArgumentParser()

    # isoperimetry arguments
    parser.add_argument('--train-size', default=10000, type=int)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--dim', nargs='*', default=None, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--rand-label', action='store_true')
    # parser.add_argument('--loss', default='l1', type=str, choices=['l1', 'l2'])
    # parser.add_argument('--syn-data', default=None, type=str, choices=[None, 'gaussian'])

    # Training specifications
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
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
    parser.add_argument('--block-size', default=2, type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8],
                        help='size of each block')
    parser.add_argument('--lln', action='store_true', help='set last linear to be linear and normalized')

    # Dataset specifications
    parser.add_argument('--data-dir', default='./cifar-data', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'gaussian', 'mnist'],
                        help='dataset to use for training')

    # Other specifications
    parser.add_argument('--epsilon', default=36, type=int)
    parser.add_argument('--out-dir', default='LNNIso', type=str, help='Output directory')
    parser.add_argument('--workers', default=4, type=int, help='Number of workers used in data-loading')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()

def process_args(args):
    if args.conv_layer == 'cayley' and args.opt_level == 'O2':
        raise ValueError('O2 optimization level is incompatible with Cayley Convolution')

    if args.synthetic:
        if args.dataset == 'gaussian':
            args.syn_func = np.random.multivariate_normal 
        else:
            raise ValueError('Unknown synthetic dataset')
    
    args.out_dir += '_' + str(args.dataset)
    args.run_name = str(args.dataset) + ' block=' + str(args.block_size) + ' dim=' + str(args.dim) + ' MaxPool'

    args.out_dir += '_batch_size=' + str(args.batch_size)
    args.out_dir += '_' + str(args.block_size)
    args.out_dir += '_' + str(args.dim) + '_MaxPool'
    args.out_dir += '_' + str(args.lr_max)
    args.out_dir += '_train_size=' + str(args.train_size)

    if args.lln:
        args.out_dir += '_lln'

    # Only need R^d -> R lipschitz functions
    args.num_classes = 1

    return args

def isoLossEval(output1, output2, type='l1'):
    power = 2 if type == 'l2' else 1
    return -torch.mean(output1 - output2) ** power

class isoLoss(nn.Module):
    def __init__(self, loss='l1'):
        super(isoLoss, self).__init__()
        self.loss = loss
    
    def forward(self, output1, output2):
        return isoLossEval(output1, output2, type=self.loss)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def get_synthetic_loaders(batch_size, generate=np.random.multivariate_normal, dim=[3, 32, 32],train_size=10000, test_size=40000):
    total_dim = np.prod(dim)
    x_1 = generate(
        mean=np.zeros(np.prod(total_dim)),
        cov=np.identity(np.prod(total_dim)),
        size=train_size
    )
    x_2 = generate(
        mean=np.zeros(total_dim),
        cov=np.identity(total_dim),
        size=train_size
    )
    train_set_1 = torch.reshape(torch.tensor(x_1).float(), [train_size] + dim)
    train_set_2 = torch.reshape(torch.tensor(x_2).float(), [train_size] + dim)
    train_loader_1 = torch.utils.data.DataLoader(
        dataset=train_set_1,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    train_loader_2 = torch.utils.data.DataLoader(
        dataset=train_set_2,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    test = generate(
        mean=np.zeros(total_dim),
        cov=np.identity(total_dim),
        size=test_size
    )
    test_set = torch.reshape(torch.tensor(test).float(), [test_size] + dim)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=test_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    return train_loader_1, train_loader_2, test_loader

def get_loaders(dir_, batch_size, dataset_name='cifar10', normalize=True, train_size=10000, dim=None):
    if dataset_name == 'cifar10':
        dataset_func = datasets.CIFAR10
        mean = cifar10_mean if dim is None else cifar10_maxpool_mean
        std = cifar10_std if dim is None else cifar10_maxpool_std
    elif dataset_name == 'cifar100':
        dataset_func = datasets.CIFAR100
        mean = cifar100_mean if dim is None else cifar100_maxpool_mean
        std = cifar100_std if dim is None else cifar100_maxpool_std
    elif dataset_name == 'mnist':
        dataset_func = datasets.MNIST
        mean = mnist_mean
        std = mnist_std


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

    if dataset_name == 'mnist':
        train_transform = transforms.Compose([transforms.Pad(2), train_transform])
        test_transform = transforms.Compose([transforms.Pad(2), test_transform])
        
    num_workers = 4
    train_dataset = dataset_func(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = dataset_func(
        dir_, train=False, transform=test_transform, download=True)

    total_len = len(train_dataset.data) + len(test_dataset.data)

    total_set = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    train_dataset_1, train_dataset_2, test_dataset = torch.utils.data.random_split(
        total_set, 
        [train_size, train_size, total_len - 2 * train_size]
    )

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
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=total_len - 2 * train_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    return train_loader_1, train_loader_2, test_loader

def random_evaluate(synthetic, data_loader, model, size, num_sample, loss='l1'):
    losses_list = []
    # model.eval()

    for _ in range(num_sample):
        sample = np.split(np.random.choice(len(data_loader.dataset), size=size * 2, replace=False), 2)

        with torch.no_grad():
            for _, X in enumerate(data_loader):
                if synthetic == False:
                    X = X[0]
                X = X.cuda().float()
                output1 = model(X[sample[0]])
                output2 = model(X[sample[1]])
                loss = torch.tensor(np.array([isoLossEval(output1, output2, type=loss).cpu().numpy()]))
                losses_list.append(loss)
                    
            losses_array = torch.cat(losses_list, dim=0).cpu().numpy()

    return losses_array


from cayley_ortho_conv import Cayley, CayleyLinear
from block_ortho_conv import BCOP
from skew_ortho_conv import SOC

conv_mapping = {
    'standard': nn.Conv2d,
    'soc': SOC,
    'bcop': BCOP,
    'cayley': Cayley
}

from custom_activations import MaxMin, HouseHolder, HouseHolder_Order_2

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
