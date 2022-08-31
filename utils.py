from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import math

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2507, 0.2507, 0.2507)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)

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

def get_synthetic_loaders(batch_size, dataset_name='gaussian', dim=[3, 32, 32],train_size=10000, test_size=40000):
    if dataset_name == 'gaussian':
        generate = np.random.multivariate_normal
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
    train_set_1 = torch.reshape(torch.tensor(x_1), [train_size] + dim)
    train_set_2 = torch.reshape(torch.tensor(x_2), [train_size] + dim)
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
    x = generate(
            mean=np.zeros(total_dim),
            cov=np.identity(total_dim),
            size=test_size
        )
    test = torch.reshape(torch.tensor(x), [test_size] + dim)
    test_loader = torch.utils.data.DataLoader(
        dataset=test,
        batch_size=test_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    return train_loader_1, train_loader_2, test_loader

def get_loaders(dir_, batch_size, dataset_name='cifar10', normalize=True, train_size=10000):
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
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    num_workers = 4
    train_dataset = dataset_func(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = dataset_func(
        dir_, train=False, transform=test_transform, download=True)

    total_len = len(train_dataset.data) + len(test_dataset.data)

    train_dataset_1, train_dataset_2, test_dataset = torch.utils.data.random_split(
        torch.utils.data.ConcatDataset([train_dataset, test_dataset]), 
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
    model.eval()

    for _ in range(num_sample):
        sample = np.split(np.random.choice(len(data_loader.dataset), size=size * 2, replace=False), 2)

        with torch.no_grad():
            for i, X in enumerate(data_loader):
                if synthetic == False:
                    X = X[0]
                X = X.cuda() 
                output1 = model(X[sample[0]])
                output2 = model(X[sample[1]])
                loss = torch.tensor(np.array([isoLossEval(output1, output2, type=loss, randomize=False).cpu().numpy()]))
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
