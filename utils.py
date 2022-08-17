from custom_activations import MaxMin, HouseHolder, HouseHolder_Order_2
from skew_ortho_conv import SOC
from block_ortho_conv import BCOP
from cayley_ortho_conv import Cayley, CayleyLinear
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

upper_limit = ((1 - mu) / std)
lower_limit = ((0 - mu) / std)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

# [ ] Generate Gaussian Data

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


def get_eval_loaders(dir_, n_eval=10000, n=10000, dataset_name='cifar10', normalize=True, num_workers=4):
    train_dataset_1, train_dataset_2, valid_dataset_1, valid_dataset_2, test_dataset_1, test_dataset_2 = torch.utils.data.random_split(
        init_dataset(dir_, dataset_name, normalize), [n, n, n, n, n, n])

    train_loader_1 = torch.utils.data.DataLoader(
        dataset=train_dataset_1,
        batch_size=n,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    train_loader_2 = torch.utils.data.DataLoader(
        dataset=train_dataset_2,
        batch_size=n,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    valid_loader_1 = torch.utils.data.DataLoader(
        dataset=valid_dataset_1,
        batch_size=n,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    valid_loader_2 = torch.utils.data.DataLoader(
        dataset=valid_dataset_2,
        batch_size=n,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    test_loader_1 = torch.utils.data.DataLoader(
        dataset=test_dataset_1,
        batch_size=n_eval,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader_2 = torch.utils.data.DataLoader(
        dataset=test_dataset_2,
        batch_size=n_eval,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    return train_loader_1, train_loader_2, valid_loader_1, valid_loader_2, test_loader_1, test_loader_2


def get_train_loaders(dir_, batch_size, n, dataset_name='cifar10', normalize=True, num_workers=4):
    train_dataset_1, train_dataset_2, valid_dataset_1, valid_dataset_2, _ = torch.utils.data.random_split(
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

    valid_loader_1 = torch.utils.data.DataLoader(
        dataset=valid_dataset_1,
        batch_size=n,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    valid_loader_2 = torch.utils.data.DataLoader(
        dataset=valid_dataset_2,
        batch_size=n,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    return train_loader_1, train_loader_2, valid_loader_1, valid_loader_2


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            if opt is not None:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts, limit_n=float("inf")):
    epsilon = (8 / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            if n >= limit_n:
                break
    return pgd_loss/n, pgd_acc/n


def attack_pgd_l2(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            if opt is not None:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd_l2(test_loader, model, attack_iters, restarts, limit_n=float("inf")):
    epsilon = (36 / 255.) / std
    alpha = epsilon/5.
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd_l2(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            if n >= limit_n:
                break
    return pgd_loss/n, pgd_acc/n


def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n


def ortho_certificates(output, class_indices, L):
    batch_size = output.shape[0]
    batch_indices = torch.arange(batch_size)

    onehot = torch.zeros_like(output).cuda()
    onehot[torch.arange(output.shape[0]), class_indices] = 1.
    output_trunc = output - onehot*1e6

    output_class_indices = output[batch_indices, class_indices]
    output_nextmax = torch.max(output_trunc, dim=1)[0]
    output_diff = output_class_indices - output_nextmax
    return output_diff/(math.sqrt(2)*L)


def lln_certificates(output, class_indices, last_layer, L):
    batch_size = output.shape[0]
    batch_indices = torch.arange(batch_size)

    onehot = torch.zeros_like(output).cuda()
    onehot[batch_indices, class_indices] = 1.
    output_trunc = output - onehot*1e6

    lln_weight = last_layer.lln_weight
    lln_weight_indices = lln_weight[class_indices, :]
    lln_weight_diff = lln_weight_indices.unsqueeze(1) - lln_weight.unsqueeze(0)
    lln_weight_diff_norm = torch.norm(lln_weight_diff, dim=2)
    lln_weight_diff_norm = lln_weight_diff_norm + onehot

    output_class_indices = output[batch_indices, class_indices]
    output_diff = output_class_indices.unsqueeze(1) - output_trunc
    all_certificates = output_diff/(lln_weight_diff_norm*L)
    return torch.min(all_certificates, dim=1)[0]


def evaluate(loader_1, loader_2, model, criterion):
    losses_list = []
    model.eval()

    with torch.no_grad():
        for i, (X_1, X_2) in enumerate(zip(loader_1, loader_2)):
            X_1, X_2 = X_1[0], X_2[0]
            X_1, X_2 = X_1.cuda(), X_2.cuda()
            output_1, output_2 = model(X_1), model(X_2)
            loss = criterion(output_1, output_2)
            losses_list.append(loss)

        losses_array = torch.stack(losses_list).cpu().numpy()
    return losses_array[0], losses_array


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
