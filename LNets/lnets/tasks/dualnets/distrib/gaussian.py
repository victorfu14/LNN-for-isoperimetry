import numpy as np
import torch

from lnets.tasks.dualnets.distrib.base_distrib import BaseDistrib


class Gaussian(BaseDistrib):
    def __init__(self, config):
        super(Gaussian, self).__init__(config)

        self.dim = config.dim
        self.duplicate = config.duplicate
        self.perm_seed = config.perm_seed
        self.reshape_to_grid = config.reshape_to_grid
        self.rand_perm = True if self.perm_seed is not None else False

        assert self.dim > 0, "Dimensionality must be larger than 0. "
        assert self.duplicate > 0, "Duplicate must be larger than 0. "

    def __call__(self, size):
        perm = np.random.RandomState(seed=self.perm_seed).permutation

        intrinsic_dim = self.dim // self.duplicate

        intrinsic_samples = np.random.multivariate_normal(mean=np.zeros(
            shape=intrinsic_dim), cov=np.identity(intrinsic_dim), size=size)

        samples = np.empty([size, self.dim])
        if intrinsic_dim == 1536:
            for i, z in enumerate(intrinsic_samples):
                x = np.concatenate((z, z), axis=None)
                samples[i] = perm(x) if self.rand_perm else x
        elif intrinsic_dim == 1024:
            for i, z in enumerate(intrinsic_samples):
                x = np.concatenate((z, z, z), axis=None)
                samples[i] = perm(x) if self.rand_perm else x
        elif intrinsic_dim == 768:
            for i, z in enumerate(intrinsic_samples):
                x = np.concatenate((z, z, z, z), axis=None)
                samples[i] = perm(x) if self.rand_perm else x
        elif intrinsic_dim == 512:
            for i, z in enumerate(intrinsic_samples):
                x = np.concatenate((z, z, z, z, z, z), axis=None)
                samples[i] = perm(x) if self.rand_perm else x
        else:
            samples = intrinsic_samples

        if self.reshape_to_grid:
            samples = torch.reshape(torch.tensor(samples).float(), [size] + self.dim)

        return samples
