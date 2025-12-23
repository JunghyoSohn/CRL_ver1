# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy
#
#
#
#
# of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence
import numpy as np
import math
import torch.nn.functional as F
#----------------------------------------------------------------------------
class MMD_loss(torch.nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5, fix_sigma=None):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        return
    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        source = source.view(source.size(0), -1)
        target = target.view(target.size(0), -1)
        total = torch.cat([source, target], dim=0)

        x_norm = torch.sum(total ** 2, dim=1).view(-1, 1)
        y_norm = x_norm.view(1, -1)
        L2_distance = x_norm + y_norm - 2.0 * torch.matmul(total, total.t())
        
        n_samples = int(source.size()[0])+int(target.size()[0])
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

import lpips

class CMLLoss:

    def __init__(self):
        self.device = torch.device('cuda')
        self.BCE = torch.nn.BCELoss()
        self.MSE = torch.nn.MSELoss()
        self.CosineLoss = torch.nn.CosineEmbeddingLoss()
        self.MMD_loss= MMD_loss(fix_sigma=1000, kernel_num=10)
        self.lpips_dist = lpips.LPIPS(net='vgg').to(self.device)

    def __call__(self, edm, net, z, z_intv, intv, augment_pipe=None):
        net, z, z_intv, intv, augment_labels = augment_pipe(net, z, z_intv, intv) if augment_pipe is not None else (net, z, z_intv, intv, None)
        z, z_intv, z_hat, x, x_intv, x_hat, intv, mag, eps, eps_intv, eps_hat, ldj, causal_basis = net(edm, z, z_intv, intv, mag=None)

        l_mmd = self.MMD_loss(x_hat, x_intv)
        l_lpips = self.lpips_loss(x_hat, x_intv)

        l_cnf = self.compute_bits_per_dim(eps_hat, ldj)
        loss = l_mmd + l_cnf + l_lpips
        return loss, l_mmd, l_cnf, l_lpips

    def standard_normal_logprob(self, z):
        logZ = -0.5 * math.log(2 * math.pi)
        return logZ - z.pow(2) / 2

    def lpips_loss(self, x, y, weight=None, reduce=True):
        weight = weight if (weight is not None) else torch.ones(size=[x.shape[0]]).to(x.device)
        if reduce:
            return (weight * self.lpips_dist(x, y).reshape(-1)).sum()
        else:
            return weight * self.lpips_dist(x, y).reshape(-1)

    def compute_bits_per_dim(self, eps, ldj):

        # Don't use data parallelize if batch size is small.
        # if x.shape[0] < 200:
        #     model = model.module

        logp_eps = self.standard_normal_logprob(eps).view(eps.shape[0], -1).sum(1, keepdim=True)  # logp(eps)
        logp_z = logp_eps - ldj

        logpz_per_dim = torch.sum(logp_z) / eps.nelement()  # averaged over batches
        bits_per_dim = -(logpz_per_dim - np.log(256)) / np.log(2)

        return bits_per_dim
