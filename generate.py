import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from training import networks
from torch_utils import misc
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.utils as vutils
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log

#----------------------------------------------------------------------------

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64)
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl_edm',  help='Network pickle filename', metavar='PATH|URL',                  type=str, default='./MyNetwork')
@click.option('--network', 'network_pkl_cnf',  help='Network pickle filename', metavar='PATH|URL',                  type=str, default='./MyNetwork')
@click.option('--path_pre', 'dataset_path_pre',    help='Noise dataset path', metavar='PATH|URL',                   type=str, default='./MyDataset')
@click.option('--path_post', 'dataset_path_post',    help='Noise dataset path', metavar='PATH|URL',                 type=str, default='./MyDataset')
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, default='./outputs')
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-7', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=8, show_default=True)

def main(network_pkl_edm, network_pkl_cnf, dataset_path_pre, dataset_path_post, outdir, seeds, max_batch_size, device=torch.device('cuda'), **sampler_kwargs):

    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    ### Load network. folder-Pre_trained EDM
    dist.print0(f'Loading network from "{network_pkl_edm}"...')
    with open(network_pkl_edm, "rb") as f:
        net_pkl = pickle.load(f)['ema'].to(device)

    net_edm = networks.EDMPrecond(img_resolution=64,
                              img_channels=3,
                              model_type='SongUNet',
                              embedding_type='fourier',
                              encoder_type='residual',
                              channel_mult_noise=2,
                              resample_filter=[1, 3, 3, 1],
                              model_channels=128,
                              channel_mult=[1, 2, 2, 2],
                              augment_dim=9,
                              dropout=0,
                              use_fp16=False).cuda()
    net_edm.load_state_dict(net_pkl.state_dict(), strict=True)
    net_edm.eval()

    ### Load network. folder CNF
    dist.print0(f'Loading network from "{network_pkl_cnf}"...')
    with open(network_pkl_cnf, "rb") as f:
        net_pkl_cnf = pickle.load(f)['ema'].to(device)

    regularization_kwargs = dnnlib.EasyDict()
    cnf_kwargs = dnnlib.EasyDict()
    regularization_kwargs.update(l1int=None, l2int=None, dl2int=None, JFrobint=None, JdiagFrobint=None,
                                 JoffdiagFrobint=None,
                                 kinetic_energy=None, jacobian_norm2=None, total_deriv=None, directional_penalty=None)

    regularization_fns, regularization_coeffs = create_regularization_fns(regularization_kwargs)
    cnf_kwargs.update(T=1.0, train_T=False, regularization_fns=regularization_fns)
    cnf_kwargs.update(solver='dopri5', atol=1e-5, rtol=1e-5)
    net_cnf = networks.CMLPrecond(cnf_input_size=(max_batch_size, 3, 64, 64),
                                  cnf_n_scale=float('inf'),
                                  cnf_n_blocks=2,
                                  cnf_intermediate_dims=(64, 64, 64),
                                  cnf_nonlinearity='softplus',
                                  cnf_squash_input=True,
                                  alpha=0.05,
                                  cem_num_latents=3*64*64,
                                  cem_intv_dim=4,
                                  cem_gumbel_temperature=1.0,
                                  intv_cls_intv_dim=4,
                                  cnf_cnf_kwargs=cnf_kwargs).cuda()
    net_cnf.load_state_dict(net_pkl_cnf.state_dict(), strict=True)
    net_cnf.eval()



    ### Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    ### Load dataset.
    dataset_kwargs_pre = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=dataset_path_pre, use_labels=True, cache=True, xflip=False, use_npy=True)
    dataset_kwargs_post = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=dataset_path_post, use_labels=True, cache=True, xflip=False, use_npy=True)
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=1, prefetch_factor=2)

    dist.print0('Loading dataset...')
    dataset_obj_pre = dnnlib.util.construct_class_by_name(**dataset_kwargs_pre)
    dataset_obj_post = dnnlib.util.construct_class_by_name(**dataset_kwargs_post)
    dataset_sampler_pre = misc.InfiniteSampler(dataset=dataset_obj_pre, rank=dist.get_rank(), shuffle=True, num_replicas=dist.get_world_size(), seed=1432621761)
    dataset_sampler_post = misc.InfiniteSampler(dataset=dataset_obj_post, rank=dist.get_rank(), shuffle=True, num_replicas=dist.get_world_size(), seed=1432621761)
    dataset_iterator_pre = iter(torch.utils.data.DataLoader(dataset=dataset_obj_pre, sampler=dataset_sampler_pre, batch_size=max_batch_size, **data_loader_kwargs))
    dataset_iterator_post = iter(torch.utils.data.DataLoader(dataset=dataset_obj_post, sampler=dataset_sampler_post, batch_size=max_batch_size, **data_loader_kwargs))

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')


    Gamma = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]

    CC_Intervention = {
                        "red":      [1, 0, 0, 0],
                        "green":    [0, 1, 0, 0],
                        "blue":     [0, 0, 1, 0],
                        "arm":      [0, 0, 0, 1],
                       }

    cur_nimg = 0
    while True:

        torch.distributed.barrier()

        # Batch Sampling
        xT_pre, _ = next(dataset_iterator_pre)
        xT_post, intervention = next(dataset_iterator_post)

        # For GPU Device
        xT_pre = xT_pre.to(device)
        xT_post = xT_post.to(device)
        for key, value in CC_Intervention.items():
            intv = torch.tensor(value).to(device).repeat(8, 1)
            all_batches = []
            # print(1)
            for g in Gamma:
                with torch.no_grad():
                    mag = g * torch.ones(len(intv[0])).to(device)
                    z, z_intv, z_hat, x, x_intv, x_hat, intv, mag, eps, eps_intv, eps_hat, ldj, causal_basis = net_cnf(net_edm, xT_pre, xT_post, intv, mag=mag)
                    if g == 0:
                        x0_hat = edm_sampler(net_edm, xT_pre)
                    else:
                        x0_hat = edm_sampler(net_edm, z_hat)
                    all_batches.append(x0_hat * 0.5 + 0.5)
            grids = [vutils.make_grid(batch, nrow=1, padding=1, normalize=True) for batch in all_batches]
            final_grid = torch.cat(grids, 2) 
            save_image(final_grid, f'./results/{key}/{cur_nimg}.png')

        cur_nimg += max_batch_size
        if cur_nimg < 10000:
            continue
        else:
            break

    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
