# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop

from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import add_spectral_norm, spectral_norm_power_iteration
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from train_misc import build_model_tabular

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

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

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, default='./results')
@click.option('--data_pre',      help='Path to the dataset', metavar='ZIP|DIR',                     type=str, default='./Mydata')
@click.option('--data_post',     help='Path to the dataset', metavar='ZIP|DIR',                     type=str, default='./Mydata')
@click.option('--use_labels',    help='Use intervention labels', metavar='BOOL',                    type=bool, default=True, show_default=True)
@click.option('--latent_data',   help='Train normalizing flow model', metavar='BOOL',               type=bool, default=True, show_default=True)
@click.option('--arch',          help='Network architecture', metavar='cnf',                        type=str, default='cnf', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='ffjord',          type=click.Choice(['ffjord']), default='ffjord', show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=1e-4, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)

# Performance-related.
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=5, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=5, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
# @click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str, default="./training-state-000000.pt")
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)

def main(**kwargs):

    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    ### Initialize config dict.
    c = dnnlib.EasyDict()
    c.edm_network_kwargs = dnnlib.EasyDict(img_resolution=64, img_channels=3, model_type='SongUNet',
                                           embedding_type='fourier', encoder_type='residual', channel_mult_noise=2,
                                           resample_filter=[1, 3, 3, 1], model_channels=128, channel_mult=[1, 2, 2, 2],
                                           augment_dim=9, dropout=0, use_fp16=False)
    c.dataset_kwargs_pre = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data_pre, use_labels=opts.use_labels, cache=opts.cache, use_npy=opts.latent_data)
    c.dataset_kwargs_post = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data_post, use_labels=opts.use_labels, cache=opts.cache, use_npy=opts.latent_data)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.loss_kwargs = dnnlib.EasyDict()
    c.network_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9, 0.999], eps=1e-8)

    ### Validate dataset options.
    try:
        dataset_obj_pre = dnnlib.util.construct_class_by_name(**c.dataset_kwargs_pre)
        dataset_obj_post = dnnlib.util.construct_class_by_name(**c.dataset_kwargs_post)
        dataset_name = dataset_obj_post.name
        c.dataset_kwargs_pre.resolution = dataset_obj_pre.resolution
        c.dataset_kwargs_post.resolution = dataset_obj_post.resolution  # be explicit about dataset resolution
        c.dataset_kwargs_pre.max_size = len(dataset_obj_pre)  # be explicit about dataset size
        c.dataset_kwargs_post.max_size = len(dataset_obj_post)  # be explicit about dataset size
        if not dataset_obj_post.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj_post
        del dataset_obj_pre
        # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # CNF
    c.network_kwargs.update(cnf_input_size=(opts.batch, 3, 64, 64), cnf_n_scale=float('inf'),
                            cnf_n_blocks=2, cnf_intermediate_dims=(64, 64, 64), cnf_nonlinearity='softplus')
    c.network_kwargs.update(cnf_squash_input=True, alpha=0.05)

    SOLVERS = ["dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun", "euler", "midpoint", "rk4", "explicit_adams",
               "implicit_adams", "fixed_adams", "scipy_solver"]
    regularization_kwargs = dnnlib.EasyDict()
    cnf_kwargs = dnnlib.EasyDict()

    regularization_kwargs.update(l1int=None, l2int=None, dl2int=None, JFrobint=None, JdiagFrobint=None, JoffdiagFrobint=None,
                                 kinetic_energy=None, jacobian_norm2=None, total_deriv=None, directional_penalty=None)

    regularization_fns, regularization_coeffs = create_regularization_fns(regularization_kwargs)
    cnf_kwargs.update(T=1.0, train_T=False, regularization_fns=regularization_fns)
    cnf_kwargs.update(solver='dopri5', atol=1e-5, rtol=1e-5)
    c.network_kwargs.update(cnf_cnf_kwargs=cnf_kwargs)

    # CEM
    c.network_kwargs.update(cem_num_latents=3*64*64, cem_intv_dim=4, cem_gumbel_temperature=0.1)

    # Intv_cls
    c.network_kwargs.update(intv_cls_intv_dim=4)

    ### Preconditioning & loss function.
    c.edm_network_kwargs.class_name = 'training.networks.EDMPrecond'
    c.network_kwargs.class_name   = 'training.networks.CMLPrecond'
    c.loss_kwargs.class_name = 'training.loss.CMLLoss'

    ### Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    ### Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    ### Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume


    ### Description string.
    # cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    desc = f'{dataset_name:s}-{opts.arch:s}-{opts.precond:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'


    ### Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if callable(obj):
                return str(obj) 
            return json.JSONEncoder.default(self, obj)

    ### Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2, cls=CustomEncoder))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs_post.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs_post.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0()

    ### Dry run
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    ### Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2, cls=CustomEncoder)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    ### Train.
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
