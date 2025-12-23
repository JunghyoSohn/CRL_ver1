# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Tool for creating ZIP/PNG based datasets."""

import functools
import gzip
import io
import json
import os
import pickle
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import click
import numpy as np
import PIL.Image
from tqdm import tqdm

import torch
from torchvision.utils import save_image

from collections import OrderedDict

#----------------------------------------------------------------------------
# Parse a 'M,N' or 'MxN' integer tuple.
# Example: '4x2' returns (4,2)

def parse_tuple(s: str) -> Tuple[int, int]:
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise click.ClickException(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION

#----------------------------------------------------------------------------

def open_latents(source_dir, *, max_images: Optional[int]):

    data_parts = [np.load(source_dir + f'/{x}.npy', mmap_mode='r') for x in range(max_images // 1024 + 1)]
    data = np.concatenate(data_parts)
    input_images = data[:max_images, :, :, :]
    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        for idx, img in enumerate(input_images):
            # Check if the file is a .npy file

            yield dict(img=img, label=None)

            if idx >= max_idx - 1:
                break

    return max_idx, iterate_images()


def open_causalcircuit(source_dir, *, max_images: Optional[int], tag='train', paired=None):
    if tag == 'train':
        filenames = [Path(source_dir) / f"{tag}-{i}.npz" for i in range(10)]
    else:
        filenames = [Path(source_dir) / f"{tag}.npz"]

    data_parts = []
    for filename in filenames:
        assert filename.exists(), f"Dataset not found at {filename}. Consult README.md."
        data_parts.append(dict(np.load(filename)))

    data = {k: np.concatenate([data[k] for data in data_parts]) for k in data_parts[0]}
    input_images = data["imgs"]
    labels = data['intervention_masks'].astype(int)
    max_idx = maybe_min(len(input_images), max_images)

    if paired=='pre':
        def iterate_images():
            for idx, fname in enumerate(input_images):
                # Check if the file is a .npy file
                buffer = io.BytesIO()
                for index in [0]:
                    buffer.write(fname[index])
                    img = np.array(PIL.Image.open(buffer))

                yield dict(img=img, label=None)
                if idx >= max_idx - 1:
                    break
        return max_idx, iterate_images()

    elif paired == 'post':
        def iterate_images():
            for idx, fname in enumerate(input_images):
                # Check if the file is a .npy file
                buffer = io.BytesIO()
                for index in [1]:
                    buffer.write(fname[index])
                    img = np.array(PIL.Image.open(buffer))

                yield dict(img=img, label=labels[idx].tolist())
                if idx >= max_idx - 1:
                    break
        return max_idx, iterate_images()

    else:
        def iterate_images():
            for idx, fname in enumerate(input_images):
                # Check if the file is a .npy file
                for index in [0, 1]:
                    buffer = io.BytesIO()
                    buffer.write(fname[index])
                    img = np.array(PIL.Image.open(buffer))
                    yield dict(img=img, label=None)
                    if idx >= max_idx - 1:
                        break

        return max_idx, iterate_images()

def open_image_folder(source_dir, *, max_images: Optional[int]):
    input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if is_image_ext(f) and os.path.isfile(f)]
    arch_fnames = {fname: os.path.relpath(fname, source_dir).replace('\\', '/') for fname in input_images}
    max_idx = maybe_min(len(input_images), max_images)

    # Load labels.
    labels = dict()
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            data = json.load(file)['labels']
            if data is not None:
                labels = {x[0]: x[1] for x in data}

    # No labels available => determine from top-level directory names.
    if len(labels) == 0:
        toplevel_names = {arch_fname: arch_fname.split('/')[0] if '/' in arch_fname else '' for arch_fname in arch_fnames.values()}
        toplevel_indices = {toplevel_name: idx for idx, toplevel_name in enumerate(sorted(set(toplevel_names.values())))}
        if len(toplevel_indices) > 1:
            labels = {arch_fname: toplevel_indices[toplevel_name] for arch_fname, toplevel_name in toplevel_names.items()}

    def iterate_images():
        for idx, fname in enumerate(input_images):
            # Check if the file is a .npy file
            img = np.array(PIL.Image.open(fname))

            yield dict(img=img, label=labels.get(arch_fnames.get(fname)))
            if idx >= max_idx - 1:
                break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int]
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), PIL.Image.Resampling.LANCZOS)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        if img.ndim == 2:
            img = img[:, :, np.newaxis].repeat(3, axis=2)
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        if img.ndim == 2:
            img = img[:, :, np.newaxis].repeat(3, axis=2)
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if output_width is None or output_height is None:
            raise click.ClickException('must specify --resolution=WxH when using ' + transform + 'transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if output_width is None or output_height is None:
            raise click.ClickException('must specify --resolution=WxH when using ' + transform + ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    assert False, 'unknown transform'

#----------------------------------------------------------------------------

def open_dataset(source, *, max_images: Optional[int], paired=None):
    if os.path.isdir(source):
        ### original Image (png)
        if os.path.basename(source) == ('causalcircuit_original'):
            return open_causalcircuit(source, max_images=max_images, tag='train', paired='pre')

        ### Latent Image (npy)
        elif os.path.basename(source) == ('cc_pre_19'):
            return open_latents(source, max_images=100000)

        elif os.path.basename(source) == ('cc_post_19'):
            return open_latents(source, max_images=100000)

        elif os.path.basename(source) == ('cc_pre_79'):
            return open_latents(source, max_images=100000)

        elif os.path.basename(source) == ('cc_post_79'):
            return open_latents(source, max_images=100000)
        else:
            return open_image_folder(source, max_images=max_images)
    else:
        raise click.ClickException(f'Missing input file or directory: {source}')

#----------------------------------------------------------------------------

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            raise click.ClickException('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

#----------------------------------------------------------------------------

datapath = {'image_cc': '/causalcircuit_original',
            'image_cc_pre': '/causalcircuit_original',
            'image_cc_post': '/causalcircuit_original',

            'cc_pre_19': '/original/latents/cc_pre_19',
            'cc_post_19': '/original/latents/cc_post_19',

            'cc_pre_79': '/original/latents/cc_pre_79',
            'cc_post_79': '/original/latents/cc_post_79'
    }

outpath = { 'image_cc': '/image/cc',
            'image_cc_pre': '/image/cc_pre',
            'image_cc_post': '/image/cc_post',

            'cc_pre_19': '/latent_19/cc_pre_19.zip',
            'cc_post_19': '/latent_19/cc_post_19.zip',

            'cc_pre_79': '/latent_79/cc_pre_79.zip',
            'cc_post_79': '/latent_79/cc_post_79.zip'
    }
@click.command()
@click.option('--directory',  help='Input directory or archive name', metavar='PATH',   type=str, default='./causal_datasets')
@click.option('--dataname',     help='Input directory or archive name', metavar='PATH',
              type=click.Choice(['image_cc','image_cc_pre', 'image_cc_post',
                                 'cc_pre_19', 'cc_post_19','cc_pre_79', 'cc_post_79']), default='cc_post_19')
@click.option('--max-images', help='Maximum number of images to output', metavar='INT', type=int)
@click.option('--transform',  help='Input crop/resize mode', metavar='MODE',            type=click.Choice(['center-crop', 'center-crop-wide']))
@click.option('--resolution', help='Output resolution (e.g., 512x512)', metavar='WxH',  type=parse_tuple, default='64x64')
@click.option('--save_png',  help='Output .png or .npy', metavar='BOOL',   default=False)

def main(
    directory: str,
    dataname: str,
    # dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resolution: Optional[Tuple[int, int]],
    save_png: bool
):
    source = directory + datapath[dataname]
    dest = directory + outpath[dataname]
    if dataname == 'image_cc_pre':
        paired = 'pre'
    elif dataname == 'image_cc_post':
        paired = 'post'
    else:
        paired = None

    PIL.Image.init()

    if dest == '':
        raise click.ClickException('--dest output filename or directory must not be an empty string')

    num_files, input_iter = open_dataset(source, max_images=max_images, paired=paired)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)

    if resolution is None: resolution = (None, None)
    transform_image = make_transform(transform, *resolution)

    dataset_attrs = None

    labels = []
    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        idx_str = f'{idx:08d}'

        if save_png:
            archive_fname = f'{idx_str[:5]}/img{idx_str}.png'
            img = transform_image(image['img'])
        else:
            archive_fname = f'{idx_str[:5]}/img{idx_str}.npy'
            img = image['img']

        # Apply crop and resize.
        if img is None:
            continue

        # Error check to require uniform image attributes across
        # the whole dataset.
        channels = img.shape[2] if img.ndim == 3 else 1
        cur_image_attrs = {'width': img.shape[1], 'height': img.shape[0], 'channels': channels}
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs['width']
            height = dataset_attrs['height']
            if width != height:
                raise click.ClickException(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
            if dataset_attrs['channels'] not in [1, 3]:
                raise click.ClickException('Input images must be stored as RGB or grayscale')
            if width != 2 ** int(np.floor(np.log2(width))):
                raise click.ClickException('Image width/height after scale and crop are required to be power-of-two')
        elif dataset_attrs != cur_image_attrs:
            err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()]
            raise click.ClickException(f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

        if save_png:
            ### Save the image as an uncompressed PNG.
            img = PIL.Image.fromarray(img, {1: 'L', 3: 'RGB'}[channels])
            image_bits = io.BytesIO()
            img.save(image_bits, format='png', compress_level=0, optimize=False)
            save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
            labels.append([archive_fname, image['label']] if image['label'] is not None else None)

        else:
            ## Save the image as an uncompressed .npy.
            buffer = io.BytesIO()
            np.save(buffer, img)
            buffer.seek(0)
            save_bytes(os.path.join(archive_root_dir, archive_fname), buffer.read())
            labels.append([archive_fname, image['label']] if image['label'] is not None else None)

    metadata = {'labels': labels if all(x is not None for x in labels) else None}
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
