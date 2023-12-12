#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-16 12:11:42

import sys
import pickle
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import math
import torch
import random
import argparse
import numpy as np
from omegaconf import OmegaConf

from utils import util_image
from utils import util_common

from datapipe.masks import MixedMaskGenerator

parser = argparse.ArgumentParser()
parser.add_argument(
        "-o",
        "--save_dir",
        type=str,
        default='',
        help="Folder to save the testing data",
        )
parser.add_argument(
        "-i",
        "--in_dir",
        type=str,
        default='',
        help="Path to save the source images",
        )
parser.add_argument("--num_val", type=int, default=2000, help="Random seed")
parser.add_argument("--seed", type=int, default=12345, help="Random seed")
args = parser.parse_args()

# setting seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# checking save_dir
lq_dir = Path(args.save_dir) / "lq"
mask_dir = Path(args.save_dir) / "mask"
hq_dir = Path(args.save_dir) / "hq"
info_dir = Path(args.save_dir) / "infos"

util_common.mkdir(lq_dir, delete=True)
util_common.mkdir(hq_dir, delete=True)
util_common.mkdir(mask_dir, delete=True)
util_common.mkdir(info_dir, delete=True)

files_path = [x for x in Path(args.in_dir).glob("*.png")]
assert args.num_val <= len(files_path)
files_path = files_path[:args.num_val]
print(f'Number of images in validation: {len(files_path)}')

cfg_path = str(Path(__file__).parents[1] / 'configs' / 'training' / 'estimator_lama_inpainting.yaml')
configs = OmegaConf.load(cfg_path)

mask_types = ['box', 'irregular', 'expand', 'half']
mask_split = {}

num_val_each_type = 500
num_types = len(mask_types)
assert args.num_val == (num_types * num_val_each_type)
for current_mask_type in mask_types:
    mask_split[current_mask_type] = []

for ii, im_path in enumerate(files_path):
    if (ii+1) % 100 == 0:
        print(f'processing {ii+1}/{len(files_path)}...')
    im_name = Path(im_path).name

    im_gt = util_image.imread(im_path, chn='bgr', dtype='float32')
    if ii == 0:
        mask_generator = MixedMaskGenerator(
                box_proba=1, box_kwargs={
                    'margin': 10,
                    'bbox_min_size': 30,
                    'bbox_max_size': 150,
                    'max_times': 4,
                    'min_times': 1,
                    },
                 irregular_proba=0, irregular_kwargs=None,
                 squares_proba=0, squares_kwargs=None,
                 superres_proba=0, superres_kwargs=None,
                 outpainting_proba=0, outpainting_kwargs=None,
                 expand_proba=0, expand_kwargs=None,
                 half_proba=0, half_kwargs=None,
                 alterline_proba=0, invert_proba=0,
                 )
    elif ii == 1 * num_val_each_type:
        mask_generator = MixedMaskGenerator(
                 irregular_proba=1,
                 irregular_kwargs={
                            'max_angle': 4,
                            'max_len': 200,
                            'max_width': 100,
                            'max_times': 5,
                            'min_times': 1
                             },
                 box_proba=0, box_kwargs=None,
                 squares_proba=0, squares_kwargs=None,
                 superres_proba=0, superres_kwargs=None,
                 outpainting_proba=0, outpainting_kwargs=None,
                 expand_proba=0, expand_kwargs=None,
                 half_proba=0, half_kwargs=None,
                 alterline_proba=0, invert_proba=0,
                 )
    elif ii == 2 * num_val_each_type:
        mask_generator = MixedMaskGenerator(
                 expand_proba=1,
                 expand_kwargs={'masking_percent': 0.50, 'center': True},
                 irregular_proba=0, irregular_kwargs=None,
                 box_proba=0, box_kwargs=None,
                 squares_proba=0, squares_kwargs=None,
                 superres_proba=0, superres_kwargs=None,
                 outpainting_proba=0, outpainting_kwargs=None,
                 half_proba=0, half_kwargs=None,
                 alterline_proba=0, invert_proba=0,
                 )
    elif ii == 3 * num_val_each_type:
        mask_generator = MixedMaskGenerator(
                 half_proba=1, half_kwargs={'masking_percent': 0.50},
                 expand_proba=0, expand_kwargs=None,
                 irregular_proba=0, irregular_kwargs=None,
                 box_proba=0, box_kwargs=None,
                 squares_proba=0, squares_kwargs=None,
                 superres_proba=0, superres_kwargs=None,
                 outpainting_proba=0, outpainting_kwargs=None,
                 alterline_proba=0, invert_proba=0,
                 )

    mask = mask_generator(im_gt.transpose(2,0,1), iter_i=ii+1).transpose(1,2,0)   # h x w x 1, mask area: 1, unmask area: 0
    im_masked = im_gt * (1 - mask)

    im_save_path = lq_dir / im_name
    util_image.imwrite(im_masked, im_save_path, chn="bgr", dtype_in='float32')

    im_save_path = hq_dir / im_name
    util_image.imwrite(im_gt, im_save_path, chn="bgr", dtype_in='float32')

    im_save_path = mask_dir / im_name
    util_image.imwrite(mask.squeeze(-1), im_save_path, chn="bgr", dtype_in='float32')

    if ii < num_val_each_type * 1:
        mask_split['box'].append(im_name)
    elif ii < num_val_each_type * 2:
        mask_split['irregular'].append(im_name)
    elif ii < num_val_each_type * 3:
        mask_split['expand'].append(im_name)
    else:
        mask_split['half'].append(im_name)

info_path = info_dir / 'mask_split.pkl'
with open(str(info_path), mode='wb') as ff:
    pickle.dump(mask_split, ff)

# writing txt file
for current_mask_type in mask_types:
    txt_path = info_dir / f"{current_mask_type}.txt"
    if txt_path.exists():
        txt_path.unlink()
    with open(txt_path, mode='w') as ff:
        for line in mask_split[current_mask_type]:
            ff.write(line+'\n')

