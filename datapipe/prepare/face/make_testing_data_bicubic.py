#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-16 12:42:42

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import os
import math
import torch
import argparse
from einops import rearrange
from datapipe.datasets import DatasetBicubic

from utils import util_image
from utils import util_common

parser = argparse.ArgumentParser()
parser.add_argument(
        "--files_txt",
        type=str,
        default='./datapipe/files_txt/celeba512_val.txt',
        help="File names")
parser.add_argument(
        "--sf",
        type=int,
        default=8,
        help="Number of trainging iamges",
        )
parser.add_argument(
        "--bs",
        type=int,
        default=8,
        help="Batch size",
        )
parser.add_argument(
        "--save_dir",
        type=str,
        default='',
        help="Folder to save the fake iamges",
        )
parser.add_argument(
        "--num_images",
        type=int,
        default=100,
        help="Number of iamges",
        )
args = parser.parse_args()

save_dir = Path(args.save_dir)
if not save_dir.stem.endswith(f'x{args.sf}'):
    save_dir = save_dir.parent / f"{save_dir.stem}_x{args.sf}"
util_common.mkdir(save_dir, delete=True)

dataset = DatasetBicubic(
        files_txt=args.files_txt,
        up_back=True,
        need_gt_path=True,
        sf=args.sf,
        length=args.num_images,
        )
dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.bs,
        drop_last=False,
        num_workers=4,
        pin_memory=False,
        )

for ii, data_batch in enumerate(dataloader):
    im_lq_batch = data_batch['lq']
    im_path_batch = data_batch['gt_path']
    print(f"Processing: {ii+1}/{math.ceil(len(dataset) / args.bs)}...")

    for jj in range(im_lq_batch.shape[0]):
        im_lq = rearrange(
            im_lq_batch[jj].clamp(0.0, 1.0).numpy(),
            'c h w -> h w c',
                )
        im_name = Path(im_path_batch[jj]).name
        im_path = save_dir / im_name
        util_image.imwrite(im_lq, im_path, chn='rgb', dtype_in='float32')

