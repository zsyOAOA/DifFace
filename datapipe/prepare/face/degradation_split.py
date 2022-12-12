#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-16 12:11:42

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import os
import math
import torch
import random
import argparse
import numpy as np
from einops import rearrange

from utils import util_image
from utils import util_common

from datapipe.face_degradation_testing import face_degradation

parser = argparse.ArgumentParser()
parser.add_argument("--lq_dir", type=str, default='', help="floder for the lq image")
parser.add_argument("--source_txt", type=str, default='', help="ffhq or celeba")
parser.add_argument("--prefix", type=str, default='celeba512', help="Data type")
parser.add_argument("--seed", type=int, default=10000, help="Random seed")
args = parser.parse_args()

qf_list = [30, 40, 50, 60, 70]  # quality factor for jpeg compression
sf_list = [4, 8, 16, 24, 30]    # scale factor for upser-resolution
nf_list = [1, 5, 10, 15, 20]    # noise level for gaussian noise
sig_list = [2, 4, 6, 8, 10, 12, 14] # sigma for gaussian kernel
theta_list = [x*math.pi for x in [0, 0.25, 0.5, 0.75]]  # angle for gaussian kernel
num_val = len(qf_list) * len(sf_list) * len(nf_list) * len(sig_list) * len(theta_list)

# setting seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

files_path = util_common.readline_txt(args.source_txt)
assert num_val <= len(files_path)
print(f'Number of images in validation: {num_val}')

save_dir = Path(args.lq_dir).parent / (Path(args.lq_dir).stem+'_split')
if not save_dir.exists():
    save_dir.mkdir()

for sf_target in sf_list:
    num_iters = 0

    num_sf = 0
    file_path = save_dir / f"{args.prefix}_val_sf{sf_target}.txt"
    if file_path.exists():
        file_path.unlink()
    with open(file_path, mode='w') as ff:
        for qf in qf_list:
            for sf in sf_list:
                for nf in nf_list:
                    for sig_x in sig_list:
                        for theta in theta_list:

                            im_name = Path(files_path[num_iters]).name
                            im_path = str(Path(args.lq_dir).parent / im_name)
                            if sf == sf_target:
                                ff.write(im_path+'\n')
                                num_sf += 1

                            num_iters += 1

    print(f'{num_sf} images for sf: {sf_target}')

