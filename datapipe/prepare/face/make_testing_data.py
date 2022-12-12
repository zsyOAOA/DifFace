#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-16 12:11:42

import sys
import pickle
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
parser.add_argument("--save_dir", type=str, default='', help="Folder to save the testing data")
parser.add_argument("--files_txt", type=str, default='', help="ffhq or celeba")
parser.add_argument("--seed", type=int, default=10000, help="Random seed")
args = parser.parse_args()

############################ ICLR ####################################################
# qf_list = [30, 40, 50, 60, 70]  # quality factor for jpeg compression
# sf_list = [4, 8, 16, 24, 30]    # scale factor for upser-resolution
# nf_list = [1, 5, 10, 15, 20]    # noise level for gaussian noise
# sig_list = [2, 4, 6, 8, 10, 12, 14] # sigma for gaussian kernel
# theta_list = [x*math.pi for x in [0, 0.25, 0.5, 0.75]]  # angle for gaussian kernel
######################################################################################

############################ Journal #################################################
qf_list = [30, 40, 50, 60, 70]  # quality factor for jpeg compression
nf_list = [1, 5, 10, 15, 20]    # noise level for gaussian noise
sig_list = [4, 8, 12, 16] # sigma for gaussian kernel
theta_list = [x*math.pi for x in [0, 0.25, 0.5, 0.75]]  # angle for gaussian kernel
sf_list = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]    # scale factor for upser-resolution
############################ ICLR ####################################################

num_val = len(qf_list) * len(sf_list) * len(nf_list) * len(sig_list) * len(theta_list)

# setting seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# checking save_dir
lq_dir = Path(args.save_dir) / "lq"
hq_dir = Path(args.save_dir) / "hq"
info_dir = Path(args.save_dir) / "split_infos"

util_common.mkdir(lq_dir, delete=True)
util_common.mkdir(hq_dir, delete=True)
util_common.mkdir(info_dir, delete=True)

files_path = util_common.readline_txt(args.files_txt)
assert num_val <= len(files_path)
print(f'Number of images in validation: {num_val}')

sf_split = {}
for sf in sf_list:
    sf_split[f"sf{sf}"] = []

num_iters = 0
for qf in qf_list:
    for sf in sf_list:
        for nf in nf_list:
            for sig_x in sig_list:
                for theta in theta_list:
                    if (num_iters+1) %  100 == 0:
                        print(f'Processing: {num_iters+1}/{num_val}')
                    im_gt_path = files_path[num_iters]
                    im_gt = util_image.imread(im_gt_path, chn='bgr', dtype='float32')

                    sig_y = random.choice(sig_list)
                    im_lq = face_degradation(
                            im_gt,
                            sf=sf,
                            sig_x=sig_x,
                            sig_y=sig_y,
                            theta=theta,
                            qf=qf,
                            nf=nf,
                            )

                    im_name = Path(im_gt_path).name

                    sf_split[f"sf{sf}"].append(im_name)

                    im_save_path = lq_dir / im_name
                    util_image.imwrite(im_lq, im_save_path, chn="bgr", dtype_in='float32')

                    im_save_path = hq_dir / im_name
                    util_image.imwrite(im_gt, im_save_path, chn="bgr", dtype_in='float32')

                    num_iters += 1

info_path = info_dir / 'sf_split.pkl'
with open(str(info_path), mode='wb') as ff:
    pickle.dump(sf_split, ff)

