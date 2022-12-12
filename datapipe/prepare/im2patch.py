#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-18 10:26:59

import os
import cv2
import argparse
import numpy as np
from math import ceil
import multiprocessing
from pathlib import Path

parser = argparse.ArgumentParser(prog='Patch Generation')
parser.add_argument('--big_dir', default='/home/jupyter/data/DIV2K/DIV2K_train_HR', type=str,
                                                                 help="Path to save the big images")
parser.add_argument('--small_dir', default='/home/jupyter/data/DIV2K/train_patch512', type=str,
                                                              help="Path to save the image patches")
parser.add_argument('--pch_size', default=512, type=int, help="Cropped patch size, (default: None)")
parser.add_argument('--stride', default=128, type=int, help="Stride for cropping")
parser.add_argument('--ext', default='png', type=str, help="Image format")
args = parser.parse_args()

pch_size = args.pch_size
stride = args.stride

# check the pch dir
pch_dir = Path(args.small_dir)
if not pch_dir.exists():
    pch_dir.mkdir()

path_hr_list = [x for x in Path(args.big_dir).glob('*.'+args.ext)]
print('Number of HR images: {:d}'.format(len(path_hr_list)))

def crop_image(im_path):
    '''
    Crop the image into small patches.
    '''
    im = cv2.imread(str(im_path), cv2.IMREAD_UNCHANGED)
    h, w = im.shape[:2]
    ind_h = list(range(0, h-pch_size, stride)) + [h-pch_size,]
    ind_w = list(range(0, w-pch_size, stride)) + [w-pch_size,]
    num_pch = 0
    for start_h in ind_h:
        for start_w in ind_w:
            num_pch += 1
            pch = im[start_h:start_h+pch_size, start_w:start_w+pch_size, ]
            pch_name = im_path.stem + '_pch_{:05d}.png'.format(num_pch)
            cv2.imwrite(str(pch_dir/pch_name), pch)

num_workers = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_workers)
pool.imap(func=crop_image, iterable=path_hr_list, chunksize=16)
pool.close()
pool.join()

num_pch = len([x for x in pch_dir.glob('*.png')])
print('Total {:d} small patches!'.format(num_pch))

