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

parser = argparse.ArgumentParser(prog='SISR dataset Generation')
parser.add_argument(
        '--DIV2K_dir',
        type=str,
        metavar='PATH',
        default='/mnt/vdb/data/DIV2K/DIV2K_train_HR',
        help="Path to save the HR images of DIV2K, (default: None)",
        )
parser.add_argument(
        '--Flickr_dir',
        type=str,
        metavar='PATH',
        default='/mnt/vdb/data/Flickr2K/Flickr2K_HR/',
        help="Path to save the HR images of Flickr2K, (default: None)",
        )
parser.add_argument(
        '--save_dir',
        type=str,
        metavar='PATH',
        default='/mnt/vdb/data/DF2K_pch512',
        help="Path to save the HR images of Flickr2K, (default: None)",
        )
parser.add_argument(
        '--pch_size',
        default=512,
        type=int,
        metavar='PATH',
        help="Cropped patch size, (default: None)",
        )
parser.add_argument(
        '--stride',
        default=256,
        type=int,
        metavar='PATH',
        help="Stride for cropping",
        )
args = parser.parse_args()

# check the floder to save the cropped patches
pch_dir = Path(args.save_dir)
if not pch_dir.exists():
    pch_dir.mkdir()

pch_size = args.pch_size
stride = args.stride

path_hr_list = [[x, 'div2k'] for x in Path(args.DIV2K_dir).glob('*.png')]
path_hr_list.extend([[x, 'flickr'] for x in Path(args.Flickr_dir).glob('*.png')])
print('Number of HR images: {:d}'.format(len(path_hr_list)))

def crop_image(path_info):
    '''
    Crop the image into small patches.
    '''
    im_path, prefix = path_info
    im = cv2.imread(str(im_path), cv2.IMREAD_UNCHANGED)
    h, w = im.shape[:2]
    ind_h = list(range(0, h-pch_size, stride)) + [h-pch_size,]
    ind_w = list(range(0, w-pch_size, stride)) + [w-pch_size,]
    num_pch = 0
    for start_h in ind_h:
        for start_w in ind_w:
            num_pch += 1
            pch = im[start_h:start_h+pch_size, start_w:start_w+pch_size, ]
            pch_name = prefix + '_' + im_path.stem + '_{:05d}.png'.format(num_pch)
            cv2.imwrite(str(pch_dir/pch_name), pch)

num_workers = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_workers)
pool.imap(func=crop_image, iterable=path_hr_list, chunksize=16)
pool.close()
pool.join()

num_pch = len([x for x in pch_dir.glob('*.png')])
print('Total {:d} small patches in training set'.format(num_pch))  # 512-->84726, 256-->516792

