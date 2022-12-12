#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-10-17 15:44:17

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[3]))

import random
import argparse
import albumentations
from degradation_bsrgan.utils_blindsr import degradation_bsrgan

from utils import util_image
from utils import util_common

parser = argparse.ArgumentParser(prog='Making testing data')
parser.add_argument(
        '--data_dir',
        default='/mnt/vdb/data/ImageNet/val/',
        type=str,
        help="Path to save the Imagenet validation dataset",
        )
parser.add_argument(
        '--save_dir',
        default='/mnt/vdb/IRDiff/SR/testing_data/imagenet256/',
        type=str,
        help="Path to save the generated images",
        )
parser.add_argument(
        '--hq_size',
        default=256,
        type=int,
        help="High quality image resolution, (default: 256)",
        )
parser.add_argument(
        '--sf',
        default=4,
        type=int,
        help="Scale factor",
        )
parser.add_argument(
        '--ext',
        default='JPEG',
        type=str,
        help="Image format",
        )
parser.add_argument(
        '-n',
        '--num_images',
        default=3000,
        type=int,
        help="Number of images in maked testing dataset",
        )
args = parser.parse_args()

random.seed(10000)
preprocessor = albumentations.Compose([
        albumentations.SmallestMaxSize(max_size=args.hq_size),
        albumentations.CenterCrop(height=args.hq_size, width=args.hq_size),
    ])

path_list = sorted([x for x in Path(args.data_dir).glob(f"*.{args.ext}")])
path_list = random.sample(path_list, args.num_images)

hq_dir = Path(args.save_dir) / 'hq'
util_common.mkdir(hq_dir, delete=True, parents=True)
lq_dir = Path(args.save_dir) / 'lq'
util_common.mkdir(lq_dir, delete=True, parents=True)

for ii, im_path in enumerate(path_list):
    if (ii+1) % 100 == 0:
        print(f'Processing: {ii+1}/{len(path_list)}...')

    im_hq = util_image.imread(im_path, dtype='float32', chn='rgb')
    im_hq = preprocessor(image=im_hq)['image']

    # degradation
    im_lq, im_hq = degradation_bsrgan(im_hq, args.sf, int(args.hq_size / args.sf), isp_model=None)

    im_name = im_path.stem

    im_path = hq_dir / f"{im_name}.png"
    util_image.imwrite(im_hq, im_path, dtype_in='float32', chn='rgb')

    im_path = lq_dir / f"{im_name}.png"
    util_image.imwrite(im_lq, im_path, dtype_in='float32', chn='rgb')

