#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-05-18 07:58:01

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import argparse
import multiprocessing
import albumentations as Aug
from utils import util_image

parser = argparse.ArgumentParser(prog='SISR dataset Generation')
parser.add_argument('--face_dir', default='/home/jupyter/data/FFHQ/images1024x1024', type=str,
                                            metavar='PATH', help="Path to save the HR face images")
parser.add_argument('--save_dir', default='/home/jupyter/data/FFHQ/', type=str,
                                       metavar='PATH', help="Path to save the resized face images")
# FFHQ: png
parser.add_argument('--ext', default='png', type=str, help="Image format of the HR face images")
parser.add_argument('--pch_size', default=512, type=int, metavar='PATH', help="Cropped patch size")
args = parser.parse_args()

# check the floder to save the cropped patches
pch_size = args.pch_size
pch_dir = Path(args.face_dir).parent / f"images{pch_size}x{pch_size}"
if not pch_dir.exists(): pch_dir.mkdir(parents=False)

transform = Aug.Compose([Aug.SmallestMaxSize(max_size=pch_size),])

# HR face path
path_hr_list = [x for x in Path(args.face_dir).glob('*.'+args.ext)]

def process(im_path):
    im = util_image.imread(im_path, chn='rgb', dtype='uint8')
    pch = transform(image=im)['image']
    pch_path = pch_dir / (im_path.stem + '.png')
    util_image.imwrite(pch, pch_path, chn='rgb')

num_workers = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_workers)
pool.imap(func=process, iterable=path_hr_list, chunksize=16)
pool.close()
pool.join()

num_pch = len([x for x in pch_dir.glob('*.png')])
print('Totally process {:d} images'.format(num_pch))
