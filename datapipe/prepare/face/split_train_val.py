#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-05-18 09:18:02

import random
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(prog='Face dataset Generation')
parser.add_argument('--face_dir', default='/home/jupyter/data/FFHQ/images512x512', type=str,
                                                metavar='PATH', help="Path to save the face images")
# FFHQ: png, Celeba: png
parser.add_argument('--prefix', default='ffhq', type=str, help="Image format of the HR face images")
parser.add_argument('--num_val', default=500, type=int, help="Ratio for Validation set")
parser.add_argument('--seed', default=1234, type=int, help="Random seed")
parser.add_argument('--im_size', default=512, type=int, help="Random seed")
args = parser.parse_args()

base_dir = Path(__file__).resolve().parents[2] / 'files_txt'
if not base_dir.exists():
    base_dir.mkdir()

path_list = sorted([str(x.resolve()) for x in Path(args.face_dir).glob('*.png')])

file_path = base_dir / f"{args.prefix}{args.im_size}.txt"
if file_path.exists():
    file_path.unlink()
with open(file_path, mode='w') as ff:
    for line in path_list: ff.write(line+'\n')

random.seed(args.seed)
random.shuffle(path_list)
num_train = int(len(path_list) - args.num_val)

file_path_train = base_dir / f"{args.prefix}{args.im_size}_train.txt"
if file_path_train.exists():
    file_path_train.unlink()
with open(file_path_train, mode='w') as ff:
    for line in path_list[:num_train]: ff.write(line+'\n')

file_path_val = base_dir / f"{args.prefix}{args.im_size}_val.txt"
if file_path_val.exists():
    file_path_val.unlink()
with open(file_path_val, mode='w') as ff:
    for line in path_list[num_train:]: ff.write(line+'\n')

print('Train / Validation: {:d}/{:d}'.format(num_train, len(path_list)-num_train))

