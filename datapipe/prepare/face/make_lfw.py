#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-19 12:32:34

import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default="./testdata/LFW-Test",
                                  help="Folder to save the LR images")
parser.add_argument("--data_dir", type=str, default="./testdata/lfw",
                                  help="LFW Testing dataset")
parser.add_argument("--txt_file", type=str, default="./testdata/peopleDevTest.txt",
                                  help="LFW Testing data file paths")
args = parser.parse_args()

with open(args.txt_file, 'r') as ff:
    file_dirs = [x.split('\t')[0] for x in ff.readlines()][1:]

if not Path(args.save_dir).exists():
    Path(args.save_dir).mkdir(parents=True)

for current_dir in file_dirs:
    current_dir = Path(args.data_dir) / current_dir
    file_path = sorted([str(x) for x in current_dir.glob('*.jpg')])[0]
    commond = f'cp {file_path} {args.save_dir}'
    os.system(commond)

num_images = len([x for x in Path(args.save_dir).glob('*.jpg')])
print(f'Number of images: {num_images}')

