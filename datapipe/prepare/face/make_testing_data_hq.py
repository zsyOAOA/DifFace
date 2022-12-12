#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-16 12:42:42

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import os
import argparse
from utils import util_common

parser = argparse.ArgumentParser()
parser.add_argument("--files_txt", type=str, default='', help="File names")
parser.add_argument("--num_images", type=int, default=3000, help="Number of trainging iamges")
parser.add_argument("--save_dir", type=str, default='', help="Folder to save the fake iamges")
args = parser.parse_args()

files_path = util_common.readline_txt(args.files_txt)
print(f'Number of images in txt file: {len(files_path)}')

assert len(files_path) >= args.num_images
files_path = files_path[:args.num_images]

if not Path(args.save_dir).exists():
    Path(args.save_dir).mkdir(parents=False)

for path in files_path:
    commond = f'cp {path} {args.save_dir}'
    os.system(commond)
