#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-02 20:43:41

import os
import torch
import argparse
import numpy as np
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf
from skimage import img_as_ubyte

from utils import util_opts
from utils import util_image
from utils import util_common

from sampler import DifFaceSampler
from ResizeRight.resize_right import resize
from basicsr.utils.download_util import load_file_from_url

_START_TIMESTEPS = {'restoration': 100, 'inpainting': 120}
_GAMMA = {'restoration': 0.0, 'inpainting': 0.5}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-i",
            "--in_path",
            type=str,
            default='./testdata/cropped_faces',
            help='Folder to save the low quality image',
            )
    parser.add_argument(
            "-o",
            "--out_path",
            type=str,
            default='./results',
            help='Folder to save the restored results',
            )
    parser.add_argument(
            "--aligned",
            action='store_true',
            help='Input are alinged faces',
            )
    parser.add_argument(
            "--use_fp16",
            action='store_true',
            help='Activate float16 for inference',
            )
    parser.add_argument(
            "--task",
            type=str,
            default='restoration',
            choices=['restoration', 'inpainting'],
            help='Task',
            )
    parser.add_argument(
            "--eta",
            type=float,
            default=0.5,
            help='Hyper-parameter eta in ddim',
            )
    parser.add_argument(
            "--bs",
            type=int,
            default=1,
            help='Batch size for inference',
            )
    parser.add_argument(
            "--seed",
            type=int,
            default=12345,
            help='Random Seed',
            )
    parser.add_argument(
            "--draw_box",
            action='store_true',
            help='Draw box for face in the unaligned case',
            )
    args = parser.parse_args()

    # configurations
    if args.task == 'restoration':
        cfg_path = 'configs/sample/iddpm_ffhq512_swinir.yaml'
    elif args.task == 'inpainting':
        cfg_path = 'configs/sample/difface_inpainting_lama256.yaml'
    else:
        raise ValueError("Only accept task types of 'restoration' and 'inpainting'!")

    # setting configurations
    configs = OmegaConf.load(cfg_path)
    configs.seed = args.seed
    configs.diffusion.params.timestep_respacing = 'ddim250'

    # prepare the checkpoint
    if args.task == 'restoration':
        if not Path(configs.model_ir.ckpt_path).exists():
            load_file_from_url(
                url="https://github.com/zsyOAOA/DifFace/releases/download/V1.0/swinir_restoration512_L1.pth",
                model_dir=str(Path(configs.model_ir.ckpt_path).parent),
                progress=True,
                file_name=Path(configs.model_ir.ckpt_path).name,
                )
        if not Path(configs.model.ckpt_path).exists():
            load_file_from_url(
                url="https://github.com/zsyOAOA/DifFace/releases/download/V1.0/iddpm_ffhq512_ema500000.pth",
                model_dir=str(Path(configs.model.ckpt_path).parent),
                progress=True,
                file_name=Path(configs.model.ckpt_path).name,
                )
        configs.aligned = args.aligned
    elif args.task == 'inpainting':
        if not Path(configs.model_ir.ckpt_path).exists():
            load_file_from_url(
                url="https://github.com/zsyOAOA/DifFace/releases/download/V1.0/lama_inpainting256.pth",
                model_dir=str(Path(configs.model_ir.ckpt_path).parent),
                progress=True,
                file_name=Path(configs.model_ir.ckpt_path).name,
                )
        if not Path(configs.model.ckpt_path).exists():
            load_file_from_url(
                url="https://github.com/zsyOAOA/DifFace/releases/download/V1.0/iddpm_ffhq256_ema750000.pth",
                model_dir=str(Path(configs.model.ckpt_path).parent),
                progress=True,
                file_name=Path(configs.model.ckpt_path).name,
                )
        configs.aligned = True
    else:
        raise ValueError("Only accept task types of 'restoration' and 'inpainting'!")

    if not configs.aligned and args.bs != 1:
        args.bs = 1
        print("Resetting batchsize to be 1 for unaligned case.")

    # build the sampler for diffusion
    sampler_dist = DifFaceSampler(
            configs,
            im_size=configs.model.params.image_size,
            use_fp16=args.use_fp16,
            )
    sampler_dist.inference(
            in_path=args.in_path,
            out_path=args.out_path,
            bs=args.bs,
            start_timesteps=_START_TIMESTEPS[args.task],
            task=args.task,
            need_restoration=True,
            gamma=_GAMMA[args.task],
            num_update=1,
            draw_box=args.draw_box,
            suffix=None,
            eta=args.eta if args.task =='restoration' else 1.0,
            mask_back=True,
            )

if __name__ == '__main__':
    main()
