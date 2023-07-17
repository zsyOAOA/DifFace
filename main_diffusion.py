#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-05-19 15:15:17

import argparse
from trainer import TrainerDiffusionFace as Trainer
from omegaconf import OmegaConf

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--save_dir", type=str, default="./save_dir",
                                             help="Folder to save the checkpoints and training log")
    parser.add_argument("--resume", type=str, const=True, default="", nargs="?",
                                                      help="resume from the save_dir or checkpoint")
    parser.add_argument("--cfg_path", type=str, default="./configs/inpainting_debug.yaml",
                                                                        help="Configs of yaml file")
    parser.add_argument("--seed", type=int, default=10000, help="Random seed")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_parser()

    configs = OmegaConf.load(args.cfg_path)

    # merge args to config
    for key in vars(args):
        configs[key] = getattr(args, key)

    trainer = Trainer(configs)
    trainer.train()
