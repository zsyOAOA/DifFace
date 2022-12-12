#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-02-06 10:34:59

import importlib
from pathlib import Path

def mkdir(dir_path, delete=False, parents=True):
    import shutil
    if not isinstance(dir_path, Path):
        dir_path = Path(dir_path)
    if delete:
        if dir_path.exists():
            shutil.rmtree(str(dir_path))
    if not dir_path.exists():
        dir_path.mkdir(parents=parents)

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_filenames(dir_path, exts=['png', 'jpg'], recursive=True):
    '''
    Get the file paths in the given folder.
    param exts: list, e.g., ['png',]
    return: list
    '''
    if not isinstance(dir_path, Path):
        dir_path = Path(dir_path)

    file_paths = []
    for current_ext in exts:
        if recursive:
            file_paths.extend([str(x) for x in dir_path.glob('**/*.'+current_ext)])
        else:
            file_paths.extend([str(x) for x in dir_path.glob('*.'+current_ext)])

    return file_paths

def readline_txt(txt_file):
    if txt_file is None:
        out = []
    else:
        with open(txt_file, 'r') as ff:
            out = [x[:-1] for x in ff.readlines()]
    return out
