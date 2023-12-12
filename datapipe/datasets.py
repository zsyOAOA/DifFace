import random
import numpy as np
from pathlib import Path
from einops import rearrange

import torch
import torchvision as thv
from torch.utils.data import Dataset

from utils import util_sisr
from utils import util_image
from utils import util_common

from basicsr.data.realesrgan_dataset import RealESRGANDataset
from .ffhq_degradation_dataset import FFHQDegradationDataset
from .masks import MixedMaskGenerator

def get_transforms(transform_type, kwargs):
    '''
    Accepted optins in kwargs.
        mean: scaler or sequence, for nornmalization
        std: scaler or sequence, for nornmalization
        crop_size: int or sequence, random or center cropping
        scale, out_shape: for Bicubic
        min_max: tuple or list with length 2, for cliping
    '''
    if transform_type == 'default':
        transform = thv.transforms.Compose([
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'bicubic_norm':
        transform = thv.transforms.Compose([
            util_sisr.Bicubic(scale=kwargs.get('scale', None), out_shape=kwargs.get('out_shape', None)),
            util_image.Clamper(min_max=kwargs.get('min_max', (0.0, 1.0))),
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'bicubic_back_norm':
        transform = thv.transforms.Compose([
            util_sisr.Bicubic(scale=kwargs.get('scale', None)),
            util_sisr.Bicubic(scale=1/kwargs.get('scale', None)),
            util_image.Clamper(min_max=kwargs.get('min_max', (0.0, 1.0))),
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'aug_crop_norm':
        transform = thv.transforms.Compose([
            util_image.SpatialAug(),
            thv.transforms.ToTensor(),
            thv.transforms.RandomCrop(
                crop_size=kwargs.get('crop_size', None),
                pad_if_needed=True,
                padding_mode='reflect',
                ),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    else:
        raise ValueError(f'Unexpected transform_variant {transform_variant}')
    return transform

def create_dataset(dataset_config):
    if dataset_config['type'] == 'gfpgan':
        dataset = FFHQDegradationDataset(dataset_config['params'])
    elif dataset_config['type'] == 'bicubic':
        dataset = DatasetBicubic(**dataset_config['params'])
    elif dataset_config['type'] == 'folder':
        dataset = BaseDataFolder(**dataset_config['params'])
    elif dataset_config['type'] == 'realesrgan':
        dataset = RealESRGANDataset(dataset_config['params'])
    elif dataset_config['type'] == 'inpainting':
        dataset = InpaintingDataSet(**dataset_config['params'])
    else:
        raise NotImplementedError(dataset_config['type'])

    return dataset

class DatasetBicubic(Dataset):
    def __init__(self,
            files_txt=None,
            val_dir=None,
            ext='png',
            sf=None,
            up_back=False,
            need_gt_path=False,
            length=None):
        super().__init__()
        if val_dir is None:
            self.files_names = util_common.readline_txt(files_txt)
        else:
            self.files_names = [str(x) for x in Path(val_dir).glob(f"*.{ext}")]
        self.sf = sf
        self.up_back = up_back
        self.need_gt_path = need_gt_path

        if length is None:
            self.length = len(self.files_names)
        else:
            self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        im_path = self.files_names[index]
        im_gt = util_image.imread(im_path, chn='rgb', dtype='float32')
        im_lq = util_image.imresize_np(im_gt, scale=1/self.sf)
        if self.up_back:
            im_lq = util_image.imresize_np(im_lq, scale=self.sf)

        im_lq = rearrange(im_lq, 'h w c -> c h w')
        im_lq = torch.from_numpy(im_lq).type(torch.float32)

        im_gt = rearrange(im_gt, 'h w c -> c h w')
        im_gt = torch.from_numpy(im_gt).type(torch.float32)

        if self.need_gt_path:
            return {'lq':im_lq, 'gt':im_gt, 'gt_path':im_path}
        else:
            return {'lq':im_lq, 'gt':im_gt}

class BaseDataFolder(Dataset):
    def __init__(
            self,
            dir_path,
            transform_type,
            transform_kwargs=None,
            dir_path_extra=None,
            length=None,
            need_path=False,
            im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            recursive=False,
            ):
        super(BaseDataFolder, self).__init__()

        file_paths_all = util_common.scan_files_from_folder(dir_path, im_exts, recursive)
        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all

        self.length = length
        self.need_path = need_path
        self.dir_path_extra = dir_path_extra
        self.transform = get_transforms(transform_type, transform_kwargs)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im = util_image.imread(im_path, chn='rgb', dtype='float32')
        im = self.transform(im)
        out_dict = {'image':im, 'lq':im}

        if self.dir_path_extra is not None:
            im_path_extra = Path(self.dir_path_extra) / Path(im_path).name
            im_extra = util_image.imread(im_path_extra, chn='rgb', dtype='float32')
            im_extra = self.transform(im_extra)
            out_dict['gt'] = im_extra

        if self.need_path:
            out_dict['path'] = im_path

        return out_dict

    def reset_dataset(self):
        self.file_paths = random.sample(self.file_paths_all, self.length)

class BaseDataTxt(BaseDataFolder):
    def __init__(
            self,
            txt_path,
            transform_type,
            transform_kwargs=None,
            dir_path_extra=None,
            length=None,
            need_path=False,
            ):

        file_paths_all = util_common.readline_txt(txt_path)
        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all

        self.length = length
        self.need_path = need_path
        self.dir_path_extra = dir_path_extra
        self.transform = get_transforms(transform_type, transform_kwargs)

class InpaintingDataSet(Dataset):
    def __init__(
            self,
            dir_path,
            transform_type,
            transform_kwargs,
            mask_kwargs,
            length=None,
            need_path=False,
            im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            recursive=False,
            ):
        super().__init__()

        file_paths_all = util_common.scan_files_from_folder(dir_path, im_exts, recursive)
        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all

        self.length = length
        self.need_path = need_path
        self.transform = get_transforms(transform_type, transform_kwargs)
        self.mask_generator = MixedMaskGenerator(**mask_kwargs)
        self.iter_i = 0

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im = util_image.imread(im_path, chn='rgb', dtype='float32')
        im = self.transform(im)        # c x h x w
        out_dict = {'gt':im, }

        mask = self.mask_generator(im, iter_i=self.iter_i)   # c x h x w
        self.iter_i += 1
        im_masked = im *  (1 - mask)
        out_dict['lq'] = im_masked
        out_dict['mask'] = mask

        if self.need_path:
            out_dict['path'] = im_path

        return out_dict

    def reset_dataset(self):
        self.file_paths = random.sample(self.file_paths_all, self.length)

