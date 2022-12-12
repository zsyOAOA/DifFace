import random
import numpy as np
from pathlib import Path
from ResizeRight.resize_right import resize
from einops import rearrange

import torch
import torchvision as thv
from torch.utils.data import Dataset

from utils import util_sisr
from utils import util_image
from utils import util_common

from basicsr.data.realesrgan_dataset import RealESRGANDataset
from .ffhq_degradation_dataset import FFHQDegradationDataset

def get_transforms(transform_type, out_size, sf):
    if transform_type == 'default':
        transform = thv.transforms.Compose([
            util_image.SpatialAug(),
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    elif transform_type == 'face':
        transform = thv.transforms.Compose([
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    elif transform_type == 'bicubic':
        transform = thv.transforms.Compose([
            util_sisr.Bicubic(1/sf),
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    else:
        raise ValueError(f'Unexpected transform_variant {transform_variant}')
    return transform

def create_dataset(dataset_config):
    if dataset_config['type'] == 'gfpgan':
        dataset = FFHQDegradationDataset(dataset_config['params'])
    elif dataset_config['type'] == 'face':
        dataset = BaseDatasetFace(**dataset_config['params'])
    elif dataset_config['type'] == 'bicubic':
        dataset = DatasetBicubic(**dataset_config['params'])
    elif dataset_config['type'] == 'folder':
        dataset = BaseDataFolder(**dataset_config['params'])
    elif dataset_config['type'] == 'realesrgan':
        dataset = RealESRGANDataset(dataset_config['params'])
    else:
        raise NotImplementedError(dataset_config['type'])

    return dataset

class BaseDatasetFace(Dataset):
    def __init__(self, celeba_txt=None,
                       ffhq_txt=None,
                       out_size=256,
                       transform_type='face',
                       sf=None,
                       length=None):
        super().__init__()
        self.files_names = util_common.readline_txt(celeba_txt) + util_common.readline_txt(ffhq_txt)

        if length is None:
            self.length = len(self.files_names)
        else:
            self.length = length

        self.transform = get_transforms(transform_type, out_size, sf)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        im_path = self.files_names[index]
        im = util_image.imread(im_path, chn='rgb', dtype='uint8')
        im = self.transform(im)
        return {'image':im,}

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
        im_lq = resize(im_gt, scale_factors=1/self.sf)
        if self.up_back:
            im_lq = resize(im_lq, scale_factors=self.sf)

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
            dir_path_gt,
            need_gt_path=True,
            length=None,
            ext=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            mean=0.5,
            std=0.5,
            ):
        super(BaseDataFolder, self).__init__()
        if isinstance(ext, str):
            files_path = [str(x) for x in Path(dir_path).glob(f'*.{ext}')]
        else:
            assert isinstance(ext, list) or isinstance(ext, tuple)
            files_path = []
            for current_ext in ext:
                files_path.extend([str(x) for x in Path(dir_path).glob(f'*.{current_ext}')])
        self.files_path = files_path if length is None else files_path[:length]
        self.dir_path_gt = dir_path_gt
        self.need_gt_path = need_gt_path
        self.mean=mean
        self.std=std

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, index):
        im_path = self.files_path[index]
        im = util_image.imread(im_path, chn='rgb', dtype='float32')
        im = util_image.normalize_np(im, mean=self.mean, std=self.std, reverse=False)
        im = rearrange(im, 'h w c -> c h w')
        out_dict = {'image':im.astype(np.float32), 'lq':im.astype(np.float32)}

        if self.need_gt_path:
            out_dict['path'] = im_path

        if self.dir_path_gt is not None:
            gt_path = str(Path(self.dir_path_gt) / Path(im_path).name)
            im_gt = util_image.imread(gt_path, chn='rgb', dtype='float32')
            im_gt = util_image.normalize_np(im_gt, mean=self.mean, std=self.std, reverse=False)
            im_gt = rearrange(im_gt, 'h w c -> c h w')
            out_dict['gt'] = im_gt.astype(np.float32)

        return out_dict
