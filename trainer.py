#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-05-18 13:04:06

import os
import sys
import math
import time
import lpips
import random
import datetime
import functools
import numpy as np
from pathlib import Path
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
from einops import rearrange

from datapipe.datasets import create_dataset
from models.resample import UniformSampler

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.utils.data as udata
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import util_net
from utils import util_common
from utils import util_image

from basicsr.utils import DiffJPEG
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

class TrainerBase:
    def __init__(self, configs):
        self.configs = configs

        # setup distributed training: self.num_gpus, self.rank
        self.setup_dist()

        # setup seed
        self.setup_seed()

        # setup logger: self.logger
        self.init_logger()

        # logging the configurations
        if self.rank == 0: self.logger.info(OmegaConf.to_yaml(self.configs))

        # build model: self.model, self.loss
        self.build_model()

        # setup optimization: self.optimzer, self.sheduler
        self.setup_optimizaton()

        # resume
        self.resume_from_ckpt()

    def setup_dist(self):
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    )

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def setup_seed(self, seed=None):
        seed = self.configs.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def init_logger(self):
        if self.configs.resume:
            assert self.configs.resume.endswith(".pth")
            save_dir = Path(self.configs.resume).parents[1]
            project_id = save_dir.name
        else:
            project_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            save_dir = Path(self.configs.save_dir) / project_id
            if not save_dir.exists() and self.rank == 0:
                save_dir.mkdir(parents=True)

        # setting log counter
        if self.rank == 0:
            self.log_step = {phase: 1 for phase in ['train', 'val']}
            self.log_step_img = {phase: 1 for phase in ['train', 'val']}

        # text logging
        logtxet_path = save_dir / 'training.log'
        if self.rank == 0:
            if logtxet_path.exists():
                assert self.configs.resume
            self.logger = logger
            self.logger.remove()
            self.logger.add(logtxet_path, format="{message}", mode='a', level='INFO')
            self.logger.add(sys.stdout, format="{message}")

        # tensorboard logging
        log_dir = save_dir / 'tf_logs'
        self.tf_logging = self.configs.train.tf_logging
        if self.rank == 0 and self.tf_logging:
            if not log_dir.exists():
                log_dir.mkdir()
            self.writer = SummaryWriter(str(log_dir))

        # checkpoint saving
        ckpt_dir = save_dir / 'ckpts'
        self.ckpt_dir = ckpt_dir
        if self.rank == 0 and (not ckpt_dir.exists()):
            ckpt_dir.mkdir()
        if 'ema_rate' in self.configs.train:
            self.ema_rate = self.configs.train.ema_rate
            assert isinstance(self.ema_rate, float), "Ema rate must be a float number"
            ema_ckpt_dir = save_dir / 'ema_ckpts'
            self.ema_ckpt_dir = ema_ckpt_dir
            if self.rank == 0 and (not ema_ckpt_dir.exists()):
                ema_ckpt_dir.mkdir()

        # save images into local disk
        self.local_logging = self.configs.train.local_logging
        if self.rank == 0 and self.local_logging:
            image_dir = save_dir / 'images'
            if not image_dir.exists():
                (image_dir / 'train').mkdir(parents=True)
                (image_dir / 'val').mkdir(parents=True)
            self.image_dir = image_dir

        # logging the configurations
        if self.rank == 0:
            self.logger.info(OmegaConf.to_yaml(self.configs))

    def close_logger(self):
        if self.rank == 0 and self.tf_logging:
            self.writer.close()

    def resume_from_ckpt(self):
        def _load_ema_state(ema_state, ckpt):
            for key in ema_state.keys():
                if key not in ckpt and key.startswith('module'):
                    ema_state[key] = deepcopy(ckpt[7:].detach().data)
                elif key not in ckpt and (not key.startswith('module')):
                    ema_state[key] = deepcopy(ckpt['module.'+key].detach().data)
                else:
                    ema_state[key] = deepcopy(ckpt[key].detach().data)

        if self.configs.resume:
            if type(self.configs.resume) == bool:
                ckpt_index = max([int(x.stem.split('_')[1]) for x in Path(self.ckpt_dir).glob('*.pth')])
                ckpt_path = str(Path(self.ckpt_dir) / f"model_{ckpt_index}.pth")
            else:
                ckpt_path = self.configs.resume
            assert os.path.isfile(ckpt_path)
            if self.rank == 0:
                self.logger.info(f"=> Loading checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            util_net.reload_model(self.model, ckpt['state_dict'])

            # EMA model
            if self.rank == 0 and hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / ("ema_"+Path(ckpt_path).name)
                self.logger.info(f"=> Loading EMA checkpoint from {str(ema_ckpt_path)}")
                ema_ckpt = torch.load(ema_ckpt_path, map_location=f"cuda:{self.rank}")
                _load_ema_state(self.ema_state, ema_ckpt)

            torch.cuda.empty_cache()

            # starting iterations
            self.iters_start = ckpt['iters_start']

            # learning rate scheduler
            for ii in range(self.iters_start):
                self.adjust_lr(ii)

            # logging counter
            if self.rank == 0:
                self.log_step = ckpt['log_step']
                self.log_step_img = ckpt['log_step_img']

            # reset the seed
            self.setup_seed(self.iters_start)
        else:
            self.iters_start = 0

    def setup_optimizaton(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.configs.train.lr,
                                           weight_decay=self.configs.train.weight_decay)

    def build_model(self):
        params = self.configs.model.get('params', dict)
        model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        if self.num_gpus > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.model = DDP(model.cuda(), device_ids=[self.rank,])  # wrap the network
        else:
            self.model = model.cuda()
        if hasattr(self.configs.model, 'ckpt_path') and self.configs.model.ckpt_path is not None:
            ckpt_path = self.configs.model.ckpt_path
            if self.rank == 0:
                self.logger.info(f"Initializing model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(self.model, ckpt)

        # EMA
        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_model = deepcopy(model).cuda()
            self.ema_state = OrderedDict(
                {key:deepcopy(value.data) for key, value in self.model.state_dict().items()}
                )
            self.ema_ignore_keys = [x for x in self.ema_state.keys() if ('running_' in x or 'num_batches_tracked' in x)]

        # model information
        self.print_model_info()

    def build_dataloader(self):
        def _wrap_loader(loader):
            while True: yield from loader

        datasets = {}
        phases = ['train', ]
        if 'val' in self.configs.data:
            phases.append('val')
        for current_phase in phases:
            dataset_config = self.configs.data.get(current_phase, dict)
            datasets[current_phase] = create_dataset(dataset_config)

        dataloaders = {}
        # train dataloader
        if self.rank == 0:
            for current_phase in phases:
                length = len(datasets[current_phase])
                self.logger.info('Number of images in {:s} data set: {:d}'.format(current_phase, length))
        if self.num_gpus > 1:
            shuffle = False
            sampler = udata.distributed.DistributedSampler(datasets['train'],
                                                           num_replicas=self.num_gpus,
                                                           rank=self.rank)
        else:
            shuffle = True
            sampler = None
        dataloaders['train'] = _wrap_loader(udata.DataLoader(
                                    datasets['train'],
                                    batch_size=self.configs.train.batch[0] // self.num_gpus,
                                    shuffle=shuffle,
                                    drop_last=False,
                                    num_workers=self.configs.train.num_workers,
                                    pin_memory=True,
                                    prefetch_factor=self.configs.train.prefetch_factor,
                                    worker_init_fn=my_worker_init_fn,
                                    sampler=sampler))
        if 'val' in phases and self.rank == 0:
            dataloaders['val'] = udata.DataLoader(
                    datasets['val'],
                    batch_size=self.configs.train.batch[1],
                    shuffle=False,
                    drop_last=False,
                    num_workers=0,
                    pin_memory=True,
                    )

        self.datasets = datasets
        self.dataloaders = dataloaders
        self.sampler = sampler

    def print_model_info(self):
        if self.rank == 0:
            num_params = util_net.calculate_parameters(self.model) / 1000**2
            # self.logger.info("Detailed network architecture:")
            # self.logger.info(self.model.__repr__())
            self.logger.info(f"Number of parameters: {num_params:.2f}M")

    def prepare_data(self, data, phase='train'):
        return {key:value.cuda() for key, value in data.items()}

    def validation(self):
        pass

    def train(self):
        self.build_dataloader() # prepare data: self.dataloaders, self.datasets, self.sampler

        self.model.train()
        num_iters_epoch = math.ceil(len(self.datasets['train']) / self.configs.train.batch[0])
        for ii in range(self.iters_start, self.configs.train.iterations):
            self.current_iters = ii + 1

            # prepare data
            data = self.prepare_data(next(self.dataloaders['train']), phase='train')

            # training phase
            self.training_step(data)

            # validation phase
            if (ii+1) % self.configs.train.val_freq == 0 and 'val' in self.dataloaders and self.rank==0:
                self.validation()

            #update learning rate
            self.adjust_lr()

            # save checkpoint
            if (ii+1) % self.configs.train.save_freq == 0 and self.rank == 0:
                self.save_ckpt()

            if (ii+1) % num_iters_epoch == 0 and not self.sampler is None:
                self.sampler.set_epoch(ii+1)

        # close the tensorboard
        if self.rank == 0:
            self.close_logger()

    def training_step(self, data):
        pass

    def adjust_lr(self, current_iters=None):
        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()

    def save_ckpt(self):
        if self.rank == 0:
            ckpt_path = self.ckpt_dir / 'model_{:d}.pth'.format(self.current_iters)
            torch.save({'iters_start': self.current_iters,
                        'log_step': {phase:self.log_step[phase] for phase in ['train', 'val']},
                        'log_step_img': {phase:self.log_step_img[phase] for phase in ['train', 'val']},
                        'state_dict': self.model.state_dict()}, ckpt_path)
            if hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / 'ema_model_{:d}.pth'.format(self.current_iters)
                torch.save(self.ema_state, ema_ckpt_path)

    def logging_image(self, im_tensor, tag, phase, add_global_step=False, nrow=8):
        """
        Args:
            im_tensor: b x c x h x w tensor
            im_tag: str
            phase: 'train' or 'val'
            nrow: number of displays in each row
        """
        assert self.tf_logging or self.local_logging
        im_tensor = vutils.make_grid(im_tensor, nrow=nrow, normalize=True, scale_each=True) # c x H x W
        if self.local_logging:
            im_path = str(self.image_dir / phase / f"{tag}-{self.log_step_img[phase]}.png")
            im_np = im_tensor.cpu().permute(1,2,0).numpy()
            util_image.imwrite(im_np, im_path)
        if self.tf_logging:
            self.writer.add_image(
                    f"{phase}-{tag}-{self.log_step_img[phase]}",
                    im_tensor,
                    self.log_step_img[phase],
                    )
        if add_global_step:
            self.log_step_img[phase] += 1

    def logging_metric(self, metrics, tag, phase, add_global_step=False):
        """
        Args:
            metrics: dict
            tag: str
            phase: 'train' or 'val'
        """
        if self.tf_logging:
            tag = f"{phase}-{tag}"
            if isinstance(metrics, dict):
                self.writer.add_scalars(tag, metrics, self.log_step[phase])
            else:
                self.writer.add_scalar(tag, metrics, self.log_step[phase])
            if add_global_step:
                self.log_step[phase] += 1
        else:
            pass

    def update_ema_model(self):
        if self.num_gpus > 1:
            dist.barrier()
        if self.rank == 0:
            source_state = self.model.state_dict()
            rate = self.ema_rate
            for key, value in self.ema_state.items():
                if key in self.ema_ignore_keys:
                    self.ema_state[key] = source_state[key]
                else:
                    self.ema_state[key].mul_(rate).add_(source_state[key].detach().data, alpha=1-rate)

    def reload_ema_model(self):
        if self.rank == 0:
            if self.num_gpus > 1:
                model_state = {key[7:]:value for key, value in self.ema_state.items()}
            else:
                model_state = self.ema_state
            self.ema_model.load_state_dict(model_state)

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

class TrainerSR(TrainerBase):
    def build_model(self):
        super().build_model()

        # LPIPS metric
        lpips_loss = lpips.LPIPS(net='alex').cuda()
        self.freeze_model(lpips_loss)
        self.lpips_loss = lpips_loss.eval()

    def feed_data(self, data, phase='train'):
        if phase == 'train':
            pred = self.model(data['lq'])
        elif phase == 'val':
            with torch.no_grad():
                if hasattr(self.configs.train, 'ema_rate'):
                    pred = self.ema_model(data['lq'])
                else:
                    pred = self.model(data['lq'])
        else:
            raise ValueError(f"Phase must be 'train' or 'val', now phase={phase}")

        return pred

    def get_loss(self, pred, data):
        target = data['gt']
        if self.configs.train.loss_type == "L1":
            return F.l1_loss(pred, target, reduction='mean')
        elif self.configs.train.loss_type == "L2":
            return F.mse_loss(pred, target, reduction='mean')
        else:
            raise ValueError(f"Not supported loss type: {self.configs.train.loss_type}")

    def setup_optimizaton(self):
        super().setup_optimizaton()   # self.optimizer
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max = self.configs.train.iterations,
                eta_min=self.configs.train.lr_min,
                )

    def training_step(self, data):
        current_batchsize = data['lq'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        self.optimizer.zero_grad()
        for jj in range(0, current_batchsize, micro_batchsize):
            micro_data = {key:value[jj:jj+micro_batchsize,] for key, value in data.items()}
            last_batch = (jj+micro_batchsize >= current_batchsize)
            hq_pred = self.feed_data(data, phase='train')
            if last_batch or self.num_gpus <= 1:
                loss = self.get_loss(hq_pred, micro_data)
            else:
                with self.model.no_sync():
                    loss = self.get_loss(hq_pred, micro_data)
            loss /= num_grad_accumulate
            loss.backward()

            # make logging
            self.log_step_train(hq_pred, loss, micro_data, flag=last_batch)

        self.optimizer.step()
        if hasattr(self.configs.train, 'ema_rate'):
            self.update_ema_model()

    def log_step_train(self, hq_pred, loss, batch, flag=False, phase='train'):
        '''
        param loss: loss value
        '''
        if self.rank == 0:
            chn = batch['lq'].shape[1]
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = 0

            self.loss_mean += loss.item()

            if self.current_iters % self.configs.train.log_freq[0] == 0 and flag:
                self.loss_mean /= self.configs.train.log_freq[0]
                log_str = 'Train:{:06d}/{:06d}, Loss:{:.2e}, lr:{:.2e}'.format(
                        self.current_iters,
                        self.configs.train.iterations,
                        self.loss_mean,
                        self.optimizer.param_groups[0]['lr']
                        )
                self.logger.info(log_str)
                self.logging_metric(self.loss_mean, 'Loss', phase, add_global_step=True)
            if self.current_iters % self.configs.train.log_freq[1] == 0 and flag:
                self.logging_image(batch['lq'], tag="lq", phase=phase, add_global_step=False)
                self.logging_image(batch['gt'], tag="hq", phase=phase, add_global_step=False)
                self.logging_image(hq_pred.detach(), tag="pred", phase=phase, add_global_step=True)

            if self.current_iters % self.configs.train.save_freq == 1 and flag:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0 and flag:
                self.toc = time.time()
                elaplsed = (self.toc - self.tic)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("="*60)

    def validation(self, phase='val'):
        if hasattr(self.configs.train, 'ema_rate'):
            self.reload_ema_model()
            self.ema_model.eval()
        else:
            self.model.eval()

        psnr_mean = lpips_mean = 0
        total_iters = math.ceil(len(self.datasets[phase]) / self.configs.train.batch[1])
        for ii, data in enumerate(self.dataloaders[phase]):
            data = self.prepare_data(data, phase='val')
            hq_pred = self.feed_data(data, phase='val')
            hq_pred.clamp_(0.0, 1.0)
            lpips = self.lpips_loss((hq_pred-0.5)*2, (data['gt']-0.5)*2).sum().item()
            psnr = util_image.batch_PSNR(hq_pred, data['gt'], ycbcr=True)

            psnr_mean += psnr
            lpips_mean += lpips

            if (ii+1) % self.configs.train.log_freq[2] == 0:
                log_str = '{:s}:{:03d}/{:03d}, PSNR={:5.2f}, LPIPS={:6.4f}'.format(
                        phase,
                        ii+1,
                        total_iters,
                        psnr / hq_pred.shape[0],
                        lpips / hq_pred.shape[0]
                        )
                self.logger.info(log_str)
                self.logging_image(data['lq'], tag="lq", phase=phase, add_global_step=False)
                self.logging_image(data['gt'], tag="hq", phase=phase, add_global_step=False)
                self.logging_image(hq_pred.detach(), tag="pred", phase=phase, add_global_step=True)

        psnr_mean /= len(self.datasets[phase])
        lpips_mean /= len(self.datasets[phase])
        self.logging_metric(
                {"PSRN": psnr_mean, "lpips": lpips_mean},
                tag='Metrics',
                phase=phase,
                add_global_step=True,
                )
        # logging
        self.logger.info(f'PSNR={psnr_mean:5.2f}, LPIPS={lpips_mean:6.4f}')
        self.logger.info("="*60)

        if not hasattr(self.configs.train, 'ema_rate'):
            self.model.train()

class TrainerInpainting(TrainerSR):
    def get_loss(self, pred, data, weight_known=1, weight_missing=10):
        if self.configs.train.loss_type == "L1":
            mask, target = data['mask'], data['gt']
            per_pixel_loss = F.l1_loss(pred, target, reduction='none')
            pixel_weights = mask * weight_missing + (1 - mask) * weight_known
            loss = (pixel_weights * per_pixel_loss).sum() / pixel_weights.sum()
        elif self.configs.train.loss_type == "L2":
            mask, target = data['mask'], data['gt']
            per_pixel_loss = F.mse_loss(pred, target, reduction='none')
            pixel_weights = mask * weight_missing + (1 - mask) * weight_known
            loss = (pixel_weights * per_pixel_loss).sum() / pixel_weights.sum()
        else:
            raise ValueError(f"Not supported loss type: {self.configs.train.loss_type}")

        return loss

    def feed_data(self, data, phase='train'):
        if not 'mask' in data:
            ysum = torch.sum(data['lq'], dim=1, keepdim=True)
            mask = torch.where(
                    ysum==0,
                    torch.ones_like(ysum),
                    torch.zeros_like(ysum),
                    ).to(dtype=torch.float32, device=data['lq'].device)
        else:
            mask = data['mask']

        inputs = torch.cat([data['lq'], mask], dim=1)

        if phase == 'train':
            pred = self.model(inputs)
        elif phase == 'val':
            with torch.no_grad():
                if hasattr(self.configs.train, 'ema_rate'):
                    pred = self.ema_model(inputs)
                else:
                    pred = self.model(inputs)
        else:
            raise ValueError(f"Phase must be 'train' or 'val', now phase={phase}")

        return pred

class TrainerDiffusionFace(TrainerBase):
    def build_model(self):
        super().build_model()
        params = self.configs.diffusion.get('params', dict)
        self.base_diffusion = util_common.get_obj_from_str(self.configs.diffusion.target)(**params)
        self.sample_scheduler_diffusion = UniformSampler(self.base_diffusion.num_timesteps)

    def training_step(self, data):
        current_batchsize = data['image'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        if self.configs.train.use_fp16:
            scaler = amp.GradScaler()

        self.optimizer.zero_grad()
        for jj in range(0, current_batchsize, micro_batchsize):
            micro_data = {key:value[jj:jj+micro_batchsize,] for key, value in data.items()}
            last_batch = (jj+micro_batchsize >= current_batchsize)
            tt, weights = self.sample_scheduler_diffusion.sample(
                    micro_data['image'].shape[0],
                    device=f"cuda:{self.rank}",
                    use_fp16=self.configs.train.use_fp16
                    )
            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.model,
                micro_data['image'],
                tt,
                model_kwargs={'y':micro_data['label']} if 'label' in micro_data else None,
            )
            if self.configs.train.use_fp16:
                with amp.autocast():
                    if last_batch or self.num_gpus <= 1:
                        losses = compute_losses()
                    else:
                        with self.model.no_sync():
                            losses = compute_losses()
                    loss = (losses["loss"] * weights).mean() / num_grad_accumulate
                scaler.scale(loss).backward()
            else:
                if last_batch or self.num_gpus <= 1:
                    losses = compute_losses()
                else:
                    with self.model.no_sync():
                        losses = compute_losses()
                loss = (losses["loss"] * weights).mean() / num_grad_accumulate
                loss.backward()

            # make logging
            self.log_step_train(losses, tt, micro_data, last_batch)

        if self.configs.train.use_fp16:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()

        self.update_ema_model()

    def adjust_lr(self, current_iters=None):
        current_iters = self.current_iters if current_iters is None else current_iters
        base_lr = self.configs.train.lr
        linear_steps = self.configs.train.milestones[0]
        if current_iters <= linear_steps:
            for params_group in self.optimizer.param_groups:
                params_group['lr'] = (current_iters / linear_steps) * base_lr

    def log_step_train(self, loss, tt, batch, flag=False, phase='train'):
        '''
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        '''
        if self.rank == 0:
            chn = batch['image'].shape[1]
            num_timesteps = self.base_diffusion.num_timesteps
            record_steps = [1, (num_timesteps // 2) + 1, num_timesteps]
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {key:torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                                  for key in loss.keys()}
                self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)
            for jj in range(len(record_steps)):
                for key, value in loss.items():
                    index = record_steps[jj] - 1
                    mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
                    current_loss = torch.sum(value.detach() * mask)
                    self.loss_mean[key][jj] += current_loss.item()
                self.loss_count[jj] += mask.sum().item()

            if self.current_iters % self.configs.train.log_freq[0] == 0 and flag:
                if torch.any(self.loss_count == 0):
                    self.loss_count += 1e-4
                for key in loss.keys():
                    self.loss_mean[key] /= self.loss_count
                log_str = 'Train: {:06d}/{:06d}, Loss: '.format(
                        self.current_iters,
                        self.configs.train.iterations)
                for jj, current_record in enumerate(record_steps):
                    if 'vb' in self.loss_mean:
                        log_str += 't({:d}):{:.2e}/{:.2e}/{:.2e}, '.format(
                                current_record,
                                self.loss_mean['loss'][jj].item(),
                                self.loss_mean['mse'][jj].item(),
                                self.loss_mean['vb'][jj].item(),
                                )
                    else:
                        log_str += 't({:d}):{:.2e}, '.format(
                                current_record,
                                self.loss_mean['loss'][jj].item(),
                                )
                log_str += 'lr:{:.2e}'.format(self.optimizer.param_groups[0]['lr'])
                self.logger.info(log_str)
            if self.current_iters % self.configs.train.log_freq[1] == 0 and flag:
                self.logging_image(batch['image'], tag='image', phase=phase, add_global_step=True)

            if self.current_iters % self.configs.train.save_freq == 1 and flag:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0 and flag:
                self.toc = time.time()
                elaplsed = (self.toc - self.tic) * num_timesteps  / (num_timesteps - 1)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("="*130)

    def validation(self, phase='val'):
        self.reload_ema_model(self.ema_rates[0])
        self.ema_model.eval()

        indices = [int(self.base_diffusion.num_timesteps * x) for x in [0.25, 0.5, 0.75, 1]]
        chn = 3
        batch_size = self.configs.train.batch[1]
        shape = (batch_size, chn,) + (self.configs.data.train.params.out_size,) * 2
        num_iters = 0
        for sample in self.base_diffusion.p_sample_loop_progressive(
                model = self.ema_model,
                shape = shape,
                noise = None,
                clip_denoised = True,
                model_kwargs = None,
                device = f"cuda:{self.rank}",
                progress=False
                ):
            num_iters += 1
            img = util_image.normalize_th(sample['sample'], reverse=True)
            if num_iters == 1:
                im_recover = img
            elif num_iters in indices:
                im_recover_last = img
                im_recover = torch.cat((im_recover, im_recover_last), dim=1)
        im_recover = rearrange(im_recover, 'b (k c) h w -> (b k) c h w', c=chn)
        self.logging_image(
                im_recover,
                tag='progress',
                phase=phase,
                add_global_step=True,
                nrow=len(indices),
                )

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__ == '__main__':
    from utils import util_image
    from  einops import rearrange
    im1 = util_image.imread('./testdata/inpainting/val/places/Places365_val_00012685_crop000.png',
                            chn = 'rgb', dtype='float32')
    im2 = util_image.imread('./testdata/inpainting/val/places/Places365_val_00014886_crop000.png',
                            chn = 'rgb', dtype='float32')
    im = rearrange(np.stack((im1, im2), 3), 'h w c b -> b c h w')
    im_grid = im.copy()
    for alpha in [0.8, 0.4, 0.1, 0]:
        im_new = im * alpha + np.random.randn(*im.shape) * (1 - alpha)
        im_grid = np.concatenate((im_new, im_grid), 1)

    im_grid = np.clip(im_grid, 0.0, 1.0)
    im_grid = rearrange(im_grid, 'b (k c) h w -> (b k) c h w', k=5)
    xx = vutils.make_grid(torch.from_numpy(im_grid), nrow=5, normalize=True, scale_each=True).numpy()
    util_image.imshow(np.concatenate((im1, im2), 0))
    util_image.imshow(xx.transpose((1,2,0)))

