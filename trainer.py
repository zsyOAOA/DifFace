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
        if self.configs.gpu_id:
            gpu_id = self.configs.gpu_id
            num_gpus = len(gpu_id)
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([gpu_id[ii] for ii in range(num_gpus)])
        else:
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
        # only should be run on rank: 0
        save_dir = Path(self.configs.save_dir)
        logtxet_path = save_dir / 'training.log'
        log_dir = save_dir / 'logs'
        ckpt_dir = save_dir / 'ckpts'
        self.ckpt_dir = ckpt_dir
        if self.rank == 0:
            if not save_dir.exists():
                save_dir.mkdir()
            else:
                assert self.configs.resume,  '''Please check the resume parameter. If you do not
                                                want to resume from some checkpoint, please delete
                                                the saving folder first.'''

            # text logging
            if logtxet_path.exists():
                assert self.configs.resume
            self.logger = logger
            self.logger.remove()
            self.logger.add(logtxet_path, format="{message}", mode='a')
            self.logger.add(sys.stderr, format="{message}")

            # tensorboard log
            if not log_dir.exists():
                log_dir.mkdir()
            self.writer = SummaryWriter(str(log_dir))
            self.log_step = {phase: 1 for phase in ['train', 'val']}
            self.log_step_img = {phase: 1 for phase in ['train', 'val']}

            if not ckpt_dir.exists():
                ckpt_dir.mkdir()

    def close_logger(self):
        if self.rank == 0: self.writer.close()

    def resume_from_ckpt(self):
        if self.configs.resume:
            if type(self.configs.resume) == bool:
                ckpt_index = max([int(x.stem.split('_')[1]) for x in Path(self.ckpt_dir).glob('*.pth')])
                ckpt_path = str(Path(self.ckpt_dir) / f"model_{ckpt_index}.pth")
            else:
                ckpt_path = self.configs.resume
            assert os.path.isfile(ckpt_path)
            if self.rank == 0:
                self.logger.info(f"=> Loaded checkpoint {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            util_net.reload_model(self.model, ckpt['state_dict'])
            torch.cuda.empty_cache()

            # iterations
            self.iters_start = ckpt['iters_start']
            # learning rate scheduler
            for ii in range(self.iters_start): self.adjust_lr(ii)
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
            self.model = DDP(model.cuda(), device_ids=[self.rank,])  # wrap the network
        else:
            self.model = model.cuda()

        # LPIPS metric
        if self.rank == 0:
            self.lpips_loss = lpips.LPIPS(net='vgg').cuda()

        # model information
        self.print_model_info()

    def build_dataloader(self):
        def _wrap_loader(loader):
            while True: yield from loader

        datasets = {}
        for phase in ['train', ]:
            dataset_config = self.configs.data.get(phase, dict)
            datasets[phase] = create_dataset(dataset_config)

        dataloaders = {}
        # train dataloader
        if self.rank == 0:
            for phase in ['train',]:
                length = len(datasets[phase])
                self.logger.info('Number of images in {:s} data set: {:d}'.format(phase, length))
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
                                    num_workers=self.configs.train.num_workers // self.num_gpus,
                                    pin_memory=True,
                                    prefetch_factor=self.configs.train.prefetch_factor,
                                    worker_init_fn=my_worker_init_fn,
                                    sampler=sampler))

        self.datasets = datasets
        self.dataloaders = dataloaders
        self.sampler = sampler

    def print_model_info(self):
        if self.rank == 0:
            num_params = util_net.calculate_parameters(self.model) / 1000**2
            self.logger.info("Detailed network architecture:")
            self.logger.info(self.model.__repr__())
            self.logger.info(f"Number of parameters: {num_params:.2f}M")

    def prepare_data(self, phase='train'):
        pass

    def validation(self):
        pass

    def train(self):
        self.build_dataloader() # prepare data: self.dataloaders, self.datasets, self.sampler

        self.model.train()
        num_iters_epoch = math.ceil(len(self.datasets['train']) / self.configs.train.batch[0])
        for ii in range(self.iters_start, self.configs.train.iterations):
            self.current_iters = ii + 1

            # prepare data
            data = self.prepare_data(
                    next(self.dataloaders['train']),
                    self.configs.data.train.type.lower() == 'realesrgan',
                    )

            # training phase
            self.training_step(data)

            # validation phase
            if (ii+1) % self.configs.train.val_freq == 0 and 'val' in self.dataloaders:
                if self.rank==0:
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

    def adjust_lr(self):
        if hasattr(self, 'lr_sheduler'):
            self.lr_sheduler.step()

    def save_ckpt(self):
        ckpt_path = self.ckpt_dir / 'model_{:d}.pth'.format(self.current_iters)
        torch.save({'iters_start': self.current_iters,
                    'log_step': {phase:self.log_step[phase] for phase in ['train', 'val']},
                    'log_step_img': {phase:self.log_step_img[phase] for phase in ['train', 'val']},
                    'state_dict': self.model.state_dict()}, ckpt_path)

class TrainerSR(TrainerBase):
    def __init__(self, configs):
        super().__init__(configs)

    def loss_fun(self, pred, target):
        return F.mse_loss(pred, target, reduction='sum')

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_size'):
            self.queue_size = self.configs.data.train.params.get('queue_size', b*50)
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def prepare_data(self, data, real_esrgan=True):
        if real_esrgan:
            if not hasattr(self, 'jpeger'):
                self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts

            im_gt = data['gt'].cuda()
            kernel1 = data['kernel1'].cuda()
            kernel2 = data['kernel2'].cuda()
            sinc_kernel = data['sinc_kernel'].cuda()

            ori_h, ori_w = im_gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(im_gt, kernel1)
            # random resize
            updown_type = random.choices(
                    ['up', 'down', 'keep'],
                    self.configs.degradation['resize_prob'],
                    )[0]
            if updown_type == 'up':
                scale = random.uniform(1, self.configs.degradation['resize_range'][1])
            elif updown_type == 'down':
                scale = random.uniform(self.configs.degradation['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.configs.degradation['gray_noise_prob']
            if random.random() < self.configs.degradation['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.configs.degradation['noise_range'],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                    )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.configs.degradation['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if random.random() < self.configs.degradation['second_blur_prob']:
                out = filter2D(out, kernel2)
            # random resize
            updown_type = random.choices(
                    ['up', 'down', 'keep'],
                    self.configs.degradation['resize_prob2'],
                    )[0]
            if updown_type == 'up':
                scale = random.uniform(1, self.configs.degradation['resize_range2'][1])
            elif updown_type == 'down':
                scale = random.uniform(self.configs.degradation['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(int(ori_h / self.configs.model.params.sf * scale),
                          int(ori_w / self.configs.model.params.sf * scale)),
                    mode=mode,
                    )
            # add noise
            gray_noise_prob = self.configs.degradation['gray_noise_prob2']
            if random.random() < self.configs.degradation['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.configs.degradation['noise_range2'],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                    )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.configs.degradation['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False,
                    )

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if random.random() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(ori_h // self.configs.model.params.sf,
                              ori_w // self.configs.model.params.sf),
                        mode=mode,
                        )
                out = filter2D(out, sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(ori_h // self.configs.model.params.sf,
                              ori_w // self.configs.model.params.sf),
                        mode=mode,
                        )
                out = filter2D(out, sinc_kernel)

            # clamp and round
            im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.configs.degradation['gt_size']
            im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, self.configs.model.params.sf)
            self.lq, self.gt = im_lq, im_gt

            # training pair pool
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

            return {'lq':self.lq, 'gt':self.gt}
        else:
            return {key:value.cuda() for key, value in data.items()}

    def setup_optimizaton(self):
        super().setup_optimizaton()   # self.optimizer
        self.lr_sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
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
            hq_pred = self.model(micro_data['lq'])
            if last_batch or self.num_gpus <= 1:
                loss = self.loss_fun(hq_pred, micro_data['gt']) / hq_pred.shape[0]
            else:
                with self.model.no_sync():
                    loss = self.loss_fun(hq_pred, micro_data['gt']) / hq_pred.shape[0]
            loss /= num_grad_accumulate
            loss.backward()

            # make logging
            self.log_step_train(hq_pred, loss, micro_data, flag=last_batch)

        self.optimizer.step()

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
                mse_pixel = self.loss_mean / batch['gt'].numel() * batch['gt'].shape[0]
                log_str = 'Train:{:05d}/{:05d}, Loss:{:.2e}, MSE:{:.2e}, lr:{:.2e}'.format(
                        self.current_iters // 100,
                        self.configs.train.iterations // 100,
                        self.loss_mean,
                        mse_pixel,
                        self.optimizer.param_groups[0]['lr']
                        )
                self.logger.info(log_str)
                # tensorboard
                self.writer.add_scalar(f'Loss-Train', self.loss_mean, self.log_step[phase])
                self.log_step[phase] += 1
            if self.current_iters % self.configs.train.log_freq[1] == 0 and flag:
                x1 = vutils.make_grid(batch['lq'], normalize=True, scale_each=True)
                self.writer.add_image("Train LQ Image", x1, self.log_step_img[phase])
                x2 = vutils.make_grid(batch['gt'], normalize=True, scale_each=True)
                self.writer.add_image("Train HQ Image", x2, self.log_step_img[phase])
                x3 = vutils.make_grid(hq_pred.detach().data, normalize=True, scale_each=True)
                self.writer.add_image("Train Recovered Image", x3, self.log_step_img[phase])
                self.log_step_img[phase] += 1

            if self.current_iters % self.configs.train.save_freq == 1 and flag:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0 and flag:
                self.toc = time.time()
                elaplsed = (self.toc - self.tic)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("="*60)

    def validation(self, phase='val'):
        if self.rank == 0:
            self.model.eval()
            psnr_mean = lpips_mean = 0
            total_iters = math.ceil(len(self.datasets[phase]) / self.configs.train.batch[1])
            for ii, data in enumerate(self.dataloaders[phase]):
                data = self.prepare_data(data, real_esrgan=(self.configs.data.val.type=='realesrgan'))
                with torch.no_grad():
                    hq_pred = self.model(data['lq'])
                    hq_pred.clamp_(0.0, 1.0)
                    lpips = self.lpips_loss(
                            util_image.normalize_th(hq_pred, reverse=False),
                            util_image.normalize_th(data['gt'], reverse=False),
                            ).sum().item()
                psnr = util_image.batch_PSNR(
                        hq_pred,
                        data['gt'],
                        ycbcr=True
                        )

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
                    x1 = vutils.make_grid(data['lq'], normalize=True, scale_each=True)
                    self.writer.add_image("Validation LQ Image", x1, self.log_step_img[phase])
                    x2 = vutils.make_grid(data['gt'], normalize=True, scale_each=True)
                    self.writer.add_image("Validation HQ Image", x2, self.log_step_img[phase])
                    x3 = vutils.make_grid(hq_pred.detach().data, normalize=True, scale_each=True)
                    self.writer.add_image("Validation Recovered Image", x3, self.log_step_img[phase])
                    self.log_step_img[phase] += 1

            psnr_mean /= len(self.datasets[phase])
            lpips_mean /= len(self.datasets[phase])
            # tensorboard
            self.writer.add_scalar('Validation PSRN', psnr_mean, self.log_step[phase])
            self.writer.add_scalar('Validation LPIPS', lpips_mean, self.log_step[phase])
            self.log_step[phase] += 1
            # logging
            self.logger.info(f'PSNR={psnr_mean:5.2f}, LPIPS={lpips_mean:6.4f}')
            self.logger.info("="*60)

            self.model.train()

    def build_dataloader(self):
        super().build_dataloader()
        if self.rank == 0 and 'val' in self.configs.data:
            dataset_config = self.configs.data.get('val', dict)
            self.datasets['val'] = create_dataset(dataset_config)
            self.dataloaders['val'] = udata.DataLoader(
                    self.datasets['val'],
                    batch_size=self.configs.train.batch[1],
                    shuffle=False,
                    drop_last=False,
                    num_workers=0,
                    pin_memory=True,
                    )

class TrainerDiffusionFace(TrainerBase):
    def __init__(self, configs):
        # ema settings
        self.ema_rates = OmegaConf.to_object(configs.train.ema_rates)
        super().__init__(configs)

    def init_logger(self):
        super().init_logger()

        save_dir = Path(self.configs.save_dir)
        ema_ckpt_dir = save_dir / 'ema_ckpts'
        if self.rank == 0:
            if not ema_ckpt_dir.exists():
                util_common.mkdir(ema_ckpt_dir, delete=False, parents=False)
            else:
                if not self.configs.resume:
                    util_common.mkdir(ema_ckpt_dir, delete=True, parents=False)

        self.ema_ckpt_dir = ema_ckpt_dir

    def resume_from_ckpt(self):
        super().resume_from_ckpt()

        def _load_ema_state(ema_state, ckpt):
            for key in ema_state.keys():
                ema_state[key] = deepcopy(ckpt[key].detach().data)

        if self.configs.resume:
            # ema model
            if type(self.configs.resume) == bool:
                ckpt_index = max([int(x.stem.split('_')[1]) for x in Path(self.ckpt_dir).glob('*.pth')])
                ckpt_path = str(Path(self.ckpt_dir) / f"model_{ckpt_index}.pth")
            else:
                ckpt_path = self.configs.resume
            assert os.path.isfile(ckpt_path)
            # EMA model
            for rate in self.ema_rates:
                ema_ckpt_path = self.ema_ckpt_dir / (f"ema0{int(rate*1000)}_"+Path(ckpt_path).name)
                ema_ckpt = torch.load(ema_ckpt_path, map_location=f"cuda:{self.rank}")
                _load_ema_state(self.ema_state[f"0{int(rate*1000)}"], ema_ckpt)

    def build_model(self):
        params = self.configs.model.get('params', dict)
        model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        self.ema_model = deepcopy(model.cuda())
        if self.num_gpus > 1:
            self.model = DDP(model.cuda(), device_ids=[self.rank,])  # wrap the network
        else:
            self.model = model.cuda()

        self.ema_state = {}
        for rate in self.ema_rates:
            self.ema_state[f"0{int(rate*1000)}"] = OrderedDict(
                {key:deepcopy(value.data) for key, value in self.model.state_dict().items()}
                )

        # model information
        self.print_model_info()

        params = self.configs.diffusion.get('params', dict)
        self.base_diffusion = util_common.get_obj_from_str(self.configs.diffusion.target)(**params)
        self.sample_scheduler_diffusion = UniformSampler(self.base_diffusion.num_timesteps)

    def prepare_data(self, data, realesrgan=False):
        data = {key:value.cuda() for key, value in data.items()}
        return data

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

    def update_ema_model(self):
        if self.num_gpus > 1:
            dist.barrier()
        if self.rank == 0:
            for rate in self.ema_rates:
                ema_state = self.ema_state[f"0{int(rate*1000)}"]
                source_state = self.model.state_dict()
                for key, value in ema_state.items():
                    ema_state[key].mul_(rate).add_(source_state[key].detach().data, alpha=1-rate)

    def adjust_lr(self, ii):
        base_lr = self.configs.train.lr
        linear_steps = self.configs.train.milestones[0]
        if ii <= linear_steps:
            for params_group in self.optimizer.param_groups:
                params_group['lr'] = (ii / linear_steps) * base_lr
        elif ii in self.configs.train.milestones:
            for params_group in self.optimizer.param_groups:
                params_group['lr'] *= 0.5

    def log_step_train(self, loss, tt, batch, flag=False, phase='train'):
        '''
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        '''
        if self.rank == 0:
            chn = batch['image'].shape[1]
            num_timesteps = self.base_diffusion.num_timesteps
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {key:torch.zeros(size=(num_timesteps,), dtype=torch.float64)
                                  for key in loss.keys()}
                self.loss_count = torch.zeros(size=(num_timesteps,), dtype=torch.float64)
            for key, value in loss.items():
                self.loss_mean[key][tt, ] += value.detach().data.cpu()
            self.loss_count[tt,] += 1

            if self.current_iters % self.configs.train.log_freq[0] == 0 and flag:
                if torch.any(self.loss_count == 0):
                    self.loss_count += 1e-4
                for key, value in loss.items():
                    self.loss_mean[key] /= self.loss_count
                log_str = 'Train: {:05d}/{:05d}, Loss: '.format(
                        self.current_iters // 100,
                        self.configs.train.iterations // 100)
                for kk in [1, num_timesteps // 2, num_timesteps]:
                    if 'vb' in self.loss_mean:
                        log_str += 't({:d}):{:.2e}/{:.2e}/{:.2e}, '.format(
                                kk,
                                self.loss_mean['loss'][kk-1].item(),
                                self.loss_mean['mse'][kk-1].item(),
                                self.loss_mean['vb'][kk-1].item(),
                                )
                    else:
                        log_str += 't({:d}):{:.2e}, '.format(kk, self.loss_mean['loss'][kk-1].item())
                log_str += 'lr:{:.2e}'.format(self.optimizer.param_groups[0]['lr'])
                self.logger.info(log_str)
                # tensorboard
                for kk in [1, num_timesteps // 2, num_timesteps]:
                    self.writer.add_scalar(f'Loss-Step-{kk}',
                                           self.loss_mean['loss'][kk-1].item(),
                                           self.log_step[phase])
                self.log_step[phase] += 1
            if self.current_iters % self.configs.train.log_freq[1] == 0 and flag:
                x1 = vutils.make_grid(batch['image'], normalize=True, scale_each=True)
                self.writer.add_image("Training Image", x1, self.log_step_img[phase])
                self.log_step_img[phase] += 1

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
        # noise  = torch.randn(shape,
                             # dtype=torch.float32,
                             # generator=torch.Generator('cpu').manual_seed(10000)).cuda()
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
        x1 = vutils.make_grid(im_recover, nrow=len(indices)+1, normalize=False)
        self.writer.add_image('Validation Sample', x1, self.log_step_img[phase])
        self.log_step_img[phase] += 1

    def save_ckpt(self):
        if self.rank == 0:
            ckpt_path = self.ckpt_dir / 'model_{:d}.pth'.format(self.current_iters)
            torch.save({'iters_start': self.current_iters,
                        'log_step': {phase:self.log_step[phase] for phase in ['train', 'val']},
                        'log_step_img': {phase:self.log_step_img[phase] for phase in ['train', 'val']},
                        'state_dict': self.model.state_dict()}, ckpt_path)
            for rate in self.ema_rates:
                ema_ckpt_path = self.ema_ckpt_dir / (f"ema0{int(rate*1000)}_"+ckpt_path.name)
                torch.save(self.ema_state[f"0{int(rate*1000)}"], ema_ckpt_path)

    def calculate_lpips(self, inputs, targets):
        inputs, targets = [(x-0.5)/0.5 for x in [inputs, targets]] # [-1, 1]
        with torch.no_grad():
            mean_lpips = self.lpips_loss(inputs, targets)
        return mean_lpips.mean().item()

    def reload_ema_model(self, rate):
        model_state = {key[7:]:value for key, value in self.ema_state[f"0{int(rate*1000)}"].items()}
        self.ema_model.load_state_dict(model_state)

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

