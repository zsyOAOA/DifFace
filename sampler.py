#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-13 16:59:27


import os, math, random
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from einops import rearrange
from collections import OrderedDict

from utils import util_net
from utils import util_image
from utils import util_common

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from datapipe.datasets import BaseDataFolder

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer
from facelib.utils.face_restoration_helper import FaceRestoreHelper

class BaseSampler:
    def __init__(self, configs, im_size=512, use_fp16=False):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/sample/
        '''
        self.configs = configs
        self.configs.im_size = im_size
        self.configs.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        if hasattr(self.configs.model.params, 'use_fp16'):
            self.configs.model.params.use_fp16 = use_fp16

        self.setup_dist()  # setup distributed training: self.num_gpus, self.rank

        self.setup_seed()    # setup seed

        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.configs.seed if seed is None else seed
        seed += (self.rank) * 10000
        if self.rank == 0:
            print(f'Setting random seed {seed}', flush=True)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_dist(self):
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(backend='nccl', init_method='env://')

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def build_model(self):
        obj = util_common.get_obj_from_str(self.configs.model.target)
        model = obj(**self.configs.model.params).cuda()
        if not self.configs.model.ckpt_path is None:
            self.load_model(model, self.configs.model.ckpt_path)
        if self.configs.use_fp16:
            model.convert_to_fp16()
        self.model = model
        self.freeze_model(self.model)
        self.model.eval()

    def load_model(self, model, ckpt_path=None):
        if self.rank == 0:
            print(f'Loading from {ckpt_path}...', flush=True)
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        util_net.reload_model(model, ckpt)
        if self.rank == 0:
            print('Loaded Done', flush=True)

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

class DifFaceSampler(BaseSampler):
    def restore_func(self, y0, model_kwargs_ir):
        if model_kwargs_ir is None:
            inputs = y0
        elif 'mask' in model_kwargs_ir:
            inputs = torch.cat([y0, model_kwargs_ir['mask']], dim=1)
        else:
            raise ValueError("Not compativle type for model_kwargs!")

        original_dtype = y0.dtype
        with torch.no_grad():
            out = self.model_ir(inputs.type(self.dtype)).type(original_dtype)

        return out

    def build_model(self):
        super().build_model()

        obj = util_common.get_obj_from_str(self.configs.diffusion.target)
        self.diffusion = obj(**self.configs.diffusion.params)

        # diffused estimator, restoration model
        obj = util_common.get_obj_from_str(self.configs.model_ir.target)
        model_ir = obj(**self.configs.model_ir.params).cuda()
        self.load_model(model_ir, self.configs.model_ir.ckpt_path)
        if self.configs.use_fp16:
            model_ir = model_ir.half()
        self.model_ir = model_ir
        self.model_ir.eval()

        if not self.configs.aligned:
            assert self.num_gpus == 1, 'Only support one gpu for unalinged model'
            # face dection model
            self.face_helper = FaceRestoreHelper(
                    self.configs.detection.upscale,
                    face_size=self.configs.im_size,
                    crop_ratio=(1, 1),
                    det_model = self.configs.detection.det_model,
                    save_ext='png',
                    use_parse=True,
                    use_fp16=True,
                    device=torch.device(f'cuda:{self.rank}'),
                    )

            # background super-resolution
            bg_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            self.bg_model = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=bg_model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True,
                device=torch.device(f'cuda:{self.rank}'),
                )  # need to set False in CPU mode

        # for restoration
        if 'net_hq' in self.configs:
            ckpt_path = self.configs.net_hq.ckpt_path
            params = self.configs.net_hq.get('params', dict)
            net_hq = util_common.get_obj_from_str(self.configs.net_hq.target)(**params)
            self.net_hq = net_hq.cuda()
            self.load_model(self.net_hq, ckpt_path)
            self.net_hq.eval()
        if 'net_lq' in self.configs:
            ckpt_path = self.configs.net_lq.ckpt_path
            params = self.configs.net_lq.get('params', dict)
            net_lq = util_common.get_obj_from_str(self.configs.net_lq.target)(**params)
            self.net_lq = net_lq.cuda()
            self.load_model(self.net_lq, ckpt_path)
            self.net_lq.eval()
            self.freeze_model(self.net_lq)

    def sample_func_ir_aligned(
            self,
            y0,
            start_timesteps=None,
            need_restoration=True,
            model_kwargs_ir=None,
            gamma=0.,
            eta=0.,
            num_update=1,
            regularizer=None,
            cond_kwargs=None,
            ):
        '''
        Input:
            y0: b x c x h x w torch tensor, low-quality image, [0, 1], RGB, float32
            start_timesteps: integer, range [0, num_timesteps-1],
                for accelerated sampling (e.g., 'ddim250'), range [0, 249]
            model_kwargs_ir: additional parameters for restoration model
            gamma: hyper-parameter to regularize the predicted x_start
                    0.5 * || x - x_0 ||_2 + gamma * R(x_0, y_0)      (1)
            num_update: number of update for x_start based on Eq. (1)
            regularizer: the constraint for R in Eq. (1)
            cond_kwargs: extra params for the regrlarizer
            eta: hyper-parameter eta for ddim
        Output:
            sample: n x c x h x w, torch tensor, [0,1], RGB
        '''
        if start_timesteps is None:
            start_timesteps = self.diffusion.num_timesteps

        # basical image restoration
        device = next(self.model.parameters()).device
        y0 = y0.to(device=device)

        h_old, w_old = y0.shape[2:4]
        if not (h_old == self.configs.im_size and w_old == self.configs.im_size):
            y0 = F.interpolate(y0, size=(self.configs.im_size,)*2, mode='bicubic', antialias=True)
            if 'mask' in cond_kwargs:
                cond_kwargs['mask'] = detect_mask(y0, thres=0)

        if need_restoration:
            im_hq = self.restore_func(y0, model_kwargs_ir)
        else:
            im_hq = y0
        im_hq.clamp_(0.0, 1.0)

        # diffuse for im_hq
        yt = self.diffusion.q_sample(
                x_start=util_image.normalize_th(im_hq, mean=0.5, std=0.5, reverse=False),
                t=torch.tensor([start_timesteps,]*im_hq.shape[0], device=device),
                )

        assert yt.shape[-1] == self.configs.im_size and yt.shape[-2] == self.configs.im_size
        sample = self.diffusion.ddim_sample_loop(
                self.model,
                y0=util_image.normalize_th(y0, mean=0.5, std=0.5, reverse=False),
                shape=yt.shape,
                noise=yt,
                start_timesteps=start_timesteps,
                clip_denoised=True,
                denoised_fn=None,
                model_kwargs=None,
                device=None,
                progress=False,
                eta=eta,
                gamma=gamma,
                num_update=num_update,
                regularizer=regularizer,
                cond_kwargs=cond_kwargs,
                )
        sample = util_image.normalize_th(sample, reverse=True).clamp(0.0, 1.0)

        if not (h_old == self.configs.im_size and w_old == self.configs.im_size):
            sample = F.interpolate(sample, size=(h_old, w_old), mode='bicubic', antialias=True).clamp(0.0, 1.0)

        return sample, im_hq

    def sample_func_ir_unaligned(
            self,
            y0,
            micro_bs=16,
            start_timesteps=None,
            need_restoration=True,
            eta=0.0,
            draw_box=False,
            ):
        '''
        Input:
            y0: h x w x c numpy array, uint8, BGR, or image path
            micro_bs: batch size for face restoration
            upscale: upsampling factor for the restorated image
            start_timesteps: integer, range [0, num_timesteps-1],
                for accelerated sampling (e.g., 'ddim250'), range [0, 249]
            draw_box: draw a box for each face
            eta: hyper-parameter eat for ddim
        Output:
            restored_img: h x w x c, numpy array, uint8, BGR
            restored_faces: list, h x w x c, numpy array, uint8, BGR
        '''
        def _process_batch(cropped_faces_list):
            length = len(cropped_faces_list)
            cropped_face_t = torch.cat(
                    util_image.img2tensor(cropped_faces_list, bgr2rgb=True, out_type=self.dtype),
                    axis=0).cuda() / 255.
            restored_faces = self.sample_func_ir_aligned(
                    cropped_face_t,
                    start_timesteps=start_timesteps,
                    need_restoration=need_restoration,
                    gamma=0,
                    eta=eta,
                    model_kwargs_ir=None,
                    )[0]      # [0, 1], b x c x h x w
            return restored_faces

        assert not self.configs.aligned

        self.face_helper.clean_all()
        self.face_helper.read_image(y0)
        num_det_faces = self.face_helper.get_face_landmarks_5(
                only_center_face=False,
                resize=640,
                eye_dist_threshold=5,
                )
        if self.rank == 0:
            print(f'\tdetect {num_det_faces} faces', flush=True)
        # align and warp each face
        self.face_helper.align_warp_face()

        num_cropped_face = len(self.face_helper.cropped_faces)
        if num_cropped_face > micro_bs:
            restored_faces = []
            for idx_start in range(0, num_cropped_face, micro_bs):
                idx_end = idx_start + micro_bs if idx_start + micro_bs < num_cropped_face else num_cropped_face
                current_cropped_faces = self.face_helper.cropped_faces[idx_start:idx_end]
                current_restored_faces = _process_batch(current_cropped_faces)
                current_restored_faces = util_image.tensor2img(
                        list(current_restored_faces.split(1, dim=0)),
                        rgb2bgr=True,
                        min_max=(0, 1),
                        out_type=np.uint8,
                        )
                restored_faces.extend(current_restored_faces)
        else:
            restored_faces = _process_batch(self.face_helper.cropped_faces)
            restored_faces = util_image.tensor2img(
                    list(restored_faces.split(1, dim=0)),
                    rgb2bgr=True,
                    min_max=(0, 1),
                    out_type=np.uint8,
                    )
        for xx in restored_faces:
            self.face_helper.add_restored_face(xx)

        # paste_back
        bg_img = self.bg_model.enhance(
                self.face_helper.input_img,
                outscale=self.configs.detection.upscale,
                )[0]
        self.face_helper.get_inverse_affine(None)
        # paste each restored face to the input image
        restored_img = self.face_helper.paste_faces_to_input_image(
                upsample_img=bg_img,
                draw_box=draw_box,
                )

        return restored_img, restored_faces

    def inference(
            self,
            in_path,
            out_path,
            bs=1,
            start_timesteps=None,
            need_restoration=True,
            gamma=0.,
            num_update=1,
            task='restoration',
            draw_box=False,
            suffix=None,
            eta=0.0,
            mask_back=False,
            ):
        '''
        Input:
            in_path: testing image path or folder
            out_path: folder to save the retorated results
            bs: batch size, totally on all the GPUs
            start_timesteps: integer, range [0, num_timesteps-1],
                for accelerated sampling (e.g., 250), range [0, 249]
            need_restoration: degradation removal with diffused estimator
            gamma: hyper-parameter to regularize the predicted x_start
                    0.5 * || x - x_0 ||_2 + gamma * R(x_0, y_0)      (1)
            num_update: number of update for x_start based on Eq. (1)
            task: 'restoration' or 'inpainting'
                  For inpainting, we assumed that the masked area is initially filled with 0.
            cond_kwargs: extra params for the regrlarizer
            draw_box: draw a box for each face
            eta: eta for ddim
            mask_back: only for inparinting, lq * (1-mask) + res * mask
        '''
        def _process_batch_aligned(y0, cond_kwargs, model_kwargs_ir):
            '''
            y0: b x c x h x w, [0,1]
            '''
            sample, _ = self.sample_func_ir_aligned(
                    y0,
                    start_timesteps,
                    need_restoration=need_restoration,
                    gamma=gamma,
                    num_update=num_update,
                    regularizer= masking_regularizer if task=='inpainting' else None,
                    cond_kwargs=cond_kwargs,
                    eta=eta,
                    model_kwargs_ir=model_kwargs_ir,
                    )
            return sample

        def _process_batch_unaligned(y0):
            '''
            y0: image path or h x w x c numpy array, uint8, BGR
            '''
            restored_img, restored_faces = self.sample_func_ir_unaligned(
                    y0,
                    micro_bs=16,
                    start_timesteps=start_timesteps,
                    need_restoration=need_restoration,
                    draw_box=draw_box,
                    eta=eta,
                    )  # h x w x c, uint8, BGR
            return restored_img, restored_faces

        assert task in ['restoration', 'inpainting']
        if not self.configs.aligned:
            assert task == 'restoration', "Only support image restoration for unalinged image!"

        # prepare result path
        in_path = in_path if isinstance(in_path, Path) else Path(in_path)
        out_path = out_path if isinstance(out_path, Path) else Path(out_path)
        restored_face_dir = out_path / 'restored_faces'
        if self.rank == 0:
            util_common.mkdir(out_path, parents=True)
            util_common.mkdir(restored_face_dir, parents=True)
        if not self.configs.aligned:
            restored_image_dir = out_path / 'restored_image'
            if self.rank == 0:
                util_common.mkdir(restored_image_dir, parents=True)

        if in_path.is_dir():
            if self.configs.aligned:
                dataset = BaseDataFolder(
                        dir_path=in_path,
                        transform_type='default',
                        transform_kwargs={'mean':0, 'std':1.0},
                        need_path=True,
                        im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
                        )
                dataloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=bs,
                        shuffle=False,
                        )
                if self.rank == 0:
                    print(f'Number of testing images: {len(dataset)}', flush=True)
                for data in tqdm(dataloader):
                    micro_batchsize = math.ceil(bs / self.num_gpus)
                    ind_start = self.rank * micro_batchsize
                    ind_end = ind_start + micro_batchsize
                    if ind_start < data['image'].shape[0]:
                        current_y0 = data['image'][ind_start:ind_end]
                        current_path = data['path'][ind_start:ind_end]

                        if task == 'inpainting':
                            mask = detect_mask(current_y0, thres=0)
                            cond_kwargs = model_kwargs_ir = {'mask': mask.cuda()}
                        else:
                            cond_kwargs = model_kwargs_ir = None
                        sample = _process_batch_aligned(current_y0, cond_kwargs, model_kwargs_ir)
                        if (not cond_kwargs is None) and 'mask' in cond_kwargs and mask_back:
                            sample = sample * cond_kwargs['mask'] + current_y0.cuda() * (1-cond_kwargs['mask'])

                        for jj in range(sample.shape[0]):
                            restored_face = sample[jj,].squeeze(0).permute(1,2,0).cpu().numpy()  # h x w x c, [0,1], RGB
                            if suffix == 'gamma':
                                save_path = restored_face_dir / f'{Path(current_path[jj]).stem}_g{gamma:.2f}.png'
                            else:
                                save_path = restored_face_dir / f'{Path(current_path[jj]).stem}.png'
                            util_image.imwrite(restored_face, save_path, chn='rgb', dtype_in='float32')
            else:
                assert self.num_gpus == 1
                im_path_list = [x for x in in_path.glob('*.[jJpP][pPnN]*[gG]')]
                print(f'Number of testing images: {len(im_path_list)}', flush=True)

                for im_path_current in im_path_list:
                    restored_img, restored_faces = _process_batch_unaligned(str(im_path_current))  # h x w x c, uint8, BGR

                    if suffix == 'gamma':
                        save_path = restored_image_dir / f'{im_path_current.stem}_g{gamma:.2f}_s{start_timesteps}.png'
                    else:
                        save_path = restored_image_dir / f'{im_path_current.stem}.png'
                    util_image.imwrite(restored_img, save_path, chn='bgr', dtype_in='uint8')

                    assert isinstance(restored_faces, list)
                    for ii, restored_face in enumerate(restored_faces):
                        save_path = restored_face_dir / f'{im_path_current.stem}_{ii:03d}.png'
                        util_image.imwrite(restored_face, save_path, chn='bgr', dtype_in='uint8')
        else:
            y0 = util_image.imread(in_path, chn='rgb', dtype='float32')
            y0 = util_image.img2tensor(y0, bgr2rgb=False, out_type=torch.float32) # 1 x c x h x w, [0,1]
            if task == 'inpainting':
                mask = detect_mask(y0, thres=0)
                cond_kwargs = model_kwargs_ir = {'mask': mask.cuda()}
            else:
                cond_kwargs = model_kwargs_ir = None

            if self.configs.aligned:
                sample = _process_batch_aligned(y0, cond_kwargs, model_kwargs_ir)
                if (not cond_kwargs is None) and 'mask' in cond_kwargs and mask_back:
                    sample = sample * cond_kwargs['mask'] + y0.cuda() * (1-cond_kwargs['mask'])
                restored_face = sample.squeeze(0).permute(1,2,0).cpu().numpy()  # h x w x c, [0,1], RGB
                if suffix == 'gamma':
                    if 'ddim' in self.configs.diffusion.params.timestep_respacing:
                        save_path = restored_face_dir / f'{in_path.stem}_g{gamma:.2f}_ddim{start_timesteps}_e{eta:.1f}_n{num_update}.png'
                    else:
                        save_path = restored_face_dir / f'{in_path.stem}_g{gamma:.2f}_ddpm{start_timesteps}_n{num_update}.png'
                else:
                    save_path = restored_face_dir / f'{in_path.stem}.png'
                util_image.imwrite(restored_face, save_path, chn='rgb', dtype_in='float32')
            else:
                restored_img, restored_faces = _process_batch_unaligned(str(in_path))  # h x w x c, uint8, BGR
                if suffix == 'gamma':
                    save_path = restored_image_dir / f'{in_path.stem}_g{gamma:.2f}.png'
                else:
                    save_path = restored_face_dir / f'{in_path.stem}.png'
                util_image.imwrite(restored_img, save_path, chn='bgr', dtype_in='uint8')

                assert isinstance(restored_faces, list)
                for ii, restored_face in enumerate(restored_faces):
                    save_path = restored_face_dir / f'{in_path.stem}_{ii:03d}.png'
                    util_image.imwrite(restored_face, save_path, chn='bgr', dtype_in='uint8')

        if self.num_gpus > 1:
            dist.barrier()

        if self.rank == 0:
            print(f'Please enjoy the results in {str(out_path)}...', flush=True)

@torch.enable_grad()
def masking_regularizer(y0, x0, cond_kwargs):
    '''
    Input:
        y0: low-quality image, b x c x h x w, [-1, 1]
        x0: predicted high-quality image, b x c x h x w, [-1, 1]
        cond_kwargs: additional network parameters.
    '''
    mask = cond_kwargs['mask']
    if 'vqgan' in cond_kwargs:
        pred = cond_kwargs['vqgan'].decode(x0)
    else:
        pred = x0

    loss = (F.mse_loss(pred, y0, reduction='none') * (1 - mask)).sum()

    return loss

def detect_mask(y0, thres):
    '''
    Input:
        y0: low-quality image, b x c x h x w , [0, 1]
    '''
    ysum = torch.sum(y0, dim=1, keepdim=True)
    mask = torch.where(ysum==thres, torch.ones_like(ysum), torch.zeros_like(ysum))
    return mask

if __name__ == '__main__':
    import argparse
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--save_dir",
            type=str,
            default="./save_dir",
            help="Folder to save the checkpoints and training log",
            )
    parser.add_argument(
            "--gpu_id",
            type=str,
            default='0',
            help="GPU Index, e.g., 025",
            )
    parser.add_argument(
            "--cfg_path",
            type=str,
            default='./configs/sample/iddpm_ffhq256.yaml',
            help="Path of config files",
            )
    parser.add_argument(
            "--bs",
            type=int,
            default=32,
            help="Batch size",
            )
    parser.add_argument(
            "--num_images",
            type=int,
            default=3000,
            help="Number of sampled images",
            )
    args = parser.parse_args()

    configs = OmegaConf.load(args.cfg_path)
    configs.gpu_id = args.gpu_id
    # configs.diffusion.params.timestep_respacing = args.timestep_respacing

    # sampler_dist = DiffusionSampler(configs)
    sampler_dist = LDMSampler(configs)

    sampler_dist.sample_func(
            bs=args.bs,
            num_images=args.num_images,
            save_dir=args.save_dir,
            )

