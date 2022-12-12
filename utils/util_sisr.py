#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2021-12-07 21:37:58

import sys
import math
import torch
import numpy as np
import scipy.ndimage as snd
from scipy.special import softmax
from scipy.interpolate import interp2d

import torch.nn.functional as F


from . import util_image
from ResizeRight.resize_right import resize

def modcrop(im, sf):
    h, w = im.shape[:2]
    h -= (h % sf)
    w -= (w % sf)
    return im[:h, :w,]

#--------------------------------------------Kernel-----------------------------------------------
def sigma2kernel(sigma, k_size=21, sf=3, shift=False):
    '''
    Generate Gaussian kernel according to cholesky decomposion.
    Input:
        sigma: N x 1 x 2 x 2 torch tensor, covariance matrix
        k_size: integer, kernel size
        sf: scale factor
    Output:
        kernel: N x 1 x k x k torch tensor
    '''
    try:
        sigma_inv = torch.inverse(sigma)
    except:
        sigma_disturb = sigma + torch.eye(2, dtype=sigma.dtype, device=sigma.device).unsqueeze(0).unsqueeze(0) * 1e-5
        sigma_inv = torch.inverse(sigma_disturb)

    # Set expectation position (shifting kernel for aligned image)
    if shift:
        center = k_size // 2 + 0.5 * (sf - k_size % 2)                         # + 0.5 * (sf - k_size % 2)
    else:
        center = k_size // 2

    # Create meshgrid for Gaussian
    X, Y = torch.meshgrid(torch.arange(k_size), torch.arange(k_size))
    Z = torch.stack((X, Y), dim=2).to(device=sigma.device, dtype=sigma.dtype).view(1, -1, 2, 1)      # 1 x k^2 x 2 x 1

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - center                                                        # 1 x k^2 x 2 x 1
    ZZ_t = ZZ.permute(0, 1, 3, 2)                                          # 1 x k^2 x 1 x 2
    ZZZ = -0.5 * ZZ_t.matmul(sigma_inv).matmul(ZZ).squeeze(-1).squeeze(-1) # N x k^2
    kernel = F.softmax(ZZZ, dim=1)                                         # N x k^2

    return kernel.view(-1, 1, k_size, k_size)                # N x 1 x k x k

def shifted_anisotropic_Gaussian(k_size=21, sf=4, lambda_1=1.2, lambda_2=5., theta=0, shift=True):
    '''
    # modified version of https://github.com/cszn/USRNet/blob/master/utils/utils_sisr.py
    '''
    # set covariance matrix
    Lam = np.diag([lambda_1, lambda_2])
    U = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
    sigma = U @ Lam @ U.T                                 # 2 x 2
    inv_sigma = np.linalg.inv(sigma)[None, None, :, :]    # 1 x 1 x 2 x 2

    # set expectation position (shifting kernel for aligned image)
    if shift:
        center = k_size // 2 + 0.5*(sf - k_size % 2)
    else:
        center = k_size // 2

    # Create meshgrid for Gaussian
    X, Y = np.meshgrid(range(k_size), range(k_size))
    Z = np.stack([X, Y], 2).astype(np.float32)[:, :, :, None]                  # k x k x 2 x 1

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - center
    ZZ_t = ZZ.transpose(0,1,3,2)
    ZZZ = -0.5 * np.squeeze(ZZ_t @ inv_sigma  @ ZZ).reshape([1, -1])
    kernel = softmax(ZZZ, axis=1).reshape([k_size, k_size]) # k x k

    # The convariance of the marginal distributions along x and y axis
    s1, s2 = sigma[0, 0], sigma[1, 1]
    # Pearson corrleation coefficient
    rho = sigma[0, 1] / (math.sqrt(s1) * math.sqrt(s2))
    kernel_infos = np.array([s1, s2, rho])   # (3,)

    return kernel, kernel_infos

#------------------------------------------Degradation-------------------------------------------
def imconv_np(im, kernel, padding_mode='reflect', correlate=False):
    '''
    Image convolution or correlation.
    Input:
        im: h x w x c numpy array
        kernel: k x k numpy array
        padding_mode: 'reflect', 'constant' or 'wrap'
    '''
    if kernel.ndim != im.ndim: kernel = kernel[:, :, np.newaxis]

    if correlate:
        out = snd.correlate(im, kernel, mode=padding_mode)
    else:
        out = snd.convolve(im, kernel, mode=padding_mode)

    return out

def conv_multi_kernel_tensor(im_hr, kernel, sf, downsampler):
    '''
    Degradation model by Pytorch.
    Input:
        im_hr: N x c x h x w
        kernel: N x 1 x k x k
        sf: scale factor
    '''
    im_hr_pad = F.pad(im_hr, (kernel.shape[-1] // 2,)*4, mode='reflect')
    im_blur = F.conv3d(im_hr_pad.unsqueeze(0), kernel.unsqueeze(1), groups=im_hr.shape[0])
    if downsampler.lower() == 'direct':
        im_blur = im_blur[0, :, :, ::sf, ::sf]      # N x c x ...
    elif downsampler.lower() == 'bicubic':
        im_blur = resize(im_blur, scale_factors=1/sf)
    else:
        sys.exit('Please input the corrected downsampler: Direct or Bicubic!')

    return im_blur

def tidy_kernel(kernel, expect_size=21):
    '''
    Input:
        kernel: p x p numpy array
    '''
    k_size = kernel.shape[-1]
    kernel_new = np.zeros([expect_size, expect_size], dtype=kernel.dtype)
    if expect_size >= k_size:
        start_ind = expect_size // 2 - k_size // 2
        end_ind = start_ind + k_size
        kernel_new[start_ind:end_ind, start_ind:end_ind] = kernel
    elif expect_size < k_size:
        start_ind = k_size // 2 - expect_size // 2
        end_ind = start_ind + expect_size
        kernel_new = kernel[start_ind:end_ind, start_ind:end_ind]
        kernel_new /= kernel_new.sum()

    return kernel_new

def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf-1)*0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w-1)
    y1 = np.clip(y1, 0, h-1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x

#-----------------------------------------Transform--------------------------------------------
class Bicubic:
    def __init__(self, scale=0.25):
        self.scale = scale

    def __call__(self, im, scale=None, out_shape=None):
        scale = self.scale if scale is None else scale
        out = resize(im, scale_factors=scale, out_shape=None)
        return out
