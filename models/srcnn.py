#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-12 20:35:28

import math
from torch import nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self, in_chns, out_chns=None, num_chns=64, depth=8, sf=4):
        super().__init__()
        self.sf = sf
        out_chns = in_chns if out_chns is None else out_chns

        self.head = nn.Conv2d(in_chns, num_chns, kernel_size=5, padding=2)

        body = []
        for _ in range(depth-1):
            body.append(nn.Conv2d(num_chns, num_chns, kernel_size=5, padding=2))
            body.append(nn.LeakyReLU(0.2, inplace=True))
        self.body = nn.Sequential(*body)

        tail = []
        for _ in range(int(math.log(sf, 2))):
            tail.append(nn.Conv2d(num_chns, num_chns*4, kernel_size=3, padding=1))
            tail.append(nn.LeakyReLU(0.2, inplace=True))
            tail.append(nn.PixelShuffle(2))
        tail.append(nn.Conv2d(num_chns, out_chns, kernel_size=5, padding=2))
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        y = self.head(x)
        y = self.body(y)
        y = self.tail(y)
        return y

class SRCNNFSR(nn.Module):
    def __init__(self, in_chns, down_scale_factor=2, num_chns=64, depth=8, sf=4):
        super().__init__()
        self.sf = sf

        head = []
        in_chns_shuffle = in_chns * 4
        assert num_chns % 4 == 0
        for ii in range(int(math.log(down_scale_factor, 2))):
            head.append(nn.PixelUnshuffle(2))
            head.append(nn.Conv2d(in_chns_shuffle, num_chns, kernel_size=3, padding=1))
            if ii + 1 < int(math.log(down_scale_factor, 2)):
                head.append(nn.Conv2d(num_chns, num_chns//4, kernel_size=5, padding=2))
                head.append(nn.LeakyReLU(0.2, inplace=True))
                in_chns_shuffle = num_chns
        self.head = nn.Sequential(*head)

        body = []
        for _ in range(depth-1):
            body.append(nn.Conv2d(num_chns, num_chns, kernel_size=5, padding=2))
            body.append(nn.LeakyReLU(0.2, inplace=True))
        self.body = nn.Sequential(*body)

        tail = []
        for _ in range(int(math.log(down_scale_factor, 2))):
            tail.append(nn.Conv2d(num_chns, num_chns, kernel_size=3, padding=1))
            tail.append(nn.LeakyReLU(0.2, inplace=True))
            tail.append(nn.PixelShuffle(2))
            num_chns //= 4
        tail.append(nn.Conv2d(num_chns, in_chns, kernel_size=5, padding=2))
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        y = self.head(x)
        y = self.body(y)
        y = self.tail(y)
        return y
