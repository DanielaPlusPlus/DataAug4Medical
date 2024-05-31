"""
https://github.com/amirhossein-kz/HiFormer/blob/main/models/Decoder.py
"""

import torch.nn as nn
import torch
import numpy as np


class ConvUpsample(nn.Module):
    def __init__(self, in_chans=384, out_chans=[128], upsample=True):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.conv_tower = nn.ModuleList()
        for i, out_ch in enumerate(self.out_chans):
            if i > 0: self.in_chans = out_ch
            self.conv_tower.append(nn.Conv2d(
                self.in_chans, out_ch,
                kernel_size=3, stride=1,
                padding=1, bias=False
            ))
            self.conv_tower.append(nn.GroupNorm(32, out_ch))
            self.conv_tower.append(nn.ReLU(inplace=False))
            if upsample:
                self.conv_tower.append(nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=False))

        self.convs_level = nn.Sequential(*self.conv_tower)

    def forward(self, x):
        return self.convs_level(x)


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        super().__init__(conv2d)


class ReconstructionHead(nn.Sequential):
    def __init__(self, in_channels, out_channels=3, kernel_size=1):
        conv2d = nn.Conv2d(in_channels, 3, kernel_size=kernel_size, padding=kernel_size // 2)
        super().__init__(conv2d)

def SuperpixelPooling(x, SuperP_mask):
    # print(x.shape)#torch.Size([32, 256, 32, 32])
    SPAttention_fea_batch = []
    for sp in range(x.shape[0]):
        mask_value = np.unique(SuperP_mask[sp])
        x_sp = x[sp].reshape(x.shape[2], x.shape[3], x.shape[1])
        avgpool = []
        for v in mask_value:
            avgpool.append(x_sp[SuperP_mask[sp]==v].mean(0))
        avgpool = torch.stack(avgpool)
        # avgpool_sa = self.SA(avgpool) #get vectors
        SPAttention_fea_batch.append(avgpool)
    return SPAttention_fea_batch