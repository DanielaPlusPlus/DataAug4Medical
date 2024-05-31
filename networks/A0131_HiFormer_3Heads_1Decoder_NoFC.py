"""
https://github.com/amirhossein-kz/HiFormer/blob/main/models/HiFormer.py
"""
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from networks.A0131_HiFormer_Encoder import All2Cross
from networks.A0131_HiFormer_Decoder_3Heads_1Decoder_NoFC import ConvUpsample, SegmentationHead, ReconstructionHead, SuperpixelPooling
import torch.nn.functional as F

class HiFormer(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3, n_classes=9):
        super().__init__()
        self.img_size = img_size
        self.patch_size = [4, 16]
        self.n_classes = n_classes
        self.All2Cross = All2Cross(config = config, img_size= img_size, in_chans=in_chans)

        self.ConvUp_s = ConvUpsample(in_chans=384, out_chans=[128 ,128], upsample=True)
        self.ConvUp_l = ConvUpsample(in_chans=96, upsample=False)

        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=n_classes,
            kernel_size=1,
        )

        self.reconstruction_head = ReconstructionHead(
            in_channels=16,
            out_channels=3,
            kernel_size=1,
        )

        self.SP = SuperpixelPooling

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                128, 16,
                kernel_size=1, stride=1,
                padding=0, bias=True),
            # nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )

    def forward(self, x, train=False, superpixel_map=None):
        xs = self.All2Cross(x)
        embeddings = [x[:, 1:] for x in xs]
        reshaped_embed = []
        for i, embed in enumerate(embeddings):

            embed = Rearrange('b (h w) d -> b d h w', h=(self.img_size//self.patch_size[i]), w=(self.img_size//self.patch_size[i]))(embed)
            embed = self.ConvUp_l(embed) if i == 0 else self.ConvUp_s(embed)

            reshaped_embed.append(embed)

        C = reshaped_embed[0] + reshaped_embed[1]
        C = self.conv_pred(C)

        # out = self.segmentation_head(C)  # multi-class classification
        seg_out = F.sigmoid(self.segmentation_head(C).squeeze(1))  # binary classification

        if train==False:
            return seg_out
        else:
            recontrust_out = self.reconstruction_head(C)
            SPAttention_fea_batch = self.SP(C, superpixel_map)  # list(32*[n, 256])
            SurPLocalCls_batch = []
            for sp in range(len(SPAttention_fea_batch)):
                x_locals_out_sp = []
                for tk in range(SPAttention_fea_batch[sp].shape[0]):
                    # cls_out = self.cls_head(SPAttention_fea_batch[sp][tk])
                    # x_locals_out_sp.append(cls_out)
                    x_locals_out_sp.append(SPAttention_fea_batch[sp][tk])

                x_locals_out_sp = torch.stack(x_locals_out_sp)
                SurPLocalCls_batch.extend(x_locals_out_sp)
            SurPLocalCls_batch = torch.stack(SurPLocalCls_batch)
            # """output: seg_pred; reconstruction out; local preds for superpixels; local feas for superpixels"""
            # return seg_out, recontrust_out, SurPLocalCls_batch, SPAttention_fea_batch
            """output: seg_pred; reconstruction out; local preds for superpixels"""
            return seg_out, recontrust_out, SurPLocalCls_batch
