"""
https://github.com/Minerva-J/Pytorch-Segmentation-multi-models/blob/master/models/AttentionUnet/AttUnet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        # d1 = self.Conv_1x1(d2)
        d1 = F.sigmoid(self.Conv_1x1(d2).squeeze(1))

        return d1




class AttU_Net(nn.Module):
    def __init__(self, img_ch=3):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = F.sigmoid(self.Conv_1x1(d2).squeeze(1))

        return d1

class UNet_3Heads_1Decoder(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(UNet_3Heads_1Decoder, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.SP = self.SuperpixelPooling
        self.seg_head = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        # self.cls_head = nn.Linear(64, 64)  #local projection head for cosface classification head
        self.rec_head = nn.Conv2d(64, img_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, train=False, superpixel_map=None):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        seg_out = F.sigmoid(self.seg_head(d2).squeeze(1))

        if train==False:
            return seg_out
        else:
            recontrust_out = self.rec_head(d2)
            SPAttention_fea_batch = self.SP(d2, superpixel_map)  # list(32*[n, 256])
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


    # def SuperpixelSelfAttentionPooling(self, x, SuperP_mask):
    #     # print(x.shape)#torch.Size([32, 256, 32, 32])
    #     SPAttention_fea_batch = []
    #     for sp in range(x.shape[0]):
    #         mask_value = np.unique(SuperP_mask[sp])
    #         x_sp = x[sp].reshape(x.shape[2], x.shape[3], x.shape[1])
    #
    #         avgpool = []
    #         for v in mask_value:
    #             avgpool.append(x_sp[SuperP_mask[sp]==v].mean(0))
    #         avgpool = torch.stack(avgpool)
    #         avgpool_sa = self.SA(avgpool) #get vectors
    #         SPAttention_fea_batch.append(avgpool_sa)
    #     return SPAttention_fea_batch

    def SuperpixelPooling(self, x, SuperP_mask):
        # print(x.shape)#torch.Size([32, 256, 32, 32])
        SPAttention_fea_batch = []
        for sp in range(len(SuperP_mask)):
            mask_value = np.unique(SuperP_mask[sp])
            x_sp = x[sp].reshape(x.shape[2], x.shape[3], x.shape[1])
            avgpool = []
            for v in mask_value:
                avgpool.append(x_sp[SuperP_mask[sp]==v].mean(0))
            avgpool = torch.stack(avgpool)
            # avgpool_sa = self.SA(avgpool) #get vectors
            SPAttention_fea_batch.append(avgpool)
        return SPAttention_fea_batch


# """https://blog.csdn.net/beilizhang/article/details/115282604"""
# class LayerNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-12):
#         """Construct a layernorm module in the TF style (epsilon inside the square root).
#         """
#         super(LayerNorm, self).__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.bias = nn.Parameter(torch.zeros(hidden_size))
#         self.variance_epsilon = eps
#
#     def forward(self, x):
#         u = x.mean(-1, keepdim=True)
#         s = (x - u).pow(2).mean(-1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.variance_epsilon)
#         return self.weight * x + self.bias
#
# class SelfAttention(nn.Module):
#     def __init__(self, input_size):
#         super(SelfAttention,self).__init__()
#
#         self.query = nn.Linear(input_size, input_size)
#         self.key = nn.Linear(input_size, input_size)
#         self.value = nn.Linear(input_size, input_size)
#
#         # self.out_dropout = nn.Dropout(dropout_prob)
#         self.hidden_size = input_size
#         self.LayerNorm = LayerNorm(input_size, eps=1e-12)
#
#     def forward(self, input_tensor):
#         """input tensor (n,d)"""
#         query_layer = self.query(input_tensor)
#         key_layer = self.key(input_tensor)
#         value_layer = self.value(input_tensor)
#
#         # Take the dot product between "query" and "key" to get the raw attention scores.
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_scores = attention_scores / math.sqrt(self.hidden_size)
#
#         # Normalize the attention scores to probabilities.
#         attention_probs = nn.Softmax(dim=-1)(attention_scores)
#         # attention_probs = self.out_dropout(attention_probs)
#
#         hidden_states = torch.matmul(attention_probs, value_layer)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)
#
#         return hidden_states