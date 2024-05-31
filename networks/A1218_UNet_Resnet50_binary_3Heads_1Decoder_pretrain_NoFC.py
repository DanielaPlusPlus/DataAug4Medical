import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
import math

resnet = torchvision.models.resnet.resnet50(pretrained=True)


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """

        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


# class UNetWithResnet50Encoder(nn.Module):
#     DEPTH = 6
#
#     def __init__(self, n_classes=2):
#         super().__init__()
#         resnet = torchvision.models.resnet.resnet50(pretrained=True)
#         down_blocks = []
#         up_blocks = []
#         self.input_block = nn.Sequential(*list(resnet.children()))[:3]
#         self.input_pool = list(resnet.children())[3]
#         for bottleneck in list(resnet.children()):
#             if isinstance(bottleneck, nn.Sequential):
#                 down_blocks.append(bottleneck)
#         self.down_blocks = nn.ModuleList(down_blocks)
#         self.bridge = Bridge(2048, 2048)
#         up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
#         up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
#         up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
#         up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
#                                                     up_conv_in_channels=256, up_conv_out_channels=128))
#         up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
#                                                     up_conv_in_channels=128, up_conv_out_channels=64))
#
#         self.up_blocks = nn.ModuleList(up_blocks)
#
#         # self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)
#         self.seg_head = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x, with_output_feature_map=False):
#         pre_pools = dict()
#         pre_pools[f"layer_0"] = x
#         x = self.input_block(x)
#         pre_pools[f"layer_1"] = x
#         x = self.input_pool(x)
#
#         for i, block in enumerate(self.down_blocks, 2):
#             x = block(x)
#             if i == (UNetWithResnet50Encoder.DEPTH - 1):
#                 continue
#             pre_pools[f"layer_{i}"] = x
#
#         x = self.bridge(x)
#
#         for i, block in enumerate(self.up_blocks, 1):
#             key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
#             x = block(x, pre_pools[key])
#         output_feature_map = x
#         # x = self.out(x)
#         x = F.sigmoid(self.seg_head(x).squeeze(1))
#         del pre_pools
#         if with_output_feature_map:
#             return x, output_feature_map
#         else:
#             return x
#
# model = UNetWithResnet50Encoder().cuda()
# inp = torch.rand((2, 3, 224, 224)).cuda()
# out = model(inp)
# print(out.shape)


class UNet_R50_3Heads_1Decoder(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=2):
        super().__init__()
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        # self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)
        # self.SA = SelfAttention(input_size=32)
        # self.SAP = self.SuperpixelSelfAttentionPooling
        self.SP = self.SuperpixelPooling
        self.seg_head = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        # self.cls_head = nn.Linear(64, 64)  #local projection head for cosface classification head
        self.rec_head = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x, train=False, superpixel_map=None):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNet_R50_3Heads_1Decoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNet_R50_3Heads_1Decoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        # x = self.out(x)
        # x = F.sigmoid(self.seg_head(x).squeeze(1))
        # # del pre_pools
        # # if with_output_feature_map:
        # #     return x, output_feature_map
        # # else:
        # #     return x

        seg_out = F.sigmoid(self.seg_head(x).squeeze(1))
        if train==False:
            return seg_out
        else:
            recontrust_out = self.rec_head(x)
            SPAttention_fea_batch = self.SP(x, superpixel_map)  # list(32*[n, 256])
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
    #             avgpool.append(x_sp[SuperP_mask[sp] == v].mean(0))
    #         avgpool = torch.stack(avgpool)
    #         avgpool_sa = self.SA(avgpool)  # get vectors
    #         SPAttention_fea_batch.append(avgpool_sa)
    #     return SPAttention_fea_batch

    def SuperpixelPooling(self, x, SuperP_mask):
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

model = UNet_R50_3Heads_1Decoder().cuda()
inp = torch.rand((2, 3, 224, 224)).cuda()
out = model(inp)
print(out.shape)
