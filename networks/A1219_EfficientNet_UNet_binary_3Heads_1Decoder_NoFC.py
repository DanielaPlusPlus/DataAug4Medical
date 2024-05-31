"""
https://github.com/zhoudaxia233/EfficientUnet-PyTorch/blob/master/efficientunet/efficientunet.py
"""




from collections import OrderedDict
from .A1216_EfficientNet_layers import *
from .A1216_EfficientNet import EfficientNet
import numpy as np


__all__ = ['EfficientUnet', 'get_efficientunet_b0', 'get_efficientunet_b1', 'get_efficientunet_b2',
           'get_efficientunet_b3', 'get_efficientunet_b4', 'get_efficientunet_b5', 'get_efficientunet_b6',
           'get_efficientunet_b7'
           , 'get_efficientunet_3heads_1decoder_b2']


def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):

        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f'blocks_{count}_output_batch_norm':
                    count += 1
                    shape = output.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = output

                elif module.name == 'head_swish':
                    # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                    # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                    # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                    # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                    # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                    # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                    blocks.popitem()
                    blocks[module.name] = output

            except AttributeError:
                pass

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)

    # make a forward pass to trigger the hooks
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return blocks


class EfficientUnet(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=True):
        super().__init__()

        self.encoder = encoder
        self.concat_input = concat_input

        self.up_conv1 = up_conv(self.n_channels, 512)
        self.double_conv1 = double_conv(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(self.size[4], 32)

        # self.final_conv = nn.Conv2d(self.size[5], out_channels, kernel_size=1)
        self.seg_head = nn.Conv2d(self.size[5], 1, kernel_size=1, stride=1, padding=0)

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    def forward(self, x): #(4,3,224,224)
        input_ = x

        blocks = get_blocks_to_be_concat(self.encoder, x)
        _, x = blocks.popitem()

        x = self.up_conv1(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv1(x) #(4,512,14,14)

        x = self.up_conv2(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv2(x) #(4,256,28,28)

        x = self.up_conv3(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv3(x) #(4,128,56,56)

        x = self.up_conv4(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv4(x)  #(4,64,112,112)

        if self.concat_input:
            x = self.up_conv_input(x)
            x = torch.cat([x, input_], dim=1)
            x = self.double_conv_input(x)  #(4,32,224,224)

        # x = self.final_conv(x) #(4,3,224,224)
        x = F.sigmoid(self.seg_head(x).squeeze(1))

        return x


class EfficientUnet_3heads_1decoder(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=True):
        super().__init__()

        self.encoder = encoder
        self.concat_input = concat_input

        self.up_conv1 = up_conv(self.n_channels, 512)
        self.double_conv1 = double_conv(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(self.size[4], 32)

        # self.final_conv = nn.Conv2d(self.size[5], out_channels, kernel_size=1)
        self.SP = self.SuperpixelPooling
        # self.SA = SelfAttention(input_size=32)
        # self.SAP = self.SuperpixelSelfAttentionPooling
        self.seg_head = nn.Conv2d(self.size[5], 1, kernel_size=1, stride=1, padding=0)
        # self.cls_head = nn.Linear(self.size[5], 64)  #local projection head for cosface classification head
        self.rec_head = nn.Conv2d(self.size[5], 3, kernel_size=1, stride=1, padding=0)

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    def forward(self, x, train=False, superpixel_map=None): #(4,3,224,224)
        input_ = x

        blocks = get_blocks_to_be_concat(self.encoder, x)
        _, x = blocks.popitem()

        x = self.up_conv1(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv1(x) #(4,512,14,14)

        x = self.up_conv2(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv2(x) #(4,256,28,28)

        x = self.up_conv3(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv3(x) #(4,128,56,56)

        x = self.up_conv4(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv4(x)  #(4,64,112,112)

        if self.concat_input:
            x = self.up_conv_input(x)
            x = torch.cat([x, input_], dim=1)
            x = self.double_conv_input(x)  #(4,32,224,224)

        # # x = self.final_conv(x) #(4,3,224,224)
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
    #             avgpool.append(x_sp[SuperP_mask[sp]==v].mean(0))
    #         avgpool = torch.stack(avgpool)
    #         avgpool_sa = self.SA(avgpool) #get vectors
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

def get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b1(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b1', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b2(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b2', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b3(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b3', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b4(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b4', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b5(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b5', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b6(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b6', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b7(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b7', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model





def get_efficientunet_3heads_1decoder_b2(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b2', pretrained=pretrained)
    model = EfficientUnet_3heads_1decoder(encoder, out_channels=out_channels, concat_input=concat_input)
    return model