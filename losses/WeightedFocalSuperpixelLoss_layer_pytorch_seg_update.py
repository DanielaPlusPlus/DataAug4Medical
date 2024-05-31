from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
# from torch.nn.parameter import Parameter
import math
"""
m_new = 0.5m + 2r(1-r)
https://github.com/MuggleWang/CosFace_pytorch
"""
def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1,w2).clamp(min=eps)#x1.x2/!x1x2!

class CosFaceLoss(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40,gamma = 3.5):
        super(CosFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features)).cuda()
        nn.init.xavier_uniform_(self.weight)
        self.gamma = gamma

    def infer(self,input):
        cosine = cosine_sim(input, self.weight.cuda())
        return cosine

    def forward(self, input, label, alpha):
        input = input.float().cuda()
        alpha = alpha.float().cuda()
        cosine = cosine_sim(input, self.weight)
        # print('cosine:',  cosine.shape)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)#scatter_(dim, index, src) → Tensor,通过一个张量 src  来修改另一个张量，哪个元素需要修改、用 src 中的哪个元素来修改由 dim 和 index 决定
        # dynamic_m = [self.m*4*r*(1-r) for r in alpha]
        dynamic_m = [self.m * (0.5 + 2 * r * (1 - r)) for r in alpha]
        dynamic_m = torch.tensor(dynamic_m).unsqueeze(1).cuda()
        # print('alpha:', alpha, alpha.shape)
        # print('dynamic_m:', dynamic_m, dynamic_m.shape)
        output = self.s * (cosine - dynamic_m*one_hot)
        # print(output.shape)
        # loss = F.cross_entropy(output,label)
        # return loss

        # print('output:', output, output.shape)
        BCE_loss = F.cross_entropy(output,label, reduction='none').cuda()
        # print('output:', output.shape)
        # print('label:', label, label.shape)
        # print('BCE_loss:', BCE_loss)
        pt = torch.exp(-BCE_loss)
        F_loss = alpha * (1 - pt) ** self.gamma * BCE_loss
        # print(F_loss.mean())
        return F_loss.mean()


    # def __repr__(self):
    #     return self.__class__.__name__ + '(' \
    #            + 'in_features=' + str(self.in_features) \
    #            + ', out_features=' + str(self.out_features) \
    #            + ', s=' + str(self.s) \
    #            + ', m=' + str(self.m) + ')'

"""
#对比方法
class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'
"""