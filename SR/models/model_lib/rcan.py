# from common import *
from .common import *

import torch.nn as nn

## Channel Attention Block
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(RCAN, self).__init__()
        n_resgroups = args.n_resgroups  # n_RGs/n_RCABs:output = (n_resblock*RCAB+Conv2d)(input)+input
        n_resblocks = args.n_resblocks  # n_Resblocks: RCAB中Reblock的个数
        n_feats = args.n_feats  # n_channel
        kernel_size = 3  # convolution kernel size
        reduction = args.n_reduction  #

        scale = args.scale  # 上采样器的放大倍数
        act = nn.ReLU()

        # define head module,输入的channel数注意修改
        modules_head = [conv(3, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

        self.weight_init(self.head)
        self.weight_init(self.body)
        self.weight_init(self.tail)

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            nn.init.kaiming_normal_(m.bias, mode='fan_out')

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x

if __name__ == '__main__':
    import argparse
    # import numpy as np
    from thop.profile import profile
    parser = argparse.ArgumentParser(description="DepthSR")
    arg = parser.parse_args()
    arg.n_feats = 32
    arg.scale = 4

    arg.n_resgroups = 2
    arg.n_resblocks = 2
    arg.n_reduction = 16

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    model = RCAN(arg).to(device)
    # model.train()
    model.eval()
    dsize1 = (1, 3, 32, 32)

    name = "SR"
    input1 = torch.randn(dsize1).to(device)

    # Time consuming
    # Warn-up
    _ = model(input1)