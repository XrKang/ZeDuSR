'''
Portions of this code copyright 2017, Clement Pinard
'''

import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np
from utils.Backward_warp_layer import Backward_warp,backwarp
from models.model_lib.submodules import *
import torch.nn.functional as F
# 'Parameter count : 38,676,504 '

class FlowNetS(nn.Module):
    def __init__(self, input_channels = 2, batchNorm=True):
        super(FlowNetS,self).__init__()

        self.batchNorm = batchNorm
        self.conv0 = conv(self.batchNorm,1,64,kernel_size=7,stride=1)
        self.conv1   = conv(self.batchNorm,  64,   64, kernel_size=5, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)

        self.deconv1 = deconv(128,64)
        self.deconv0 = deconv(130,32)

        self.predict_flow2 = predict_flow(128)
        self.predict_flow1 = predict_flow(130)
        self.predict_flow0 = predict_flow(98)

        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         if m.bias is not None:
        #             init.constant_(m.bias,0)
        #         init.constant_(m.weight, 0)
        #
        #      if isinstance(m, nn.ConvTranspose2d):
        #         if m.bias is not None:
        #             init.constant_(m.bias,0)
        #         init.constant_(m.weight,0)



        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.warp = Backward_warp()

    def forward(self, x):
        # input = torch.cat((x,y),dim=1)
        input = x
        out_conv0 = self.conv0(input)
        out_conv1 = self.conv1(out_conv0)
        out_conv2 = self.conv2(out_conv1)

        flow2 = self.predict_flow2(out_conv2)
        flow2_up    = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv1(out_conv2)

        concat1 = torch.cat((out_conv1,out_deconv1,flow2_up),1)
        flow1 = self.predict_flow1(concat1)
        flow1_up    = self.upsampled_flow1_to_0(flow1)
        out_deconv0 = self.deconv0(concat1)

        concat0 = torch.cat((out_conv0,out_deconv0,flow1_up),1)
        flow0 = self.predict_flow0(concat0)

        # warped = self.warp(x, flow0)
        warped = backwarp(x, flow0)
        if self.training:
            return warped,flow0
        else:
            return warped
