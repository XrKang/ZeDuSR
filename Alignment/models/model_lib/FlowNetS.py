'''
Portions of this code copyright 2017, Clement Pinard
'''

import torch
import torch.nn as nn
from torch.nn import init

from Backward_warp_layer import Backward_warp,backwarp
from submodules import *
import torch.nn.functional as F

# 'Parameter count : 38,676,504 '

class FlowNetS(nn.Module):
    def __init__(self, input_channels=6, batchNorm=True):
        super(FlowNetS, self).__init__()

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, input_channels, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

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
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2,

class FlowNet2S(FlowNetS):
    def __init__(self, batchNorm=False, div_flow=20):
        super(FlowNet2S, self).__init__(input_channels=6, batchNorm=batchNorm)
        self.div_flow = div_flow
        self.rgb_max = 1.0

    def forward(self, x1, x2):
        # rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1,))
        # x = (inputs - rgb_mean) / self.rgb_max
        # x = torch.cat((x[:, :, 0, :, :], x[:, :, 1, :, :]), dim=1)
        inputs = torch.cat((x1.unsqueeze(2), x2.unsqueeze(2)), dim=2)

        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat((x[:, :, 0, :, :], x[:, :, 1, :, :]), dim=1)

        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)

        return self.upsample1(flow2 * self.div_flow)


# class FlowNetS(nn.Module):
#     def __init__(self, input_channels=6, batchNorm=True):
#         super(FlowNetS,self).__init__()
#
#         self.batchNorm = batchNorm
#         self.conv0 = conv(self.batchNorm, input_channels, 64, kernel_size=7, stride=1)
#         self.conv1   = conv(self.batchNorm,  64,   64, kernel_size=5, stride=2)
#         self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
#         self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
#         self.conv3_1 = conv(self.batchNorm, 256,  256)
#         self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
#         self.conv4_1 = conv(self.batchNorm, 512,  512)
#
#         self.deconv3 = deconv(512,128)
#         self.deconv2 = deconv(386,64)
#         self.deconv1 = deconv(194,64)
#         self.deconv0 = deconv(130,32)
#
#         self.predict_flow4 = predict_flow(512)
#         self.predict_flow3 = predict_flow(386)
#         self.predict_flow2 = predict_flow(194)
#         self.predict_flow1 = predict_flow(130)
#         self.predict_flow0 = predict_flow(98)
#
#         self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
#         self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
#         self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
#         self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 if m.bias is not None:
#                     init.uniform_(m.bias)
#                 init.xavier_uniform_(m.weight)
#
#             if isinstance(m, nn.ConvTranspose2d):
#                 if m.bias is not None:
#                     init.uniform_(m.bias)
#                 init.xavier_uniform_(m.weight)
#
#         self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
#
#     def forward(self, x, y):
#         input = torch.cat((x, y),dim=1)
#         # input = x
#         out_conv0 = self.conv0(input)
#         out_conv1 = self.conv1(out_conv0)
#         out_conv2 = self.conv2(out_conv1)
#         out_conv3 = self.conv3_1(self.conv3(out_conv2))
#         out_conv4 = self.conv4_1(self.conv4(out_conv3))
#
#         flow4       = self.predict_flow4(out_conv4)
#         flow4_up    = self.upsampled_flow4_to_3(flow4)
#         out_deconv3 = self.deconv3(out_conv4)
#
#         concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
#         flow3       = self.predict_flow3(concat3)
#         flow3_up    = self.upsampled_flow3_to_2(flow3)
#         out_deconv2 = self.deconv2(concat3)
#
#         concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
#         flow2 = self.predict_flow2(concat2)
#         flow2_up    = self.upsampled_flow2_to_1(flow2)
#         out_deconv1 = self.deconv1(concat2)
#
#         concat1 = torch.cat((out_conv1,out_deconv1,flow2_up),1)
#         flow1 = self.predict_flow1(concat1)
#         flow1_up    = self.upsampled_flow1_to_0(flow1)
#         out_deconv0 = self.deconv0(concat1)
#
#         concat0 = torch.cat((out_conv0,out_deconv0,flow1_up),1)
#         flow0 = self.predict_flow0(concat0)
#
#         warped = backwarp(x, flow0)
#         return warped
#
#         # if self.training:
#         #     return warped,flow0
#         # else:
#         #     return warped

if __name__ == '__main__':
    model = FlowNet2S()
    model = model.cuda()
    path = 'D:/!Never Give up/zeroLens/camsr2/FlowNet2-S_checkpoint.pth.tar'
    loadModel = torch.load(path)['state_dict']
    print(type(loadModel), loadModel.keys())
    print(model.state_dict().keys())
    model.load_state_dict(loadModel)
    x1 = torch.rand((1, 3, 64, 64)).cuda()
    x2 = torch.rand((1, 3, 64, 64)).cuda()
    flow = model(x1, x2)
    print(flow.shape)