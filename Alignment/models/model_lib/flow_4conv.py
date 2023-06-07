import math
import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
from utils.Backward_warp_layer import Backward_warp
import numbers


# class Conv_ReLU_Block(nn.Module):
#     def __init__(self):
#         super(Conv_ReLU_Block, self).__init__()
#         self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         return self.relu(self.conv(x))
        
class CONV4(nn.Module):
    def __init__(self):
        super(CONV4, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64,kernel_size=7, stride=1, padding=3, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32,kernel_size=7, stride=1, padding=3, bias=False)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)
        # self.conv4 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=7, stride=1, padding=3, bias=False)

        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                # nn.init.kaiming_normal(m,mode='fan_out')
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        # out=torch.cat([x,y],1)
        out = self.relu(self.input(x))
        out = self.relu(self.conv1(out))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        # out = self.conv4(out)

        return out


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)
        
        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        
        self.register_buffer('weight', kernel)
        
        # change it to 1 if DO NOT want to be channel-wise.
        self.groups = channels
        
        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )
    
    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)





# def DoG(x):
#
#     Gauss_pyramid = build_gaussian_pyramid(x)
#     DoG_pyramid = build DoG_pyramid()
#
#     return DoG_pyramid

class Latefusion_conv4(nn.Module):
    def __init__(self):
        super(Latefusion_conv4, self).__init__()
        
        self.conv4 = CONV4()
        self.fusion = nn.Conv2d(in_channels=8,out_channels=2,kernel_size=3,stride=1,padding=1)
        
    def forward(self,x, y):
        x4 = F.interpolate(x, size=[60,60], mode='bilinear')
        y4 = F.interpolate(y, size=[60,60], mode='bilinear')
        x3 = F.interpolate(x, size=[80,80], mode='bilinear')
        y3 = F.interpolate(y, size=[80,80], mode='bilinear')
        x2 = F.interpolate(x, size=[120,120], mode='bilinear')
        y2 = F.interpolate(y, size=[120,120], mode='bilinear')
        x1 = x
        y1 = y
        
        flow4 = self.conv4(x4,y4)
        flow3 = self.conv4(x3,y3)
        flow2 = self.conv4(x2,y2)
        flow1 = self.conv4(x1,y1)

        flow4 = F.interpolate(flow4, size=[240,240], mode='bilinear')
        flow3 = F.interpolate(flow3, size=[240,240], mode='bilinear')
        flow2 = F.interpolate(flow2, size=[240,240], mode='bilinear')

        flow = torch.cat([flow1,flow2,flow3,flow4],1)
        
        flow = self.fusion(flow)
        
        return flow

class Hierarchinal_conv4(nn.Module):
    def __init__(self):
        super(Hierarchinal_conv4, self).__init__()
    
        self.conv4 = CONV4()
        self.fusion = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.backward_warp=Backward_warp()

    def forward(self, x, y):
        x4 = F.interpolate(x, size=[60, 60], mode='bilinear')
        y4 = F.interpolate(y, size=[60, 60], mode='bilinear')
        x3 = F.interpolate(x, size=[80, 80], mode='bilinear')
        y3 = F.interpolate(y, size=[80, 80], mode='bilinear')
        x2 = F.interpolate(x, size=[120, 120], mode='bilinear')
        y2 = F.interpolate(y, size=[120, 120], mode='bilinear')
        x1 = x
        y1 = y
    
        flow4 = self.conv4(x4, y4)
        flow4 = F.interpolate(flow4, size=[80, 80], mode='bilinear')
        warped4 = self.backward_warp(x3,flow4)
        
        flow3 = self.conv4(warped4, y3)
        flow3 = F.interpolate(flow3, size=[120, 120], mode='bilinear')
        warped3 = self.backward_warp(x2,flow3)
        
        flow2 = self.conv4(warped3, y2)
        flow2 = F.interpolate(flow2, size=[240, 240], mode='bilinear')
        # warped2 = self.backward_warp(x1,flow2)
        
    
        return flow2


class Hierarchinal_gaussian_conv4(nn.Module):
    def __init__(self):
        super(Hierarchinal_gaussian_conv4, self).__init__()
        
        self.conv4 = CONV4()
        self.fusion = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.backward_warp = Backward_warp()
        self.smoothing1 = GaussianSmoothing(3, 5, 1.26)
        self.smoothing2 = GaussianSmoothing(3, 5, 1.59)
        self.smoothing3 = GaussianSmoothing(3, 5, 2)
        
    def forward(self, x, y):
        x4 = F.interpolate(x, size=[60, 60], mode='bilinear')
        x4_sm1 = self.smoothing1(x4)
        x4_sm2 = self.smoothing2(x4)
        x4_sm3 = self.smoothing3(x4)
        x4_lap1 = x4_sm1-x4_sm2
        x4_lap2 = x4_sm2-x4_sm3
        x4 = torch.cat([x4_lap1,x4_lap2],1)
        
        y4 = F.interpolate(y, size=[60, 60], mode='bilinear')
        y4_sm1 = self.smoothing1(y4)
        y4_sm2 = self.smoothing2(y4)
        y4_sm3 = self.smoothing3(y4)
        y4_lap1 = y4_sm1-y4_sm2
        y4_lap2 = y4_sm2-y4_sm3
        y4 = torch.cat([y4_lap1,y4_lap2],1)
        
        x3 = F.interpolate(x, size=[80, 80], mode='bilinear')
        x3_sm1 = self.smoothing1(x3)
        x3_sm2 = self.smoothing2(x3)
        x3_sm3 = self.smoothing3(x3)
        x3_lap1 = x3_sm1-x3_sm2
        x3_lap2 = x3_sm2-x3_sm3
        x3 = torch.cat([x3_lap1,x3_lap2],1)
        
        y3 = F.interpolate(y, size=[80, 80], mode='bilinear')
        y3_sm1 = self.smoothing1(y3)
        y3_sm2 = self.smoothing2(y3)
        y3_sm3 = self.smoothing3(y3)
        y3_lap1 = y3_sm1-y3_sm2
        y3_lap2 = y3_sm2-y3_sm3
        y3 = torch.cat([y3_lap1,y3_lap2],1)
        
        x2 = F.interpolate(x, size=[120, 120], mode='bilinear')
        x2_sm1 = self.smoothing1(x2)
        x2_sm2 = self.smoothing2(x2)
        x2_sm3 = self.smoothing3(x2)
        x2_lap1 = x2_sm1-x2_sm2
        x2_lap2 = x2_sm2-x2_sm3
        x2 = torch.cat([x2_lap1,x2_lap2],1)
        
        y2 = F.interpolate(y, size=[120, 120], mode='bilinear')
        y2_sm1 = self.smoothing1(y2)
        y2_sm2 = self.smoothing2(y2)
        y2_sm3 = self.smoothing3(y2)
        y2_lap1 = y2_sm1-y2_sm2
        y2_lap2 = y2_sm2-y2_sm3
        y2 = torch.cat([y2_lap1,y2_lap2],1)
        
        flow4 = self.conv4(x4, y4)
        flow4 = F.interpolate(flow4, size=[80, 80], mode='bilinear')
        print(flow4.shape)
        warped4 = self.backward_warp(x3, flow4)
        
        flow3 = self.conv4(warped4, y3)
        flow3 = F.interpolate(flow3, size=[120, 120], mode='bilinear')
        warped3 = self.backward_warp(x2, flow3)
        
        flow2 = self.conv4(warped3, y2)
        flow2 = F.interpolate(flow2, size=[240, 240], mode='bilinear')
        # warped2 = self.backward_warp(x1,flow2)
        
        
        return flow2
if __name__ == '__main__':
    
    
    input = torch.rand(16, 3, 240, 240)
    input = input.cuda()
    # input = torch.rand(6, 3, 4032, 3024)
    model = Hierarchinal_gaussian_conv4()

    model = model.cuda()
    output = model(input,input)
    print(input.size())
    print(output.size())




