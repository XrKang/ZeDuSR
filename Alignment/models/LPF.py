import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# class LPF_Filter(nn.Module):
#     def __init__(self, k_size=5):
#         super(LPF_Filter, self).__init__()
#         self.inConv = nn.Conv2d(3, 3, kernel_size=k_size, stride=1, padding=(k_size-1)//2, bias=False)
#         # self.outConv = nn.Conv2d(8, 3, kernel_size=k_size, stride=1, padding=(k_size-1)//2, bias=False)
#
#         # self.weightConv = nn.Conv2d(3*3, 3, kernel_size=1, stride=1, padding=0, bias=False)
#
#     def forward(self, x, x_blur, x_gauss):
#         x_nn = self.inConv(x)
#         # x_nn = self.outConv(x_nn)
#
#         # out = self.weightConv((x_nn+x_blur+x_gauss))
#         # out = self.weightConv(torch.cat((x, x_blur, x_gauss), dim=1))
#         out = (x_nn+x_blur+x_gauss)/3
#         # out = (x_nn+x_gauss)
#         return out


def swap_axis(im):
    """Swap axis of a tensor from a 3 channel tensor to a batch of 3-single channel and vise-versa"""
    return im.transpose(0, 1) if type(im) == torch.Tensor else np.moveaxis(im, 0, 1)


def cal_RF(ksize_list):
    ksize = ksize_list
    N_RF = 1
    for i in range(len(ksize)):
        N_RF = (N_RF - 1) + ksize[i]
    return N_RF


class LPF_Filter(nn.Module):
    def __init__(self, ksize_list):
        super(LPF_Filter, self).__init__()
        body = []
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=ksize_list[0], stride=1,
                                     padding=(ksize_list[0] - 1) // 2 if ksize_list[0] > 1 else 0, bias=False)
        for idx in range(1, len(ksize_list) - 1):
            body.append(nn.Conv2d(64, 64, kernel_size=ksize_list[idx], stride=1,
                                  padding=(ksize_list[idx] - 1) // 2 if ksize_list[idx] > 1 else 0,
                                  bias=False))

        self.model = nn.Sequential(*body)
        self.final_layer = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=ksize_list[len(ksize_list) - 1],
                                     stride=1, bias=False,
                                     padding=(ksize_list[len(ksize_list) - 1] - 1) // 2 if ksize_list[len(
                                         ksize_list) - 1] > 1 else 0)
        NF = cal_RF(ksize_list)

        self.curr_k = torch.FloatTensor(NF, NF).to(device)

        # Calculate number of pixels shaved in the forward pass
        # self.output_size = self.forward(torch.FloatTensor(torch.ones([1, 1, conf.input_crop_size, conf.input_crop_size]))).shape[-1]
        # self.forward_shave = int(conf.input_crop_size * conf.scale_factor) - self.output_size

    def forward(self, input_tensor):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        input_tensor = swap_axis(input_tensor)
        features = self.first_layer(input_tensor)
        features = self.model(features)
        output = self.final_layer(features)
        return swap_axis(output)

    # def calc_curr_k(self):
    #     """given a generator network, the function calculates the kernel it is imitating"""
    #     delta = torch.Tensor([1.]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
    #     for ind, w in enumerate(self.G.parameters()):
    #         curr_k = F.conv2d(delta, w, padding=self.conf.G_kernel_size - 1) if ind == 0 else F.conv2d(curr_k, w)
    #     self.curr_k = curr_k.squeeze().flip([0, 1])
    #     return self.curr_k


if __name__ == '__main__':
    input_tensor = torch.rand(1, 3, 384, 384)
    model = LPF_Filter([5, 5, 1, 1])
    output_tensor = model(input_tensor)
    print(input_tensor.shape, output_tensor.shape)
