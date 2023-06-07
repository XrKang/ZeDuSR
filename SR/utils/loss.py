from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.vgg import vgg16
# from utils.losses.NCC import NCC_pytorch
# from utils.losses.CX_loss import CXLoss
import torch.nn.functional as F
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse
import getpass
# user_name = getpass.getuser() # 获取当前用户名

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ReconstructionLoss(nn.Module):
    def __init__(self, args):
        super(ReconstructionLoss, self,).__init__()
        # if torch.cuda.is_available():
        #     vgg = vgg16(pretrained=False)
        #     # vgg.load_state_dict(torch.load('/data/vgg/vgg16-397923af.pth'))
        #     vgg.load_state_dict(torch.load(args.vgg_path))
        # else:
        #     vgg = vgg16(pretrained=True)
        # self.vgg_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        # for param in self.vgg_network.parameters():
        #     param.requires_grad = False
            
        self.L2_loss = nn.MSELoss()
        # self.vgg_weight = args.vgg_weight

    def L1_Charbonnier_loss(self, X, Y):
        self.eps = 1e-3
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

    def forward(self, X, Y):
        mseloss = self.L1_Charbonnier_loss(X, Y)
        return mseloss
        # perception_loss = self.L2_loss(self.vgg_network(X), self.vgg_network(Y))
        # loss = mseloss+self.vgg_weight*perception_loss
        # return loss

class TripleLoss(nn.Module):
    def __init__(self):
        super(TripleLoss, self).__init__()
        self.L1 = nn.L1Loss()
    def forward(self, LR, Ref, pre):
        ref_fft = FFT_tensor(Ref)
        pre_fft = FFT_tensor(pre)

        mag_ref_fft = torch.abs(ref_fft)
        mag_ref_fft = torch.log(mag_ref_fft)

        mag_pre_fft = torch.abs(pre_fft)
        mag_pre_fft = torch.log(mag_pre_fft)

        # pha_ref_fft = torch.abs(torch.angle(ref_fft))
        # pha_pre_fft = torch.abs(torch.angle(pre_fft))
        # loss = self.L1(mag_ref_fft, mag_pre_fft) + self.L1(pha_ref_fft, pha_pre_fft)
        loss = self.L1(mag_ref_fft, mag_pre_fft)
        return loss

def FFT_tensor(img_tensor):
    _, _, h, w = img_tensor.shape
    img_FFT = torch.fft.fft2(img_tensor, dim=(2, 3))
    img_FFT_shift = torch.roll(img_FFT, (h // 2, w // 2), dims=(2, 3))
    return img_FFT_shift
    # Amg_FFT_shift = torch.abs(img_FFT_shift)
    # Amg_FFT_shift = torch.log(Amg_FFT_shift)
    # return Amg_FFT_shift

if __name__ == "__main__":
    loss = ReconstructionLoss()
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 3,7, 7).cuda()
    print(a)

    print(b)
    print(loss(a, b).item())




