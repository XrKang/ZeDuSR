import torch
import torch.nn as nn
from models.FlowNet2S import FlowNet2S
from models.FlowNet2S_our import FlowNetS

from models.model_lib.PatchDiscriminator import Discriminator as PatchDiscriminator_net
from models.model_lib.GlobalDiscriminator import Discriminator as GlobalDiscriminator_net
from models.model_lib.vdsr import VDSR as VDSR
from utils.util import *
import numpy as np

class Warp(nn.Module):
    def __init__(self, path='', pretrain=False):
        super(Warp, self).__init__()
        if pretrain:
            self.warp_model = FlowNet2S(path)
        else:
            self.warp_model = FlowNetS()

    def forward(self, x, y):
        if self.training:
            out = self.warp_model(x, y)
            return out
        else:
            out = self.warp_model(x, y)
            return out



class Moco(nn.Module):
    def __init__(self, num_neg=10, patch_size=64):
        super(Moco, self).__init__()

        self.encoder_lr = Encoder().to(device)
        self.encoder_pre = Encoder().to(device)
        self.encoder_hr = Encoder().to(device)

        self.num_neg = num_neg
        self.patch_size = patch_size

    def generate_random_position(self, numb):
        start_h = np.random.randint(0, self.H - self.patch_size, numb)
        start_w = np.random.randint(0, self.W - self.patch_size, numb)
        position_stack = np.stack([start_h, start_w], axis=-1)

        return position_stack

    def forward(self, x, y, out):
        BS, _, self.H, self.W = x.shape

        # moco
        lr = x[:, :, 16:self.H - 16, 16:self.W - 16]
        hr = y[:, :, 16:self.H - 16, 16:self.W - 16]
        pre = out[:, :, 16:self.H - 16, 16:self.W - 16]

        # select patch
        BS, _, self.H, self.W = lr.shape
        lr_position = self.generate_random_position(1)[0]
        hr_positions = self.generate_random_position(self.num_neg-1)

        lr_patch = lr[:, :, lr_position[0]:lr_position[0] + self.patch_size,
                   lr_position[1]:lr_position[1] + self.patch_size]

        pre_patch = pre[:, :, lr_position[0]:lr_position[0] + self.patch_size,
                    lr_position[1]:lr_position[1] + self.patch_size]
        hr_patches = [hr[:, :, hr_position[0]:hr_position[0] + self.patch_size,
                      hr_position[1]:hr_position[1] + self.patch_size]
                      for hr_position in hr_positions]
        hr_patches.append(hr[:, :, lr_position[0]:lr_position[0] + self.patch_size,
                             lr_position[1]:lr_position[1] + self.patch_size])

        embed_lr_patch = self.encoder_lr(lr_patch)
        embed_pre_patch = self.encoder_pre(pre_patch)
        embed_hr_patch = self.encoder_hr(torch.cat(hr_patches, dim=0)).view(BS, self.num_neg, -1)

        return embed_lr_patch, embed_pre_patch, embed_hr_patch


class Warp_moco(nn.Module):
    def __init__(self, path='', pretrain=False, num_neg=10, patch_size=64):
        super(Warp_moco, self).__init__()
        if pretrain:
            self.warp_model = FlowNet2S(path)
        else:
            self.warp_model = FlowNetS()
        self.encoder_lr = Encoder().to(device)
        self.encoder_pre = Encoder().to(device)
        self.encoder_hr = Encoder().to(device)

        self.num_neg = num_neg
        self.patch_size = patch_size

    def generate_random_position(self, numb):
        start_h = np.random.randint(0, self.H - self.patch_size, numb)
        start_w = np.random.randint(0, self.W - self.patch_size, numb)
        position_stack = np.stack([start_h, start_w], axis=-1)

        return position_stack

    def forward(self, x, y):
        if self.training:
            out = self.warp_model(x, y)

            BS, _, self.H, self.W = x.shape

            # moco
            lr = x[:, :, 16:self.H - 16, 16:self.W - 16]
            hr = y[:, :, 16:self.H - 16, 16:self.W - 16]
            pre = out[:, :, 16:self.H - 16, 16:self.W - 16]

            # select patch
            BS, _, self.H, self.W = lr.shape
            lr_position = self.generate_random_position(1)[0]
            hr_positions = self.generate_random_position(self.num_neg - 1)

            lr_patch = lr[:, :, lr_position[0]:lr_position[0] + self.patch_size,
                       lr_position[1]:lr_position[1] + self.patch_size]

            pre_patch = pre[:, :, lr_position[0]:lr_position[0] + self.patch_size,
                        lr_position[1]:lr_position[1] + self.patch_size]

            hr_patches = [hr[:, :, hr_position[0]:hr_position[0] + self.patch_size,
                          hr_position[1]:hr_position[1] + self.patch_size]
                          for hr_position in hr_positions]
            hr_patches.append(hr[:, :, lr_position[0]:lr_position[0] + self.patch_size,
                              lr_position[1]:lr_position[1] + self.patch_size])

            embed_lr_patch = self.encoder_lr(lr_patch)
            embed_pre_patch = self.encoder_pre(pre_patch)
            embed_hr_patch = self.encoder_hr(torch.cat(hr_patches, dim=0)).view(BS, self.num_neg, -1)

            # lr_patch_fft = FFT_tensor(lr_patch)
            # pre_patch_fft = FFT_tensor(pre_patch)
            # hr_patches_fft = [FFT_tensor(hr_patch) for hr_patch in hr_patches]

            # embed_lr_patch_fft = self.encoder_lr(lr_patch_fft)
            # embed_pre_patch_fft = self.encoder_pre(pre_patch_fft)
            # embed_hr_patch_fft = self.encoder_hr(torch.cat(hr_patches_fft, dim=0)).view(BS, self.num_neg, -1)

            # return out, embed_lr_patch_fft, embed_pre_patch_fft, embed_hr_patch_fft
            return out, embed_lr_patch, embed_pre_patch, embed_hr_patch
        else:
            out = self.warp_model(x, y)
            return out


def FFT_tensor(img_tensor):
    _, _, h, w = img_tensor.shape
    img_FFT = torch.fft.fft2(img_tensor, dim=(2, 3))
    img_FFT_shift = torch.roll(img_FFT, (h // 2, w // 2), dims=(2, 3))
    Amg_FFT_shift = torch.abs(img_FFT_shift)
    Amg_FFT_shift = torch.log(Amg_FFT_shift)
    return Amg_FFT_shift


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return out

class Degradation(nn.Module):
    def __init__(self):
        super(Degradation, self).__init__()
        self.degradation = VDSR()
        self.scale_factor = [2,2]
    def forward(self, x):
        out = self.degradation(x)
        out = out[:,:,np.round(np.linspace(0, out.shape[2] - self.scale_factor[0], out.shape[2] // 2)).astype(int)[:,None],
              np.round(np.linspace(0, out.shape[3] - self.scale_factor[1], out.shape[3] // 2)).astype(int)]
        return out

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.discriminator = PatchDiscriminator_net()
    def forward(self, x):
        out = self.discriminator(x)
        return out

class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super(GlobalDiscriminator, self).__init__()
        self.discriminator = GlobalDiscriminator_net()
    def forward(self, x):
        out = self.discriminator(x)
        return out

if __name__ =='__main__':
    input = torch.rand(6, 3, 500, 500)
    input = input.cuda()
    model = Warp_Model()
    # for m in model.modules():
    #     print(m)
    vdsr_params = model.vdsr.parameters()
    # vdsr_params = filter(lambda p: id(p)  in vdsr_params, model.parameters())

    print(vdsr_params)
    # print(conv_params)
    # model = model.cuda()
    # output = model(input,input)
    print(input.size())
    # print(output.size())



