from __future__ import print_function, division
import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16
from utils.util import create_penalty_mask, map2tensor
from torch.autograd import Variable
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from utils.Encoder import Encoder


class ReconstructionLoss_warp(nn.Module):

    def __init__(self, args, size_average=True, batch_average=True):
        super(ReconstructionLoss_warp, self).__init__()

        self.lambda_mse = float(args.lambda_list.split(',')[0])
        self.lambda_vgg = float(args.lambda_list.split(',')[-1])
        self.size_average = size_average
        self.batch_average = batch_average
        self.L2_loss = nn.MSELoss()

        # self.gan_loss = GANLoss()

        # if torch.cuda.is_available():  # and user_name!='mingd':
        #     vgg = vgg16(pretrained=False)
        #     # vgg.load_state_dict(torch.load('/data/ruikang/vgg/vgg16-397923af.pth'))
        # else:
        #     vgg = vgg16(pretrained=True)
        vgg = vgg16(pretrained=True)

        # vgg.load_state_dict(torch.load('/gdata/yaomd/pretrained/vgg/vgg16-397923af.pth'))
        self.loss_network = nn.Sequential(*list(vgg.features)[:10]).eval()
        for param in self.loss_network.parameters():
            param.requires_grad = False

    def L1_Charbonnier_loss(self, X, Y):
        self.eps = 1e-3
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

    def forward(self, X, X_, Y):

        mse_loss = self.L1_Charbonnier_loss(X_, Y)
        vgg_loss = self.L2_loss(self.loss_network(X), self.loss_network(X_)) / self.L2_loss(self.loss_network(Y), self.loss_network(X_))

        loss = self.lambda_mse * mse_loss + self.lambda_vgg * vgg_loss

        return loss, self.lambda_mse * mse_loss, self.lambda_vgg * vgg_loss


def FFT_tensor(img_tensor):
    _, _, h, w = img_tensor.shape
    img_FFT = torch.fft.fft2(img_tensor, dim=(2, 3))
    img_FFT_shift = torch.roll(img_FFT, (h // 2, w // 2), dims=(2, 3))
    Amg_FFT_shift = torch.abs(img_FFT_shift)
    Amg_FFT_shift = torch.log(Amg_FFT_shift)
    return Amg_FFT_shift
    # return img_FFT_shift


class TripleLoss(nn.Module):
    def __init__(self):
        super(TripleLoss, self).__init__()
        self.L1 = nn.L1Loss()
    def forward(self, LR, Ref, pre):
        lr_fft = FFT_tensor(LR).type(dtype=LR.dtype)
        ref_fft = FFT_tensor(Ref).type(dtype=Ref.dtype)
        pre_fft = FFT_tensor(pre).type(dtype=pre.dtype)
        loss = self.L1(pre_fft, lr_fft)/self.L1(pre_fft, ref_fft) + 1*1e-8
        return loss

class ContrastiveLoss_warp(nn.Module):

    def __init__(self, num_neg, patch_size):
        super(ContrastiveLoss_warp, self).__init__()
        self.num_neg = num_neg
        self.patch_size = patch_size
        self.encoder_lr = Encoder().to(device)
        self.encoder_pre = Encoder().to(device)
        self.encoder_hr = Encoder().to(device)
        self.loss_infoNCE = NCELoss(self.num_neg).to(device)

    def generate_random_position(self, numb):
        start_h = np.random.randint(0, self.H - self.patch_size, numb)
        start_w = np.random.randint(0, self.W - self.patch_size, numb)
        position_stack = np.stack([start_h, start_w], axis=-1)

        return position_stack

    def forward(self, lr, pre, hr):
        _, _, self.H, self.W = lr.shape

        # Crop center
        lr = lr[:, :, 16:self.H - 16, 16:self.W - 16]
        hr = hr[:, :, 16:self.H - 16, 16:self.W - 16]
        pre = pre[:, :, 16:self.H - 16, 16:self.W - 16]

        BS, _, self.H, self.W = lr.shape

        # select patch
        lr_position = self.generate_random_position(1)[0]
        hr_positions = self.generate_random_position(self.num_neg)

        lr_patch = lr[:, :, lr_position[0]:lr_position[0] + self.patch_size,
                   lr_position[1]:lr_position[1] + self.patch_size]

        pre_patch = pre[:, :, lr_position[0]:lr_position[0] + self.patch_size,
                    lr_position[1]:lr_position[1] + self.patch_size]
        hr_patches = [hr[:, :, hr_position[0]:hr_position[0] + self.patch_size,
                      hr_position[1]:hr_position[1] + self.patch_size]
                      for hr_position in hr_positions]

        lr_patch_fft = FFT_tensor(lr_patch)
        pre_patch_fft = FFT_tensor(pre_patch)
        hr_patches_fft = [FFT_tensor(hr_patch) for hr_patch in hr_patches]

        embed_lr_patch_fft = self.encoder_lr(lr_patch_fft)
        embed_pre_patch_fft = self.encoder_pre(pre_patch_fft)
        embed_hr_patch_fft = self.encoder_hr(torch.cat(hr_patches_fft, dim=0)).view(BS, self.num_neg, -1)

        loss = self.loss_infoNCE(embed_lr_patch_fft, embed_pre_patch_fft, embed_hr_patch_fft)

        return loss

class NCELoss(nn.Module):
    def __init__(self, num_neg, T=0.07):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool
        self.num_neg = num_neg
        self.T = T
        # self.mask_dtype =  torch.uint8

    def forward(self, feat_pos, feat, feat_neg):
        batch_size = feat_pos.shape[0]
        dim = feat_pos.shape[1]

        # pos logit
        l_pos = torch.bmm(
            feat_pos.view(batch_size, 1, -1), feat.view(batch_size, -1, 1))
        l_pos = l_pos.view(batch_size, 1)  # BS, 1

        # neg logit
        feat = feat.view(batch_size, 1, dim)
        feat_neg = feat_neg.view(batch_size, self.num_neg, dim)
        l_neg = torch.bmm(feat, feat_neg.transpose(2, 1))
        l_neg = l_neg.view(batch_size, self.num_neg)  # BS, num_neg

        out = torch.cat((l_pos, l_neg), dim=1) / self.T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat.device))
        # max l_pos, min l_neg
        return loss


class LPF_Kernelloss(nn.Module):

    def __init__(self, kernel_size):
        super(LPF_Kernelloss, self).__init__()
        self.sum2one_loss = SumOfWeightsLoss()
        self.boundaries_loss = BoundariesLoss(k_size=kernel_size)
        self.centralized_loss = CentralizedLoss(k_size=kernel_size, scale_factor=1)
        self.sparse_loss = SparsityLoss()

        self.lambda_sum2one = 0.5
        self.lambda_boundaries = 0.5
        self.lambda_centralized = 0
        self.lambda_sparse = 0

    def forward(self, kernel):
        loss_boundaries = self.boundaries_loss.forward(kernel=kernel)
        loss_sum2one = self.sum2one_loss.forward(kernel=kernel)
        loss_centralized = self.centralized_loss.forward(kernel=kernel)
        loss_sparse = self.sparse_loss.forward(kernel=kernel)

        loss = loss_sum2one * self.lambda_sum2one + \
               loss_boundaries * self.lambda_boundaries + loss_centralized * self.lambda_centralized + \
               loss_sparse * self.lambda_sparse

        return loss


class SumOfWeightsLoss(nn.Module):
    """ Encourages the kernel G is imitating to sum to 1 """

    def __init__(self):
        super(SumOfWeightsLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(torch.ones(1).to(kernel.device), torch.sum(kernel))


class CentralizedLoss(nn.Module):
    """ Penalizes distance of center of mass from K's center"""

    def __init__(self, k_size, scale_factor=.5):
        super(CentralizedLoss, self).__init__()
        self.indices = Variable(torch.arange(0., float(k_size)).to(device), requires_grad=False)
        wanted_center_of_mass = k_size // 2 + 0.5 * (int(1 / scale_factor) - k_size % 2)
        self.center = Variable(torch.FloatTensor([wanted_center_of_mass, wanted_center_of_mass]).to(device),
                               requires_grad=False)
        self.loss = nn.MSELoss()

    def forward(self, kernel):
        """Return the loss over the distance of center of mass from kernel center """
        r_sum, c_sum = torch.sum(kernel, dim=1).reshape(1, -1), torch.sum(kernel, dim=0).reshape(1, -1)
        return self.loss(torch.stack((torch.matmul(r_sum, self.indices) / torch.sum(kernel),
                                      torch.matmul(c_sum, self.indices) / torch.sum(kernel))), self.center)


class BoundariesLoss(nn.Module):
    """ Encourages sparsity of the boundaries by penalizing non-zeros far from the center """

    def __init__(self, k_size):
        super(BoundariesLoss, self).__init__()
        self.mask = map2tensor(create_penalty_mask(k_size, 30))
        self.zero_label = Variable(torch.zeros(k_size).to(device), requires_grad=False)
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(kernel * self.mask, self.zero_label)


class SparsityLoss(nn.Module):
    """ Penalizes small values to encourage sparsity """

    def __init__(self):
        super(SparsityLoss, self).__init__()
        self.power = 0.2
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(torch.abs(kernel) ** self.power, torch.zeros_like(kernel))


if __name__ == "__main__":
    loss = PatchNCELoss()
    a = torch.rand(5, 256).to(device)
    b = torch.rand(5, 256).to(device)
    print(loss(a, b))

    loss = NCELoss(5)
    a = torch.rand(1, 256).to(device)
    b = torch.rand(1, 256).to(device)
    c = torch.rand(1, 5, 256).to(device)
    print(loss(a, b, c))
