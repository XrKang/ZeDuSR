from __future__ import print_function
import os
from torch import nn
import torch.nn.parallel
import torch.utils.data
import argparse
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim

from dataloaders.data import ImageDataset

from models.model import Warp, PatchDiscriminator, GlobalDiscriminator

from utils.loss import ReconstructionLoss_warp, NCELoss
from utils.saver import Saver
from utils.util import *
from utils.metrics import Evaluator
from utils.FFT import FFT_tensor

torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='ZeDuSR')

# Data specifications

parser.add_argument('--scale', type=int, default=2)
parser.add_argument('--shave', type=int, default=20,
                    help='shave edge (large number for Middlebury due to large disparity)')
                    
parser.add_argument('--input_hr', type=str,
                    # default='../SynthesizedData/Data/TeleView_crop_SIFTAlign/ladder2.png',
                    default='../RealworldData/Data/TeleView_SIFTAlign_cor/IMG_2936.jpg',
                    help='TeleView image')

parser.add_argument('--input_lr', type=str,
                    # default='../SynthesizedData/Data/WideView_iso2x_JPEG75_crop/ladder2.jpg,
                    default='../RealworldData/Data/WideView_crop/IMG_2936.jpg',
                    help='WideView image')
# training hyper params
parser.add_argument('--lambda_list', type=str, default='1, 0.005, 0.001, 0.01')
# L2_loss adv_spa adv_fre vgglosss

parser.add_argument('--epochs', type=int, default=1501, metavar='N', help='number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epoch')
parser.add_argument('--fre_epoch', type=int, default=800, help='adv_fre loss start epoch')

# optimizer params
parser.add_argument('--lr_list', type=str, default='0.0001, 0.0001')
parser.add_argument('--lr_scheduler', type=str, default='step',
                    choices=['poly', 'step', 'cos'],
                    help='lr scheduler mode: (default: step)')
parser.add_argument("--milestones", type=list, default=[500, 1000], help="how many epoch to reduce the lr")
parser.add_argument("--gamma", type=int, default=0.5, help="how much to reduce the lr each time")

# checking point
parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')

# evaluation option
parser.add_argument('--eval_interval', type=int, default=50, help='evaluation interval (default: 50)')
# parser.add_argument('--output_path', type=str, default='../SynthesizedData/Data/DIAlign/', help='Root path to save ')
parser.add_argument('--output_path', type=str, default='../RealworldData/Data/DIAlign/', help='Root path to save ')
# parser.add_argument('--dataset', type=str, default='MB_isoJPEG2x/ladder2', help='Save DirName')
parser.add_argument('--dataset', type=str, default='iPhone11_wideSRTele/IMG_2936', help='Save DirName')
args = parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        # self.writer = self.saver.create_summary()

        # Define Dataloader
        Dataloader = ImageDataset(args)
        self.lr_img, self.hr_img = Dataloader.totensor()

        # Define network
        # self.model_warp = Warp()
        self.model_warp = Warp()
        self.patch_discrim_warp = PatchDiscriminator()
        self.Gobal_discrim_warp = GlobalDiscriminator()

        # Define Optimizer
        self.lr_warp = float(args.lr_list.split(',')[0])
        self.lr_discrim_patch = float(args.lr_list.split(',')[1])
        self.lr_discrim_warp = float(args.lr_list.split(',')[1])

        self.optimizer_warp = torch.optim.Adam(self.model_warp.parameters(), lr=self.lr_warp)
        self.optimizer_discrim_patch = torch.optim.Adam(self.patch_discrim_warp.parameters(), lr=self.lr_discrim_patch)
        self.optimizer_discrim_warp = torch.optim.Adam(self.Gobal_discrim_warp.parameters(), lr=self.lr_discrim_warp)

        # Define Criterion
        self.criterion_warp = ReconstructionLoss_warp(args)
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_mse = nn.MSELoss()

        # Data CUDA
        self.lr_cuda, self.hr_cuda = self.lr_img.to(device), self.hr_img.to(device)

        # Network CUDA
        self.model_warp = self.model_warp.to(device)

        self.Patch_discrim_warp = self.patch_discrim_warp.to(device)
        self.Gobal_discrim_warp = self.Gobal_discrim_warp.to(device)

        # Criterion CUDA
        self.criterion_warp = self.criterion_warp.to(device)
        self.criterion_gan = self.criterion_gan.to(device)
        self.criterion_mse = self.criterion_mse.to(device)

        # Define Evaluator
        self.evaluator = Evaluator()

        # Define lr scheduler
        self.scheduler_warp = optim.lr_scheduler.MultiStepLR(self.optimizer_warp, args.milestones, args.gamma)
        self.scheduler_discrim_patch = optim.lr_scheduler.MultiStepLR(self.optimizer_discrim_patch, args.milestones,
                                                                      args.gamma)
        self.scheduler_discrim_warp = optim.lr_scheduler.MultiStepLR(self.optimizer_discrim_warp, args.milestones,
                                                                     args.gamma)

        # Resuming checkpoint
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            self.model_warp.load_state_dict(checkpoint['warp'])
            # self.saver.print_log("=> loaded checkpoint '{}' (epoch {})"
            #                      .format(args.resume, checkpoint['epoch']))

        # self.saver.print_log(args)
        # self.saver.print_log('Starting Epoch: {}'.format(args.start_epoch))
        # self.saver.print_log('Total Epoches: {}'.format(args.epochs))

    
    def train_warp_SpaDiscri(self):
        # ---------------------------ESRGAN---------------------------
        for p in self.Patch_discrim_warp.parameters():
            p.requires_grad = True
        self.Patch_discrim_warp.zero_grad()

        D_real_spa = self.Patch_discrim_warp(self.lr_cuda)
        D_fake_spa = self.Patch_discrim_warp(self.warp_out_cuda).detach()

        real_spa = Variable(torch.ones(D_real_spa.shape), requires_grad=False).to(device)
        fake_spa = Variable(torch.zeros(D_fake_spa.shape), requires_grad=False).to(device)

        # # real
        loss_d_real_spa = self.criterion_gan(D_real_spa - D_fake_spa, real_spa) * 0.5
        loss_d_fake_spa = self.criterion_gan(D_fake_spa - D_real_spa.detach(), fake_spa) * 0.5
        self.loss_discrim_spa = loss_d_real_spa + loss_d_fake_spa

        self.loss_discrim_spa.backward()
        self.optimizer_discrim_patch.step()
        self.lr_discrim_patch = self.optimizer_discrim_patch.param_groups[0]["lr"]
    
    # def train_warp_SpaDiscri(self):
    #     # ---------------------------SRGAN---------------------------
    #     for p in self.Patch_discrim_warp.parameters():
    #         p.requires_grad = True
    #     self.Patch_discrim_warp.zero_grad()

    #     D_real_spa = self.Patch_discrim_warp(self.lr_cuda)
    #     D_fake_spa = self.Patch_discrim_warp(self.warp_out_cuda).detach()

    #     real_spa = Variable(torch.ones(D_real_spa.shape), requires_grad=False).to(device)
    #     fake_spa = Variable(torch.zeros(D_fake_spa.shape), requires_grad=False).to(device)

    #     loss_d_real_spa = self.criterion_gan(D_real_spa, real_spa) * 0.5
    #     loss_d_fake_spa = self.criterion_gan(D_fake_spa, fake_spa) * 0.5
    #     self.loss_discrim_spa = loss_d_real_spa + loss_d_fake_spa

    #     self.loss_discrim_spa.backward()
    #     self.optimizer_discrim_patch.step()
    #     self.lr_discrim_patch = self.optimizer_discrim_patch.param_groups[0]["lr"]

    def train_warp_FreDiscri(self):
        for p in self.Gobal_discrim_warp.parameters():
            p.requires_grad = True
        self.Gobal_discrim_warp.zero_grad()

        D_real_fre = self.Gobal_discrim_warp(self.lr_fft)
        D_fake_fre = self.Gobal_discrim_warp(self.warp_out_fft).detach()

        real_fre = Variable(torch.ones(D_real_fre.shape), requires_grad=False).to(device)
        fake_fre = Variable(torch.zeros(D_fake_fre.shape), requires_grad=False).to(device)

        # # real
        loss_d_real_fre = self.criterion_gan(D_real_fre, real_fre) * 0.5
        loss_d_real_fre.backward()

        # # fake
        D_fake_fre = self.Gobal_discrim_warp(self.warp_out_fft.detach())
        loss_d_fake_fre = self.criterion_gan(D_fake_fre, fake_fre) * 0.5
        loss_d_fake_fre.backward()
        self.optimizer_discrim_warp.step()

        self.lr_discrim_warp = self.optimizer_discrim_warp.param_groups[0]["lr"]
        self.loss_discrim_fre = loss_d_fake_fre.item() + loss_d_real_fre.item()

    def train_warp(self):
        self.scheduler_warp.step()
        self.scheduler_discrim_patch.step()
        self.scheduler_discrim_warp.step()

        self.lr_fft = FFT_tensor(self.lr_cuda).type(dtype=self.lr_cuda.dtype)
        # ---------------------------Warp processing---------------------------
        self.warp_out_cuda = self.model_warp(self.lr_cuda, self.hr_cuda)
        self.warp_out_fft = FFT_tensor(self.warp_out_cuda).type(dtype=self.warp_out_cuda.dtype)

        # ---------------------------Discriminator---------------------------
        if self.iteration >= args.fre_epoch:
            self.train_warp_SpaDiscri()
            self.train_warp_FreDiscri()
        else:
            self.train_warp_SpaDiscri()
            self.loss_discrim_fre = 0.0

        # ---------------------------Generator---------------------------
        for p in self.Patch_discrim_warp.parameters():
            p.requires_grad = False

        for p in self.Gobal_discrim_warp.parameters():
            p.requires_grad = False


        # ---------------------------spatial---------------------------
        # ------------ESRGAN-----------
        D_real_spa = self.Patch_discrim_warp(self.lr_cuda).detach()
        D_fake_spa = self.Patch_discrim_warp(self.warp_out_cuda)
        real_spa = Variable(torch.ones(D_real_spa.shape), requires_grad=False).to(device)
        fake_spa = Variable(torch.zeros(D_fake_spa.shape), requires_grad=False).to(device)
        loss_g_real_spa = self.criterion_gan(D_real_spa - D_fake_spa, fake_spa)
        loss_g_fake_spa = self.criterion_gan(D_fake_spa - D_real_spa, real_spa)
        loss_adversarial_spa = (loss_g_real_spa + loss_g_fake_spa) / 2
        self.loss_adversarial_spa = float(args.lambda_list.split(',')[1]) * loss_adversarial_spa
        self.D_fake_spa = D_fake_spa
        # ------------SRGAN-----------
        # D_fake_spa = self.Patch_discrim_warp(self.warp_out_cuda)
        # fake_spa = Variable(torch.zeros(D_fake_spa.shape), requires_grad=False).to(device)
        # self.loss_adversarial_spa = float(args.lambda_list.split(',')[1]) * self.criterion_gan(D_fake_spa, fake_spa)
        # self.D_fake_spa = D_fake_spa

        # ---------------------------Frequence---------------------------
        D_fake_fre = self.Gobal_discrim_warp(self.warp_out_fft)
        fake_fre = Variable(torch.zeros(D_fake_fre.shape), requires_grad=False).to(device)
        loss_adversarial_fre = self.criterion_gan(D_fake_fre, fake_fre)
        self.loss_adversarial_fre = float(args.lambda_list.split(',')[2]) * loss_adversarial_fre

        # ---------------------------L1 vgg loss---------------------------
        self.loss_warp_l1, self.mse_loss, self.cl_loss = self.criterion_warp(self.lr_cuda, self.warp_out_cuda, self.hr_cuda.detach())

        # ---------------------------total loss-------------------------
        if self.iteration >= args.fre_epoch:
            self.loss_warp_total = self.loss_warp_l1 + self.loss_adversarial_spa + self.loss_adversarial_fre
        else:
            self.loss_warp_total = self.loss_warp_l1 + self.loss_adversarial_spa
            self.loss_adversarial_fre = 0.0*self.loss_adversarial_fre


        self.optimizer_warp.zero_grad()
        self.loss_warp_total.backward()
        self.optimizer_warp.step()

        self.warp_lr = self.optimizer_warp.param_groups[0]["lr"]
        self.train_loss_warp_total = self.loss_warp_total.item()

    def training(self, iteration):
        self.model_warp.train()
        self.patch_discrim_warp.train()
        self.Gobal_discrim_warp.train()
        self.iteration = iteration

        # ---------------------------Warp---------------------------
        self.train_warp()

        if self.iteration % args.eval_interval == 0 and self.iteration>200:
            save_image(args, np.transpose(clamp_and_2numpy(self.warp_out_cuda).squeeze(), (1, 2, 0)) * 255,
                       self.iteration, '_warp')
        
        if iteration == args.epochs - 1:
            np.save(os.path.join(args.output_path, args.dataset, "PatchDisOut.npy"),
                    np.array(self.D_fake_spa.squeeze().detach().cpu().numpy()))


def main():
    trainer = Trainer(args)
    for iteration in tqdm(range(args.start_epoch + 1, args.epochs), ncols=45):
        
        trainer.training(iteration)

    # trainer.writer.close()


if __name__ == '__main__':
    main()
