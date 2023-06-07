from __future__ import print_function
import torch.optim as optim

import os
from PIL import Image
import torch.nn.parallel
import torch.utils.data
import argparse
import numpy as np
from tqdm import tqdm

from dataloaders.dataloader import load_data

from models.model import MODEL
from utils.loss import ReconstructionLoss
from utils.saver import Saver
from utils.metrics import Evaluator
from utils.util import *

current_path = os.path.dirname(__file__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='ZSSR')

# Data specifications
parser.add_argument('--train_lr', type=str,
                    default='../SynthesizedData/Data/DIAlign/LF_isoJPEG2x/LF_bedroom/out_1000_warp.png')
parser.add_argument('--train_hr', type=str,
                    default='../SynthesizedData/Data/DIAlign/LF_isoJPEG2x/LF_bedroom/HR.png')
parser.add_argument('--Invari_map', type=str,
                    default='../SynthesizedData/Data/DIAlign/LF_isoJPEG2x/LF_bedroom/PatchDisOut.npy')

parser.add_argument('--test_lr', type=str,
                    default='../SynthesizedData/Data/WideView_iso2x_JPEG75/LF_bedroom.jpg')
parser.add_argument('--test_hr', type=str,
                    default='../SynthesizedData/Data/WideView_GT/LF_bedroom.png')


# evaluation option
parser.add_argument('--eval_interval', type=int, default=10, help='evaluation interval')
parser.add_argument('--dataset', type=str, default='LF_isoJPEG2x/LF_bedroom', help='save DirName')
parser.add_argument('--output_path', type=str, default='./Results_Synthesized/', help='save RootName')


# training hyper params
parser.add_argument('--epochs', type=int, default=3001, metavar='N', help='number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--patch_size', type=int, default=32)
parser.add_argument('--scale', type=int, default=2)
parser.add_argument('--shave', type=int, default=0)

# optimizer params
parser.add_argument('--lr', type=float, default=2 * 1e-4, metavar='LR', help='learning rate')
parser.add_argument('--lr_scheduler', type=str, default='step',choices=['poly', 'step', 'cos'],
                    help='lr scheduler mode: (default: step)')

parser.add_argument("--milestones", type=list, default=[5, 500, 1000, 2000], help="how many epoch to reduce the lr")
parser.add_argument("--gamma", type=int, default=0.5, help="how much to reduce the lr each time")

parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='M', help='w-decay (default: 5e-4)')
parser.add_argument('--workers', type=int, default=8, metavar='N', help='dataloader threads')

# checking point
parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.writer = self.saver.create_summary()

        # Define Dataloader
        self.train_loader = load_data(args)

        # Define network
        self.model = MODEL(args)

        train_params = [{'params': self.model.parameters(), 'lr': args.lr}]

        # Define Optimizer
        self.optimizer = torch.optim.Adam(train_params, lr=args.lr)

        # Define Criterion
        self.criterion = ReconstructionLoss(args)

        # Using cuda
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)

        # Define Evaluator
        self.evaluator = Evaluator()

        # Define lr scheduler
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, args.milestones, args.gamma)

        # Resuming checkpoint
        if (args.resume is not None) and (args.resume != ''):
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            self.model.load_state_dict(checkpoint)
            self.saver.print_log("=> loaded checkpoint '{}'".format(args.resume))

        # Clear start epoch if fine-tuning
        self.saver.print_log(args)
        self.saver.print_log('Starting Epoch: {}'.format(args.start_epoch))
        self.saver.print_log('Total Epoches: {}'.format(args.epochs))

        self.best = 0.0



    def training(self, epoch):
        self.model.train()
        self.lr_scheduler.step()

        input_img, hr_image = self.train_loader.get_data(epoch)
        input_img, hr_image = input_img.to(device), hr_image.to(device)
        pred = self.model(input_img)

        loss = self.criterion(pred, hr_image)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        pred = torch.clamp(pred, 0.0, 1.0)
        psnr_iter_pred = cal_psnr(pred, hr_image)
        train_loss = loss.item()

        # self.saver.print_log('\n iter: {}/{}, lr: {:.9f} | PSNR:{:.4} | loss: {:.5f} | loss_rec:{:.4}'
        #                      .format(epoch, args.epochs, self.optimizer.param_groups[0]["lr"], psnr_iter_pred,
        #                              train_loss, loss.item()))
        self.writer.add_scalar('train/train_loss', train_loss, epoch)
        self.writer.add_scalar('train/PSNR', psnr_iter_pred, epoch)
        self.writer.add_scalar('loss/loss_rec', loss.item(), epoch)

    def testing(self, epoch, input_lr, input_hr):
        self.model.eval()
        input_lr = np.array(Image.open(input_lr))
        input_hr = np.array(Image.open(input_hr))

        if epoch == 1:
            save_image(args, input_lr, 0, 'LR')
            save_image(args, input_hr, 0, 'HR')

        with torch.no_grad():
            output = selfEnsemble_rot(input_lr, self.model, device)

        psnr, ssim = compare_metric(output, input_hr)
        model_save_path = os.path.join(args.output_path, args.dataset, 'model_pth')
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if psnr >= self.best:
            torch.save(self.model.state_dict(),
                       os.path.join(model_save_path, 'model_epoch_{:04d}_val_PSNR{:.2f}.pth'.format(epoch, psnr)))
        if epoch >= 2:
            self.best = max(psnr, self.best)
        self.saver.print_log('\n =========> Test-Epoch:{}| PSNR:{:.4} | SSIM:{:.4} | Best:{:.4}'
                             .format(epoch, psnr, ssim, self.best))
        save_image(args, output, epoch, 'SR')


def main():
    trainer = Trainer(args)

    trainer.testing(0, args.test_lr, args.test_hr)
    for epoch in tqdm(range(trainer.args.start_epoch, trainer.args.epochs), ncols=45):
        if epoch % args.eval_interval == 0 or epoch == 1 or epoch == 0:
            trainer.testing(epoch, args.test_lr, args.test_hr)
        trainer.training(epoch)

    trainer.writer.close()


if __name__ == '__main__':
    args = parser.parse_args()

    # RCAN setting
    args.in_ch = 3
    args.n_feats = 64
    args.n_reduction = 16  # reduction factor of Channel attention，Channel/reduction
    args.n_resgroups = 5  # RCAB number of  RAM
    args.n_resblocks = 3

    main()
