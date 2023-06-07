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
                    default='/data/ruikang/warp/iphone11/wide2x/IMG_8/out_1500_warp.png')

parser.add_argument('--train_hr', type=str,
                    default='/data/ruikang/warp/iphone11/wide2x/IMG_8/HR.png')

parser.add_argument('--Invari_map', type=str,
                    default='/data/ruikang/warp/iphone11/wide2x/IMG_8/PatchDisOut.npy')

parser.add_argument('--test_lr', type=str,
                    default='/data/ruikang/iphone11/test/wide/IMG_8.jpg')
parser.add_argument('--test_hr', type=str,
                    default='/data/ruikang/iphone11/test/tele/IMG_8.jpg')

parser.add_argument('--workers', type=int, default=8, metavar='N', help='dataloader threads')
parser.add_argument('--vgg_weight', type=float, default=0.05, help='weight of perception loss')
parser.add_argument('--vgg_path', type=str, default='/data/ruikang/vgg/vgg16-397923af.pth',
                    help='weight of perception loss')

# evaluation option
parser.add_argument('--dataset', type=str, default='wide2x/IMG_8.jpg')
parser.add_argument('--output_path', type=str, default='./Results_iphone11/',
                    help='save dir')
parser.add_argument('--eval_interval', type=int, default=200, help='evaluation interval')

# training hyper params

parser.add_argument('--epochs', type=int, default=2001, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=0,
                    metavar='N', help='start epochs (default:0)')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--patch_size', type=int, default=128)
parser.add_argument('--scale', type=int, default=2)
parser.add_argument('--shave', type=int, default=0)

# optimizer params
parser.add_argument('--lr', type=float, default=2 * 1e-4, metavar='LR', help='learning rate')
parser.add_argument('--lr_scheduler', type=str, default='step',choices=['poly', 'step', 'cos'],
                    help='lr scheduler mode: (default: step)')

parser.add_argument("--milestones", type=list, default=[5, 500, 1000, 1500], help="how many epoch to reduce the lr")
parser.add_argument("--gamma", type=int, default=0.5, help="how much to reduce the lr each time")

parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='M', help='w-decay (default: 5e-4)')

# checking point
parser.add_argument('--resume', type=str,
                    default=None,
                    help='put the path to resuming file if needed')


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
        # self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)

        # Define Evaluator
        self.evaluator = Evaluator()

        # Define lr scheduler
        # self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, lr_step=1000)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, args.milestones, args.gamma)

        # Resuming checkpoint
        if args.resume is not None and args.resume != '':
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            self.model.load_state_dict(checkpoint)
            self.saver.print_log("=> loaded checkpoint '{}'".format(args.resume))

        # Clear start epoch if fine-tuning
        self.saver.print_log(args)
        self.saver.print_log('Starting Epoch: {}'.format(args.start_epoch))
        self.saver.print_log('Total Epoches: {}'.format(args.epochs))

        self.best = 0.

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

        if epoch % 50 == 0:
            self.saver.print_log('\n iter: {}/{}, lr: {:.9f} | PSNR:{:.4} | loss: {:.5f} | loss_rec:{:.4}'
                                .format(epoch, args.epochs, self.optimizer.param_groups[0]["lr"], psnr_iter_pred,
                                        train_loss, loss.item()))
            self.writer.add_scalar('train/train_loss', train_loss, epoch)
            self.writer.add_scalar('train/PSNR', psnr_iter_pred, epoch)
            self.writer.add_scalar('loss/loss_rec', loss.item(), epoch)

    def testing(self, epoch, input_lr, input_hr):
        self.model.eval()
        input_lr = np.array(Image.open(input_lr))
        input_hr = np.array(Image.open(input_hr))
        input_lrBic = cv2.resize(input_lr, (input_lr.shape[1] * 2, input_lr.shape[0] * 2),
                                 interpolation=cv2.INTER_CUBIC)

        if epoch == 1:
            save_image(args, input_lr, 0, 'LR')
            save_image(args, input_hr, 0, 'HR')
            save_image(args, input_lrBic, 0, 'LRBic')

        # Limited Computing source
        with torch.no_grad():
            input_lr_tensor = ToTensor(input_lr)
            input_lr_tensor = input_lr_tensor.to(device)
            B, C, H, W = input_lr_tensor.shape
            output = torch.zeros((B, C, H * args.scale, W * args.scale))
            h_patch, w_patch = H//2, W//2
            for idx_h in range(0, H, h_patch):
                for idx_w in range(0, W, w_patch):
                    input_patch = input_lr_tensor[:, :, idx_h:idx_h+h_patch, idx_w:idx_w+w_patch]
                    output_patch = self.model(input_patch)
                    output[:, :, idx_h* args.scale:idx_h* args.scale+h_patch* args.scale,
                                 idx_w* args.scale:idx_w* args.scale+w_patch* args.scale] = output_patch

        # output = self.model(input_lr_tensor)

        output = torch.clamp(output, 0, 1)
        output = output.squeeze().detach().cpu().numpy()
        output = np.transpose(output * 255.0, (1, 2, 0))


        model_save_path = os.path.join(args.output_path, args.dataset, 'model_pth')
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        torch.save(self.model.state_dict(),
                       os.path.join(model_save_path, 'model_epoch_{:04d}.pth'.format(epoch, ))
                       )



        save_image(args, output, epoch, 'SR')


def main():
    trainer = Trainer(args)

    trainer.testing(0, args.test_lr, args.test_hr)
    for epoch in tqdm(range(trainer.args.start_epoch, trainer.args.epochs), ncols=45):
        # for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
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
