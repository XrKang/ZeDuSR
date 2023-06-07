import os
import numpy as np
from torch.utils import data
from torchvision import transforms
from PIL import Image
import torch
import glob
import cv2
from utils.util import *
import random


class H5Dataset():
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, input, label, args, mode='Train'):
        self.input = input
        self.label = label
        self.mode = mode
        self.patch_size = args.patch_size
        self.batch_size = args.batch_size
        self.scale = args.scale
        self.Invari_map = np.load(args.Invari_map)
        self.max_iter = args.epochs
        self.shave = args.shave


        # Read data
        self.inputimg = np.asarray(Image.open(self.input))
        self.inputlabel = np.asarray(Image.open(self.label))
        self.h, self.w, _ = self.inputimg.shape
        print('shape:', self.inputimg.shape, self.inputlabel.shape)

        if self.shave != 0:
            self.inputimg = self.shave_edges(self.inputimg, self.shave)
            self.inputlabel = self.shave_edges(self.inputlabel, self.shave*args.scale)
        self.h_shaveEdge, self.w_shaveEdge, _ = self.inputimg.shape
        print('shape:', self.inputimg.shape, self.inputlabel.shape)


    def get_data(self, iteration):
        self.aug_rand1 = np.random.randint(0, 2, self.batch_size)
        self.aug_rand2 = np.random.randint(0, 2, self.batch_size)
        self.aug_rand3 = np.random.randint(0, 2, self.batch_size)

        if iteration < 100:
            self.position_stack = self.generate_random_position()
        else:
            if np.random.randint(0, 2):
                self.patch_size = self.patch_size
                # It can be replaced by the receptive field of D_spa
                self.position_stack = self.generate_InVarDomain_position()
            else:
                self.position_stack = self.generate_random_position()

        self.lr_patch, self.hr_patch = self.crop_slice()

        sample = {'image': self.lr_patch, 'label': self.hr_patch}

        sample = self.transform_tr(sample)

        lr_patch = sample["image"]
        hr_patch = sample["label"]

        return lr_patch, hr_patch

    def data_aug(self, image, i):

        if self.aug_rand1[i]:
            image = np.fliplr(image)
        if self.aug_rand2[i]:
            image = np.flipud(image)
        if self.aug_rand3[i]:
            image = np.rot90(image)
        return image
    def crop_slice(self):
        lr_slice_stack = np.stack([self.data_aug(self.inputimg[position[0]:position[0] + self.patch_size,
                                                 position[1]:position[1] + self.patch_size, :], i)
                                   for i, position in enumerate(self.position_stack)])

        hr_slice_stack = np.stack([self.data_aug(
            self.inputlabel[position[0] * self.scale:position[0] * self.scale + self.patch_size * self.scale,
            position[1] * self.scale:position[1] * self.scale + self.patch_size * self.scale, :], i)
            for i, position in enumerate(self.position_stack)])

        return lr_slice_stack, hr_slice_stack

    def generate_random_position(self):
        start_h = np.random.randint(0, self.h_shaveEdge - self.patch_size, self.batch_size)
        start_w = np.random.randint(0, self.w_shaveEdge - self.patch_size, self.batch_size)
        position_stack = np.stack([start_h, start_w], axis=-1)
        return position_stack

    def generate_InVarDomain_position(self):
        self.Invari_map = self.shave_edges(self.Invari_map, self.shave)
        pro_mab = cv2.resize(self.Invari_map, (self.w_shaveEdge, self.h_shaveEdge), interpolation=cv2.INTER_LINEAR)
        pro_mab = pro_mab.flatten() / pro_mab.sum()
        crop_indices = np.random.choice(a=len(pro_mab), size=self.max_iter, p=pro_mab)

        start_h = []
        start_w = []
        for bs in range(self.batch_size):
            center = crop_indices[bs]
            row, col = int(center / self.h_shaveEdge), center % self.w_shaveEdge
            top, left = min(max(0, row - self.patch_size // 2), self.h_shaveEdge - self.patch_size), \
                        min(max(0, col - self.patch_size // 2), self.w_shaveEdge - self.patch_size)
            start_h.append(top)
            start_w.append(left)
        start_h = np.array(start_h)
        start_w = np.array(start_w)
        position_stack = np.stack([start_h, start_w], axis=-1)
        return position_stack

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            ToTensor()
        ])
        return composed_transforms(sample)

    def shave_edges(self, image, pixels):
        """Shave pixels from edges to avoid code-bugs"""
        # Crop pixels to avoid boundaries effects in synthetically generated examples
        if len(image.shape) == 3:
            image = image[pixels:-pixels, pixels:-pixels, :]
        if len(image.shape) == 1:
            image = image[pixels:-pixels, pixels:-pixels]

        return image


def load_data(args):
    train_path_hr = args.train_hr
    train_path_lr = args.train_lr

    train_set = H5Dataset(train_path_lr, train_path_hr, args, mode='Train')

    return train_set


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = sample['image']
        label = sample['label']

        # image = np.expand_dims(image,1)
        # label = np.expand_dims(label,1)

        image = np.array(image).astype(np.float32).transpose((0, 3, 1, 2))  # whc-->chw
        label = np.array(label).astype(np.float32).transpose((0, 3, 1, 2))  # whc-->chw
        # image = np.expand_dims(image,0)
        # label = np.expand_dims(label,0)

        image /= 255.0
        label /= 255.0
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return {'image': image, 'label': label}


def valid_load(args):
    lr = np.array(Image.open(args.valid_path_lr))
    lr = np.expand_dims(lr, 0)
    lr = np.expand_dims(lr, 0)

    lr = np.array(lr).astype(np.float32).transpose((0, 1, 3, 2))  # whc-->chw

    lr /= 255.0
    image = torch.from_numpy(lr).float()
    return image


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_lr', type=str,
                        default=r'D:\!Never Give up\zeroLens\bicDown_x4\LF_bedroom.png')
    parser.add_argument('--train_hr', type=str,
                        default=r'D:\!Never Give up\zeroLens\GT\LF_bedroom.png')

    parser.add_argument('--Invari_map', type=str, default=r'D:\!Never Give up\zeroLens\camsr2\SR_vgg_LR_our\PatchDisOut.npy')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=4001, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--scale', type=int, default=2)

    args = parser.parse_args()

    train_loader = load_data(args)
    # lr, label = train_loader.get_data(500)
    # print(lr.shape, label.shape)
    lr, label = train_loader.get_data(1)
    print(lr.shape, label.shape)