import math
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
# from utils.degradate_img import generate_degradation
import cv2
import os


class ImageDataset():

    def __init__(self, args):
        self.hr = args.input_hr
        self.lr = args.input_lr
        self.scale = args.scale
        self.shave = args.shave

        save_path = os.path.join(args.output_path, args.dataset)

        # Read data
        self.lr_img = np.asarray(Image.open(self.lr))
        self.hr_img = np.asarray(Image.open(self.hr))

        print(self.hr, "   ", self.lr)
        # print(self.lr_img.shape, self.hr_img.shape)

        print(" data 0", self.hr_img.shape, self.lr_img.shape, self.scale)

        if self.shave != 0:
            self.lr_img = self.shave_edges(self.lr_img, self.shave)
            self.hr_img = self.shave_edges(self.hr_img, self.shave * self.scale)

        crop_size_h = self.hr_img.shape[0] // (16 * self.scale) * (16 * self.scale)
        crop_size_w = self.hr_img.shape[1] // (16 * self.scale) * (16 * self.scale)
        self.hr_img = self.hr_img[0:crop_size_h, 0:crop_size_w, :]
        self.h_hr, self.w_hr, _ = self.hr_img.shape
        # print(self.h_hr, self.w_hr)

        hr_save = Image.fromarray(self.hr_img.astype(np.uint8))
        hr_save.save(os.path.join(save_path, 'HR.png'))

        self.lr_img = self.lr_img[0:self.h_hr // self.scale, 0:self.w_hr // self.scale, :]
        self.hr_img = cv2.resize(self.hr_img, (self.w_hr // self.scale, self.h_hr // self.scale),
                                 interpolation=cv2.INTER_LINEAR)

        print(" data 1", self.hr_img.shape, self.lr_img.shape, self.scale)



        # save_ori


        hr_down_save = Image.fromarray(self.hr_img.astype(np.uint8))
        hr_down_save.save(os.path.join(save_path, 'HRDown.png'))
        lr_save = Image.fromarray(self.lr_img.astype(np.uint8))
        lr_save.save(os.path.join(save_path, 'LR.png'))

    def shave_edges(self, image, pixels):
        """Shave pixels from edges to avoid code-bugs"""
        # Crop pixels to avoid boundaries effects in synthetically generated examples
        if len(image.shape) == 3:
            image = image[pixels:-pixels, pixels:-pixels, :]
        if len(image.shape) == 1:
            image = image[pixels:-pixels, pixels:-pixels]

        return image

    def get_data(self):
        return self.lr_img, self.hr_img
    def totensor(self):
        sample = {'lr': self.lr_img, 'hr': self.hr_img, }

        # to tensor
        sample = self.transform_tr(sample)

        self.lr_tensor = sample["lr"]
        self.hr_tensor = sample["hr"]

        return self.lr_tensor, self.hr_tensor

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            ToTensor()
        ])
        return composed_transforms(sample)


def ycbcr(image):
    m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [112, -93.786, -18.214]])

    image = np.array(image)
    shape = image.shape
    if len(shape) == 3:
        image = image.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(image, m.transpose() / 255.)
    ycbcr[:, 0] += 16.
    ycbcr[:, 1:] += 128.
    ycbcr = ycbcr.reshape(shape)
    return ycbcr


class Correct_illum(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        lr = sample['lr']
        hr = sample['hr']

        lr = np.array(lr)
        hr = np.array(hr)
        mean_lr = np.mean(np.mean(lr))
        mean_hr = np.mean(np.mean(hr))

        diff = mean_lr - mean_hr
        hr = hr + diff

        return {'lr': lr, 'hr': hr, 'lr_trans': lr}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        lr = sample['lr']
        hr = sample['hr']
        # lr_trans = sample['lr_trans']

        if len(lr.shape) == 2:
            lr = np.expand_dims(lr, 2)
            hr = np.expand_dims(hr, 2)
            # lr_trans = np.expand_dims(lr_trans,2)

        lr = np.array(lr).astype(np.float32).transpose((2, 0, 1))  # whc-->chw
        hr = np.array(hr).astype(np.float32).transpose((2, 0, 1))  # whc-->chw

        lr = np.expand_dims(lr, 0)
        hr = np.expand_dims(hr, 0)

        lr /= 255.0
        hr /= 255.0

        lr = torch.from_numpy(lr).float()
        hr = torch.from_numpy(hr).float()

        return {'lr': lr, 'hr': hr}
