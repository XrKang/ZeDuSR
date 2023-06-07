import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as img
from scipy.ndimage import filters, measurements, interpolation
import glob
from scipy.io import savemat
import ntpath
import os
import cv2
import imageio

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='kernelDownSample')
    # ----------------------------------------------------------------------------------------------------
    parser.add_argument('--image_path', type=str, default=r'./WideView_iso2x', help='HR image path')
    parser.add_argument('--save_path', type=str, default=r'./WideView_iso2x_JPEG75',
                        help='LR image save path')
    # ----------------------------------------------------------------------------------------------------

    args = parser.parse_args()

    images_path = args.image_path
    output_path = args.save_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filenames = os.listdir(images_path)

    for name in filenames:
        img_path = os.path.join(images_path, name)
        im = cv2.imread(img_path)
        save_path = os.path.join(output_path, name[:-4]+'.jpg')
        print(save_path)

        cv2.imwrite(save_path, im, [int(cv2.IMWRITE_JPEG_QUALITY), 75])



