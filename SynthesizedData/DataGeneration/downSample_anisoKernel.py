
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as img
from scipy.ndimage import filters, measurements, interpolation
import glob
import os
import cv2


### Function for centering a kernel
def kernel_shift(kernel, sf):
    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (sf - (kernel.shape[0] % 2))

    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass

    # Finally shift the kernel and return
    return interpolation.shift(kernel, shift_vec)



## Function for generating one kernel
def gen_kernel(k_size, scale_factor, min_var, max_var):
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = min_var + np.random.rand() * (max_var - min_var)
    lambda_2 = min_var + np.random.rand() * (max_var - min_var)
    theta = np.random.rand() * np.pi # ori
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2]);
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position (shifting kernel for aligned image)
    MU = k_size // 2 + 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - MU
    ZZ_t = ZZ.transpose(0, 1, 3, 2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

    # shift the kernel so it will be centered
    raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)

    # Normalize the kernel and return
    kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
    return kernel



## Function for downscaling an image using a kernel
def downscale(im, kernel, scale_factor, output_shape=None):
    # output shape can either be specified or, for simple cases, can be calculated.
    # see more details regarding this at: https://github.com/assafshocher/Resizer
    if output_shape is None:
        output_shape = np.array(im.shape[:-1]) / np.array(scale_factor)

    # First run a correlation (convolution with flipped kernel)
    out_im = np.zeros_like(im)
    for channel in range(np.ndim(im)):
        out_im[:, :, channel] = filters.correlate(im[:, :, channel], kernel)

    # Then subsample and return
    return out_im[np.round(np.linspace(0, im.shape[0] - scale_factor[0], output_shape[0])).astype(int)[:, None],
           np.round(np.linspace(0, im.shape[1] - scale_factor[1], output_shape[1])).astype(int), :]


if __name__ == '__main__':
    import imageio

    import argparse

    parser = argparse.ArgumentParser(description='kernelDownSample')

    # ----------------------------------------------------------------------------------------------------

    parser.add_argument('--image_path', type=str, default='./WideView_GT',
                        help='HR image path')
    parser.add_argument('--save_path', type=str, default=r'./WideView_aniso2x',
                        help='LR image save path')
    parser.add_argument('--kernel_path', type=str, default=r'./kernel_WideView_aniso2x',
                        help='Kernel save path')
    # ----------------------------------------------------------------------------------------------------

    parser.add_argument('--scale', type=float, default=2, help='downsample scale')
    parser.add_argument('--kernel_size', type=int, default=11, help='kernel size')

    parser.add_argument('--min_var', type=float, default=0.6, help='min_var')

    parser.add_argument('--max_var', type=float, default=5.0, help='min_var')

    parser.add_argument('--noise_level', type=float, default=0.25, help='noise_level')

    args = parser.parse_args()

    scale_factor = np.array([args.scale, args.scale])
    min_var = args.min_var
    max_var = args.max_var
    k_size = np.array([args.kernel_size, args.kernel_size])  # size of the kernel, should have room for the gaussian
    noise_level = args.noise_level
    # this option allows deviation from just a gaussian, by adding multiplicative noise noise

    images_path = args.image_path

    output_path = args.save_path
    output_path = os.path.join(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    kernel_path = args.kernel_path
    kernel_path = os.path.join(kernel_path)
    if not os.path.exists(kernel_path):
        os.makedirs(kernel_path)

    filenames = os.listdir(images_path)

    for name in filenames:
        img_path = os.path.join(images_path, name)
        print(name, img_path)

        im = imageio.imread(img_path).astype(np.float32)/255.0

        kernel = gen_kernel(k_size, scale_factor, min_var, max_var)

        lr = downscale(im, kernel, scale_factor)
        lr = (np.clip(lr, 0.0, 1.0)*255.0).astype(np.uint8)

        save_path = os.path.join(output_path, name[:-4]+'.png')

        print(save_path)
        imageio.imsave(save_path, lr)

        save_kernel_path = os.path.join(kernel_path, name)
        img.imsave(save_kernel_path, kernel)






