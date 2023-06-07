import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as img
from scipy.ndimage import filters, measurements, interpolation
from scipy.io import savemat
# import ntpath
from PIL import Image
import os
from utils.util import kernel_shift,zeroize_negligible_val

def gen_kernel(k_size, scale_factor, min_var, max_var,noise_level):
    
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = min_var + np.random.rand() * (max_var - min_var);
    lambda_2 = min_var + np.random.rand() * (max_var - min_var);
    theta = np.random.rand() * np.pi
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2
    
    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2]);
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]
    
    # Set expectation position (shifting kernel for aligned image)
    MU = k_size // 2  + 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]
    
    # Create meshgrid for Gaussian
    [X,Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]
    
    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z-MU
    ZZ_t = ZZ.transpose(0,1,3,2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)
    
    # shift the kernel so it will be centered
    raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)
    raw_kernel_centered = raw_kernel
    
    # Normalize the kernel and return
    kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
    return kernel


def downscale(im, kernel, scale_factor, output_shape=None):
    # output shape can either be specified or, for simple cases, can be calculated.
    # see more details regarding this at: https://github.com/assafshocher/Resizer
    if output_shape is None:
        output_shape = np.array(im.shape[:-1]) // np.array(scale_factor)
    
    # First run a correlation (convolution with flipped kernel)
    out_im = np.zeros_like(im)
    # for channel in range(np.ndim(im)):
    out_im[:, :] = filters.correlate(im[:, :], kernel)

    # Then subsample and return
    AA = out_im[np.round(np.linspace(0, im.shape[0] - scale_factor[0], output_shape[0])).astype(int)[:, None],
                  np.round(np.linspace(0, im.shape[1] - scale_factor[1], output_shape[1])).astype(int)]
    out_im = np.clip(out_im,0,1)

    return AA


def generate_degradation(image,args):
    np.random.seed(args.kernel_seed)
    scale_factor = np.array([2, 2])  # choose scale-factor
    avg_sf = np.mean(2)  # this is calculated so that min_var and max_var will be more intutitive
    min_var = 0.175 * avg_sf  # variance of the gaussian kernel will be sampled between min_var and max_var
    max_var = 2.5 * avg_sf
    k_size = np.array([17, 17])  # size of the kernel, should have room for the gaussian
    noise_level = 0.4  # this option allows deviation from just a gaussian, by adding multiplicative noise noise

    kernel = gen_kernel(k_size, scale_factor, min_var, max_var,noise_level)
    kernel = zeroize_negligible_val(kernel, n=40)
    kernel = kernel_shift(kernel,2)
    lr = downscale(image, kernel, scale_factor)

    savemat(os.path.join(args.output_path, args.dataset, args.checkname,'kernel_GT.mat'), {'Kernel': kernel})

    kernel = (kernel-np.min(kernel)) / (np.max(kernel)-np.min(kernel)) * 255
    kernel = Image.fromarray(kernel.astype(np.uint8))
    kernel.save(os.path.join(args.output_path, args.dataset, args.checkname,'kernel_GT.png'))

    return lr


if '__name__'=='__main__':

    lr_images = []
    kernels = []
    i=1
    np.random.seed(10)
    # for i, path in enumerate(glob.glob(images_path)):
    im = img.imread(images_path)
    kernel = gen_kernel(k_size, scale_factor, min_var, max_var)
    lr = downscale(im, kernel, scale_factor)
    # print(i)
    plt.subplot(1, 2, 1)
    plt.imshow(lr)
    plt.subplot(1, 2, 2)
    plt.imshow(kernel, cmap='gray')
    plt.show()
    kernel = kernel
    savemat(('%s_kernel_x4.mat' % i), {'Kernel': kernel})
    #savemat('%s/im_%d_sf_%d_%d.mat' % (output_path, i, scale_factor[0], scale_factor[1]), {'ker': kernel})

    lr = lr*255
    lr = Image.fromarray(lr.astype(np.uint8))
    lr.save('%s/im_%d_sf_%d_%d.png' % (output_path, i, scale_factor[0], scale_factor[1]))

    kernel = kernel*255
    kernel = Image.fromarray(kernel.astype(np.uint8))
    kernel.save('%s/kernel_%d_sf_%d_%d.png' % (output_path, i, scale_factor[0], scale_factor[1]))


    # plt.imsave('%s/im_%d_sf_%d_%d.png' % (output_path, i, scale_factor[0], scale_factor[1]), lr, vmin=0, vmax=1)
    # plt.imsave('%s/kernel_%d_sf_%d_%d.png' % (output_path, i, scale_factor[0], scale_factor[1]), kernel, vmin=0, vmax=1)