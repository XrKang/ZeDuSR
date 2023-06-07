import numpy as np
import torch
import torch.nn as nn
import os
from PIL import Image
import torch.nn.functional as F
from scipy.ndimage import measurements, interpolation
import scipy.io as sio
from scipy.ndimage import filters
import cv2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def FFT_visual(args, img, iteration, name):
    _, _, h, w = img.shape
    img_torchFFT = torch.fft.fft2(img.detach().cpu(), dim=(2, 3))
    img_torchFFT_shift = torch.roll(img_torchFFT, (h // 2, w // 2), dims=(2, 3))
    amp_torchFFT = torch.abs(img_torchFFT_shift)
    amp_torchFFT_log = torch.log(amp_torchFFT)

    amp_torchFFT_log = cv2.normalize(amp_torchFFT_log.squeeze(0).numpy().transpose(1, 2, 0) * 255.0, 0, 255,
                                     norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    save_path = os.path.join(args.output_path, args.dataset, args.checkname, "FFT_Amp_{}_{:d}.png".format(name, iteration))
    cv2.imwrite(save_path, amp_torchFFT_log)

    pha_torchFFT = torch.abs(torch.angle(img_torchFFT_shift)) * 255.0 * 20
    pha_torchFFT = cv2.normalize(pha_torchFFT.squeeze(0).numpy().transpose(1, 2, 0), 0, 255,
                                 norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    save_path = os.path.join(args.output_path, args.dataset, args.checkname, "FFT_Pha_{}_{:d}.png".format(name, iteration))
    cv2.imwrite(save_path, pha_torchFFT)

def clamp_and_2numpy(tensor):
    tensor =  torch.clamp(tensor, 0, 1)
    return tensor.detach().cpu().numpy()

def clamp_value(tensor):
    return torch.clamp(tensor, 0, 1)

def tensor2numpy(tensor):
    return tensor.detach().cpu().numpy()

def save_image(args, image, iteration,suffix='pred'):
    im_save = np.squeeze(image)
    im_save = np.array(im_save)
    # sio.savemat(os.path.join(args.output_path, args.dataset, args.checkname, 'out_%d%s.mat'%(iteration,suffix)), {'image': im_save})

    im_save = Image.fromarray(im_save.astype(np.uint8))
    im_save.save(os.path.join(args.output_path, args.dataset, 'out_%d%s.png'%(iteration,suffix)))

def save_init_images(args,lr,hr,lr_gt):
    save_image(args,lr,0,'lr')
    save_image(args,hr,0,'hr')
    save_image(args,lr_gt,0,'lr_gt')


def calc_curr_k(model):
    """given a generator network, the function calculates the kernel it is imitating"""
    delta = torch.Tensor([1.]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
    for ind, w in enumerate(model.parameters()):
        curr_k = F.conv2d(delta, w, padding=17 - 1) if ind == 0 else F.conv2d(
            curr_k, w)
    curr_k = curr_k.squeeze().flip([0, 1])
    return curr_k

def zeroize_negligible_val(k, n):
    """Zeroize values that are negligible w.r.t to values in k"""
    # Sort K's values in order to find the n-th largest
    k_sorted = np.sort(k.flatten())
    # Define the minimum value as the 0.75 * the n-th largest value
    k_n_min = 0.75 * k_sorted[-n - 1]
    # Clip values lower than the minimum value
    filtered_k = np.clip(k - k_n_min, a_min=0, a_max=100)
    # Normalize to sum to 1
    return filtered_k / filtered_k.sum()

def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel :
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between odd and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)
    # The second term ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) // 2 + 0.5 * (np.array(sf) - (np.array(kernel.shape) % 2))
    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass

    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    kernel = np.pad(kernel, np.int(np.ceil(np.max(np.abs(shift_vec)))) + 1, 'constant')
    # Finally shift the kernel and return
    kernel = interpolation.shift(kernel, shift_vec)

    return kernel


def downscale(im, kernel, scale_factor, output_shape=None):
    # output shape can either be specified or, for simple cases, can be calculated.
    # see more details regarding this at: https://github.com/assafshocher/Resizer
    if output_shape is None:
        output_shape = np.array(im.shape[:2]) // np.array(scale_factor)

    # First run a correlation (convolution with flipped kernel)
    out_im = np.zeros_like(im)
    # for channel in range(np.ndim(im)):
    out_im[:, :] = filters.correlate(im[:, :], kernel)

    # Then subsample and return
    AA = out_im[np.round(np.linspace(0, im.shape[0] - scale_factor[0], output_shape[0])).astype(int)[:, None],
                np.round(np.linspace(0, im.shape[1] - scale_factor[1], output_shape[1])).astype(int)]
    out_im = np.clip(out_im, 0, 1)

    return AA


def post_process_k(k, n):
    """Move the kernel to the CPU, eliminate negligible values, and centralize k"""
    k = tensor2numpy(k)
    k=np.squeeze(k)
    # Zeroize negligible values
    significant_k = zeroize_negligible_val(k, n)
    # Force centralization on the kernel
    centralized_k = kernel_shift(significant_k, sf=2)
    # return shave_a2b(centralized_k, k)
    return centralized_k

def save_final_kernel(k_2, args):
    """saves the final kernel and the analytic kernel to the results folder"""
    # k_2 = k_2[3:-3,3:-3]
    sio.savemat(os.path.join(args.output_path, args.dataset, 'kernel_LPF.mat'), {'Kernel': k_2})
    im_save = np.squeeze(k_2)
    im_save = np.array(im_save)
    im_save = im_save[3:-3, 3:-3]
    im_save = (im_save-np.min(im_save)) / (np.max(im_save)-np.min(im_save)) * 255
    # im_save = np.transpose(im_save, (1, 0))
    im_save = Image.fromarray(im_save.astype(np.uint8))
    im_save.save(os.path.join(args.output_path, args.dataset, 'kernel_LPF.png'))


def create_gaussian(size, sigma1, sigma2=-1, is_tensor=False):
    """Return a Gaussian"""
    func1 = [np.exp(-z ** 2 / (2 * sigma1 ** 2)) / np.sqrt(2 * np.pi * sigma1 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    func2 = func1 if sigma2 == -1 else [np.exp(-z ** 2 / (2 * sigma2 ** 2)) / np.sqrt(2 * np.pi * sigma2 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    return torch.FloatTensor(np.outer(func1, func2)).to(device) if is_tensor else np.outer(func1, func2)



def cal_RF(ksize_list):
    ksize = ksize_list
    N_RF = 1
    for i in range(len(ksize)):
        N_RF = (N_RF - 1) + ksize[i]
    return N_RF

def map2tensor(gray_map):
    """Move gray maps to GPU, no normalization is done"""
    return torch.FloatTensor(gray_map).unsqueeze(0).unsqueeze(0).to(device)


def create_penalty_mask(k_size, penalty_scale):
    """Generate a mask of weights penalizing values close to the boundaries"""
    center_size = k_size // 2 + k_size % 2
    mask = create_gaussian(size=k_size, sigma1=k_size, is_tensor=False)
    mask = 1 - mask / np.max(mask)
    margin = (k_size - center_size) // 2 - 1
    mask[margin:-margin, margin:-margin] = 0
    return penalty_scale * mask


def ycbcr2rgb(ycbcr1):
    # input image range should be [0,255]
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    ycbcr = ycbcr1.copy()
    shape = ycbcr.shape

    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = ycbcr
    rgb[:,0] -= 16.
    rgb[:,1:] -= 128.
#     np.linalg.inv(m.transpose()
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    rgb=rgb.clip(0, 255).reshape(shape)
    return rgb


def rgb2ycbcr(image):
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


class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).to(device)
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).to(device)

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).to(device)
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).to(device)

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]

        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding = 1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding = 1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding = 1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding = 1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding = 1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding = 1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

