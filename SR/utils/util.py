import numpy as np
import torch
import os
from PIL import Image
import torch.nn.functional as F
from scipy.ndimage import measurements, interpolation
import scipy.io as sio
from scipy.ndimage import filters
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import math
import cv2

def selfEnsemble(img, model, device):
    img_0 = img
    img_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img_lr = np.fliplr(img)
    img_hd = np.flipud(img)

    img_0 = ToTensor(img_0)
    img_0 = img_0.to(device)
    output_0 = model(img_0)

    img_90 = ToTensor(img_90)
    img_90 = img_90.to(device)
    output_90 = model(img_90)

    img_lr = ToTensor(img_lr)
    img_lr = img_lr.to(device)
    output_lr = model(img_lr)

    img_hd = ToTensor(img_hd)
    img_hd = img_hd.to(device)
    output_hd = model(img_hd)

    output_0 = torch.clamp(output_0, 0, 1)
    output_0 = output_0.squeeze().detach().cpu().numpy()
    output_0 = np.transpose(output_0 * 255.0, (1, 2, 0))

    output_90 = torch.clamp(output_90, 0, 1)
    output_90 = output_90.squeeze().detach().cpu().numpy()
    output_90 = np.transpose(output_90 * 255.0, (1, 2, 0))

    output_lr = torch.clamp(output_lr, 0, 1)
    output_lr = output_lr.squeeze().detach().cpu().numpy()
    output_lr = np.transpose(output_lr * 255.0, (1, 2, 0))

    output_hd = torch.clamp(output_hd, 0, 1)
    output_hd = output_hd.squeeze().detach().cpu().numpy()
    output_hd = np.transpose(output_hd * 255.0, (1, 2, 0))

    output_90to0 = cv2.rotate(output_90, cv2.ROTATE_90_COUNTERCLOCKWISE)
    output_lrto0 = np.fliplr(output_lr)
    output_hdto0 = np.flipud(output_hd)

    output = (output_0 + output_90to0 + output_lrto0 + output_hdto0)/4

    return output



def selfEnsemble_rot(img, model, device):
    img_0 = img
    img_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img_180 = cv2.rotate(img, cv2.ROTATE_180)
    img_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img_0 = ToTensor(img_0)
    img_0 = img_0.to(device)
    output_0 = model(img_0)

    img_90 = ToTensor(img_90)
    img_90 = img_90.to(device)
    output_90 = model(img_90)

    img_180 = ToTensor(img_180)
    img_180 = img_180.to(device)
    output_180 = model(img_180)

    img_270 = ToTensor(img_270)
    img_270 = img_270.to(device)
    output_270 = model(img_270)

    output_0 = torch.clamp(output_0, 0, 1)
    output_0 = output_0.squeeze().detach().cpu().numpy()
    output_0 = np.transpose(output_0 * 255.0, (1, 2, 0))

    output_90 = torch.clamp(output_90, 0, 1)
    output_90 = output_90.squeeze().detach().cpu().numpy()
    output_90 = np.transpose(output_90 * 255.0, (1, 2, 0))

    output_180 = torch.clamp(output_180, 0, 1)
    output_180 = output_180.squeeze().detach().cpu().numpy()
    output_180 = np.transpose(output_180 * 255.0, (1, 2, 0))

    output_270 = torch.clamp(output_270, 0, 1)
    output_270 = output_270.squeeze().detach().cpu().numpy()
    output_270 = np.transpose(output_270 * 255.0, (1, 2, 0))

    output_90to0 = cv2.rotate(output_90, cv2.ROTATE_90_COUNTERCLOCKWISE)
    output_180to0 = cv2.rotate(output_180, cv2.ROTATE_180)
    output_270to0 = cv2.rotate(output_270, cv2.ROTATE_90_CLOCKWISE)

    output = (output_0 + output_90to0 + output_180to0 + output_270to0)/4

    return output


def selfEnsemble_rot_vdsr(img, model, device, args):
    img_0 = img
    img_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img_180 = cv2.rotate(img, cv2.ROTATE_180)
    img_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img_0 = ToTensor(img_0)
    img_0 = img_0.to(device)
    img_0 = F.interpolate(img_0, scale_factor=args.scale, mode='bicubic')
    output_0 = model(img_0)

    img_90 = ToTensor(img_90)
    img_90 = img_90.to(device)
    img_90 = F.interpolate(img_90, scale_factor=args.scale, mode='bicubic')
    output_90 = model(img_90)

    img_180 = ToTensor(img_180)
    img_180 = img_180.to(device)
    img_180 = F.interpolate(img_180, scale_factor=args.scale, mode='bicubic')
    output_180 = model(img_180)

    img_270 = ToTensor(img_270)
    img_270 = img_270.to(device)
    img_270 = F.interpolate(img_270, scale_factor=args.scale, mode='bicubic')
    output_270 = model(img_270)

    output_0 = torch.clamp(output_0, 0, 1)
    output_0 = output_0.squeeze().detach().cpu().numpy()
    output_0 = np.transpose(output_0 * 255.0, (1, 2, 0))

    output_90 = torch.clamp(output_90, 0, 1)
    output_90 = output_90.squeeze().detach().cpu().numpy()
    output_90 = np.transpose(output_90 * 255.0, (1, 2, 0))

    output_180 = torch.clamp(output_180, 0, 1)
    output_180 = output_180.squeeze().detach().cpu().numpy()
    output_180 = np.transpose(output_180 * 255.0, (1, 2, 0))

    output_270 = torch.clamp(output_270, 0, 1)
    output_270 = output_270.squeeze().detach().cpu().numpy()
    output_270 = np.transpose(output_270 * 255.0, (1, 2, 0))

    output_90to0 = cv2.rotate(output_90, cv2.ROTATE_90_COUNTERCLOCKWISE)
    output_180to0 = cv2.rotate(output_180, cv2.ROTATE_180)
    output_270to0 = cv2.rotate(output_270, cv2.ROTATE_90_CLOCKWISE)

    output = (output_0 + output_90to0 + output_180to0 + output_270to0)/4

    return output


def cal_psnr_np(img1, img2):
    ### args:
    # img1: [h, w, c], range [0, 255]
    # img2: [h, w, c], range [0, 255]
    diff = (img1 - img2) / 255.0
    diff[:, :, 0] = diff[:, :, 0] * 65.738 / 256.0
    diff[:, :, 1] = diff[:, :, 1] * 129.057 / 256.0
    diff[:, :, 2] = diff[:, :, 2] * 25.064 / 256.0

    diff = np.sum(diff, axis=2)
    mse = np.mean(np.power(diff, 2))
    return -10 * math.log10(mse)


def cal_ssim_np(img1, img2):
    def ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    ### args:
    # img1: [h, w, c], range [0, 255]
    # img2: [h, w, c], range [0, 255]
    # the same outputs as MATLAB's
    border = 0
    img1_y = np.dot(img1, [65.738, 129.057, 25.064]) / 256.0 + 16.0
    img2_y = np.dot(img2, [65.738, 129.057, 25.064]) / 256.0 + 16.0
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1_y = img1_y[border:h - border, border:w - border]
    img2_y = img2_y[border:h - border, border:w - border]

    if img1_y.ndim == 2:
        return ssim(img1_y, img2_y)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def compare_metric(img_pred, img_gt):
    psnr = cal_psnr_np(img_pred, img_gt)
    ssim = cal_ssim_np(img_pred, img_gt)
    return psnr, ssim


def clamp_and_2numpy(tensor):
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.detach().cpu().numpy()


def clamp_value(tensor):
    return torch.clamp(tensor, 0, 1)


def tensor2numpy(tensor):
    return tensor.detach().cpu().numpy()


def save_image(args, image, iteration, suffix='pred'):
    # im_save = np.squeeze(image)
    # im_save = np.array(im_save)

    im_save = Image.fromarray(image.astype(np.uint8))
    im_save.save(os.path.join(args.output_path, args.dataset, 'out_%d%s.png' % (iteration, suffix)))


def save_init_images(args, lr, hr, lr_gt):
    save_image(args, lr, 0, 'lr')
    save_image(args, hr, 0, 'hr')
    save_image(args, lr_gt, 0, 'lr_gt')


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
    k = np.squeeze(k)
    # Zeroize negligible values
    significant_k = zeroize_negligible_val(k, n)
    # Force centralization on the kernel
    centralized_k = kernel_shift(significant_k, sf=2)
    # return shave_a2b(centralized_k, k)
    return centralized_k


def save_final_kernel(k_2, iteration, args):
    """saves the final kernel and the analytic kernel to the results folder"""
    # k_2 = k_2[3:-3,3:-3]
    sio.savemat(os.path.join(args.output_path, args.dataset, args.checkname, 'kernel_%d.mat' % iteration),
                {'Kernel': k_2})
    im_save = np.squeeze(k_2)
    im_save = np.array(im_save)
    im_save = im_save[3:-3, 3:-3]
    im_save = (im_save - np.min(im_save)) / (np.max(im_save) - np.min(im_save)) * 255
    # im_save = np.transpose(im_save, (1, 0))
    im_save = Image.fromarray(im_save.astype(np.uint8))
    im_save.save(os.path.join(args.output_path, args.dataset, args.checkname, 'kernel_%d.png' % iteration))


def ycbcr2rgb(ycbcr1):
    # input image range should be [0,255]
    m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [112, -93.786, -18.214]])
    ycbcr = ycbcr1.copy()
    shape = ycbcr.shape

    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = ycbcr
    rgb[:, 0] -= 16.
    rgb[:, 1:] -= 128.
    #     np.linalg.inv(m.transpose()
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    rgb = rgb.clip(0, 255).reshape(shape)
    return rgb


def rgb2ycbcr(image1):
    m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [112, -93.786, -18.214]])

    image1 = np.array(image1)
    image = image1.copy()
    shape = image.shape
    if len(shape) == 3:
        image = image.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(image, m.transpose() / 255.)
    ycbcr[:, 0] += 16.
    ycbcr[:, 1:] += 128.
    ycbcr = ycbcr.reshape(shape)
    return ycbcr


def ToTensor(image):
    image = np.expand_dims(image, 0)
    # image = np.expand_dims(image,0)

    image = np.array(image).astype(np.float32).transpose((0, 3, 1, 2))  # whc-->chw
    image /= 255.0
    image = torch.from_numpy(image).float()
    return image


def cal_psnr(img1, img2):
    img1 = img1.cpu()
    img2 = img2.cpu()
    img1_np = img1.detach().numpy()
    img2_np = img2.detach().numpy()
    return compare_psnr(img1_np, img2_np)
