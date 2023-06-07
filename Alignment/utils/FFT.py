import torch
import numpy as np
import cv2

def FFT_tensor(img_tensor):
    _, _, h, w = img_tensor.shape
    img_FFT = torch.fft.fft2(img_tensor, dim=(2, 3))
    img_FFT_shift = torch.roll(img_FFT, (h // 2, w // 2), dims=(2, 3))
    Amg_FFT_shift = torch.abs(img_FFT_shift)
    Amg_FFT_shift = torch.log(Amg_FFT_shift + 1e-8)
    Amg_FFT_shift = Amg_FFT_shift.type(torch.float32)
    return Amg_FFT_shift
    # return img_FFT_shift





if __name__ == '__main__':

    img_path = '/data/ruikang/iphone12/test/ultra2wide/wide_align_cor/0002.png'
    img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1]
    # # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img.shape)
    # # cv2.imwrite("/data/ruikang/0002.png", img.astype(np.uint8))
    #
    # img_ori = img
    # img_numpyFFT = np.fft.fft2(img, axes=(0, 1))
    # # print(img_numpyFFT.shape)
    # # cv2.imwrite("/data/ruikang/0002_numpyFFT.png", img_numpyFFT.astype(np.uint8))
    # img_numpyFFT = np.fft.fftshift(img_numpyFFT)
    # # cv2.imwrite("/data/ruikang/0002_numpyFFT_shift.png", img_numpyFFT.astype(np.uint8))
    # #
    # mag_numpyFFT = np.abs(img_numpyFFT)
    # mag_numpyFFT = 20*np.log(mag_numpyFFT)
    # mag_numpyFFT = cv2.normalize(mag_numpyFFT, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    # cv2.imwrite("/data/ruikang/0002_numpyFFT_mag_log.png", mag_numpyFFT)
    # #
    # pha_numpyFFT = 20*np.abs(np.angle(img_numpyFFT))
    # mag_numpyFFT = cv2.normalize(pha_numpyFFT, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    # cv2.imwrite("/data/ruikang/0002_numpyFFT_pha.png", pha_numpyFFT)



    img_t = torch.from_numpy((img/255.0).astype(np.float32).transpose(2, 0, 1)).unsqueeze(0)
    print(img_t.shape)
    _, _, h, w = img_t.shape

    img_torchFFT = torch.fft.fft2(img_t, dim=(2, 3))
    img_torchFFT_shift = torch.roll(img_torchFFT, (h//2, w//2), dims=(2, 3))
    mag_torchFFT = torch.abs(img_torchFFT_shift)
    mag_torchFFT_log = torch.log(mag_torchFFT)
    print(img_torchFFT_shift.shape)
    print(mag_torchFFT_log.shape)
    print(mag_torchFFT_log)

    # mag_torchFFT = torch.abs(img_torchFFT_shift)
    # mag_torchFFT_log = torch.log(mag_torchFFT)
    #
    # mag_torchFFT_log = cv2.normalize(mag_torchFFT_log.squeeze(0).numpy().transpose(1, 2, 0)*255.0, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    # cv2.imwrite("/data/ruikang/0002_torchFFT_mag_log.png", mag_torchFFT_log)
    #
    # # mag_torchFFT = cv2.normalize(mag_torchFFT.squeeze(0).numpy().transpose(1, 2, 0)*255.0, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    # # cv2.imwrite("/data/ruikang/0002_torchFFT_mag.png", mag_torchFFT)
    #
    #
    # pha_torchFFT = torch.abs(torch.angle(img_torchFFT_shift))*255.0*20
    # pha_torchFFT = cv2.normalize(pha_torchFFT.squeeze(0).numpy().transpose(1, 2, 0), 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    # cv2.imwrite("/data/ruikang/0002_torchFFT_pha.png", pha_torchFFT)

