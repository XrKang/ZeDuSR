import numpy as np
import math


class Evaluator(object):
    def __init__(self):
        self.what = 0

    # def cal_psnr(self, output, gt):
    #     _,channel,_,_ = output.shape
    #
    #
    #     psnr_channel = []
    #     gt = np.array(gt).astype(np.float32)
    #     output = np.array(output).astype(np.float32)
    #
    #     diff = gt - output
    #
    #     for i in range(channel):
    #         diff_tmp = diff[:, i, :, :]
    #         diff_tmp = diff_tmp.flatten('C')
    #         psnr = math.sqrt(np.mean(diff_tmp ** 2.))
    #         psnr_channel.append(20 * math.log10(1.0 / psnr))
    #     return np.mean(psnr_channel)


    def bgr2ycbcr(self,img):

        img = img*255
        R = img[:, 0, :, :]
        G = img[:, 1, :, :]
        B = img[:, 2, :, :]

        Y = (65.481 * R + 128.553 * G + 24.966 * B + 16) / 255
        Cb = (-37.797 * R - 74.203 * G + 122 * B + 128) / 255
        Cr =  (122 * R - 93.786 * G - 18.214 * B + 128) / 255

        ycbcr = np.concatenate([Y, Cb, Cr], 0)
        return ycbcr/255

    def PSNR1(self,pred, gt, shave_border=0):
        height, width = pred.shape[2:]
        print(height,width)
        pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
        gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
        imdff = pred - gt

        # rmse = math.sqrt(np.sum(imdff ** 2)/(width*height-375*270))
        rmse = math.sqrt(np.sum(imdff ** 2)/((width-375)*(height-270)))
        if rmse == 0:
            return 100
        return 20 * math.log10(1.0 / rmse)



    def PSNR2(self,pred, gt, shave_border=0):
        height, width = gt.shape[2:]

        pred = pred[:,:,shave_border:height - shave_border, shave_border:width - shave_border]
        gt = gt[:,:,shave_border:height - shave_border, shave_border:width - shave_border]
        imdff = pred - gt

        rmse = math.sqrt(np.mean(imdff ** 2))
        if rmse == 0:
            return 100
        return 20 * math.log10(1.0 / rmse)