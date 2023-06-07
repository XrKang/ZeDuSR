import numpy as np
import math
from utils.util import clamp_and_2numpy
class Evaluator(object):
    def __init__(self):
        self.what = 0

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
        height, width = pred.shape[2:]
        pred = clamp_and_2numpy(pred)
        gt = clamp_and_2numpy(gt)

        gt = np.array(gt)
        pred = pred[:,:,shave_border:height - shave_border, shave_border:width - shave_border]
        gt = gt[:,:,shave_border:height - shave_border, shave_border:width - shave_border]
        imdff = pred - gt

        rmse = math.sqrt(np.mean(imdff ** 2))
        if rmse == 0:
            return 100
        return 20 * math.log10(1.0 / rmse)