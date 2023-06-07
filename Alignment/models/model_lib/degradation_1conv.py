import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn as nn
import numpy as np
import scipy.stats as st
import torch.utils.model_zoo as model_zoo
# from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from math import sqrt
# from models.model_lib.simple_color_transfer import CONV4
from utils.Backward_warp_layer import Backward_warp


def gauss_kernel(kernlen=15, nsig=3, channels=1):
  interval = (2 * nsig + 1.) / (kernlen)
  x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
  kern1d = np.diff(st.norm.cdf(x))
  kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
  kernel = kernel_raw / kernel_raw.sum()
  out_filter = np.array(kernel, dtype=np.float32)
  out_filter = out_filter.reshape(( 1, 1, kernlen, kernlen))
  out_filter = np.repeat(out_filter, channels, axis=2)
  return out_filter

class Degradation_Model(nn.Module):
    def __init__(self):
        super(Degradation_Model,self).__init__()
        init_kernel = gauss_kernel(kernlen=17, nsig=3, channels=1)
        init_kernel = torch.FloatTensor(init_kernel)
        self.kernel =  nn.Parameter(data=init_kernel, requires_grad=True)

        self.scale_factor = [2, 2]

        # self.conv1 = nn.Conv2d(1,1,kernel_size=15,stride=1,padding=7)
    def forward(self, x):
        out = F.conv2d(x,weight = self.kernel,stride = 1,padding = 8)
        # out = self.conv1(x)
        # out = out[:,:,np.round(np.linspace(0, out.shape[2] - self.scale_factor[0], out.shape[2] // 2)).astype(int)[:,None],
        #       np.round(np.linspace(0, out.shape[3] - self.scale_factor[1], out.shape[3] // 2)).astype(int)]
        return out
