from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from utils.Backward_warp_layer import Backward_warp

try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

# contains [Backward_warp_Loss]
class Backward_warp_Loss(nn.Module):
    def __init__(self):
        super(Backward_warp_Loss, self).__init__()
        self.Backward_warp = Backward_warp()
        self.L2_loss = nn.MSELoss()
        
    def forward(self,input, target, flow):
        warped_image = self.Backward_warp(input,flow)
        warped_mse_loss = self.L2_loss(warped_image, target)
        return warped_mse_loss

if __name__ == "__main__":
    loss = Backward_warp_Loss()
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 3,7, 7).cuda()
    flow = torch.rand(1,2,7,7).cuda()
    print(a)
    print(b)
    print(loss(a, b,flow))




