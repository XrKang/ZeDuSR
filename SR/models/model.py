import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from models.model_lib.zssr import ZSSRNet
from models.model_lib.vdsr import VDSR
from models.model_lib.rcan import RCAN
from math import sqrt


class MODEL(nn.Module):
    def __init__(self, args):
        super(MODEL, self).__init__()
        # self.stn  =STN()
        # self.vdsr = VDSR()
        self.srmodel = RCAN(args)

        
    def forward(self, x):

        out = self.srmodel(x)
        return out


if __name__ =='__main__':
    input = torch.rand(6, 3, 500, 500)
    input = input.cuda()
    # input = torch.rand(6, 3, 4032, 3024)
    model = MODEL()
    # for m in model.modules():
    #     print(m)
    vdsr_params = model.vdsr.parameters()
    # vdsr_params = filter(lambda p: id(p)  in vdsr_params, model.parameters())

    print(vdsr_params)
    # print(conv_params)
    # model = model.cuda()
    # output = model(input,input)
    print(input.size())
    # print(output.size())



