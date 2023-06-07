##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import math
from torch.optim import lr_scheduler

class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, lr_step=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step

    def __call__(self, optimizer, iteration):
        # if self.mode == 'cos':
        #     lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        # elif self.mode == 'poly':
        #     # lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        #     lr = self.lr*(0.9999**int(epoch))
        if self.mode == 'step':
            lr = self.lr * (0.5 ** (iteration // self.lr_step))
        else:
            raise NotImplemented

        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)
        return lr
    def _adjust_learning_rate(self, optimizer, lr):
        # if len(optimizer.param_groups) == 1:
        optimizer.param_groups[0]['lr'] = lr

