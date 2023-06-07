import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]






# noinspection PyUnresolvedReferences
class GANLoss(nn.Module):
    """D outputs a [0,1] map of size of the input. This map is compared in a pixel-wise manner to 1/0 according to
    whether the input is real (i.e. from the input image) or fake (i.e. from the Generator)"""

    def __init__(self):
        super(GANLoss, self).__init__()
        # The loss function is applied after the pixel-wise comparison to the true label (0/1)
        # self.loss = nn.L1Loss(reduction='mean')
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, d_last_layer, is_d_input_real):
        # Make a shape
        d_last_layer_shape = [1, 3, d_last_layer.shape[2], d_last_layer.shape[3]]
        # The two possible label maps are pre-prepared
        self.label_tensor_fake = Variable(torch.zeros(d_last_layer_shape).to(device), requires_grad=False)
        self.label_tensor_real = Variable(torch.ones(d_last_layer_shape).to(device), requires_grad=False)

        # Determine label map according to whether current input to discriminator is real or fake
        label_tensor = self.label_tensor_real if is_d_input_real else self.label_tensor_fake
        # Compute the loss
        return self.loss(d_last_layer, label_tensor)

# noinspection PyUnresolvedReferences
class RaDiscrim_Loss(nn.Module):
    """D outputs a [0,1] map of size of the input. This map is compared in a pixel-wise manner to 1/0 according to
    whether the input is real (i.e. from the input image) or fake (i.e. from the Generator)"""

    def __init__(self):
        super(RaDiscrim_Loss, self).__init__()
        # The loss function is applied after the pixel-wise comparison to the true label (0/1)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, real, g_fake):
        # Make a shape
        d_last_layer_shape = [1, 3, g_fake.shape[2], g_fake.shape[3]]
        # The two possible label maps are pre-prepared
        self.label_tensor_fake = Variable(torch.zeros(d_last_layer_shape).to(device), requires_grad=False)
        self.label_tensor_real = Variable(torch.ones(d_last_layer_shape).to(device), requires_grad=False)

        # Compute the loss
        loss1 = self.loss(real-torch.mean(g_fake),self.label_tensor_fake)
        loss2 = self.loss(g_fake-torch.mean(real),self.label_tensor_real)

        return (loss1 + loss2) / 2




if __name__ == '__main__':
    input = torch.rand(1,3,500,500)
    input1 = torch.rand(1,3,500,500)
    model = Edge_Loss()
    print(model)
    output = model(input,input1)
    print(output)

