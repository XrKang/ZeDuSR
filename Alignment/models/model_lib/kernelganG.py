import torch
import torch.nn as nn
from util import swap_axis


class Generator(nn.Module):
    def __init__(self, conf):
        super(Generator, self).__init__()
        struct = conf.G_structure#[7, 5, 3, 1, 1, 1]
        # First layer - Converting RGB image to latent space
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=conf.G_chan, kernel_size=struct[0], bias=False)

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            feature_block += [nn.Conv2d(in_channels=conf.G_chan, out_channels=conf.G_chan, kernel_size=struct[layer], bias=False)]
        self.feature_block = nn.Sequential(*feature_block)
        # Final layer - Down-sampling and converting back to image
        self.final_layer = nn.Conv2d(in_channels=conf.G_chan, out_channels=1, kernel_size=struct[-1],
                                     stride=int(1 / conf.scale_factor), bias=False)

        # Calculate number of pixels shaved in the forward pass
        self.output_size = self.forward(torch.FloatTensor(torch.ones([1, 1, conf.input_crop_size, conf.input_crop_size]))).shape[-1]
        self.forward_shave = int(conf.input_crop_size * conf.scale_factor) - self.output_size #(64*0.5-)26=6

    def forward(self, input_tensor):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        input_tensor = swap_axis(input_tensor) #(3,1,64,64)-->(3,1,64,64)
        downscaled = self.first_layer(input_tensor)#(3,1,64,64)-->(3,64,58,58)
        features = self.feature_block(downscaled) #(3,64,58,58)-->(3,64,52,52)
        output = self.final_layer(features) #(3,64,52,52)-->(3,1,26,26)
        return_param = swap_axis(output) #(3,1,26,26)-->(1,3,26,26)
        return return_param


class Discriminator(nn.Module):

    def __init__(self, conf):
        super(Discriminator, self).__init__()

        # First layer - Convolution (with no ReLU) D_chan=64, D_kernel_size = 7
        self.first_layer = nn.utils.spectral_norm(nn.Conv2d(in_channels=3, out_channels=conf.D_chan, kernel_size=conf.D_kernel_size, bias=True))
        feature_block = []  # Stacking layers with 1x1 kernels
        for _ in range(1, conf.D_n_layers - 1):
            feature_block += [nn.utils.spectral_norm(nn.Conv2d(in_channels=conf.D_chan, out_channels=conf.D_chan, kernel_size=1, bias=True)),
                              nn.BatchNorm2d(conf.D_chan),
                              nn.ReLU(True)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(in_channels=conf.D_chan, out_channels=1, kernel_size=1, bias=True)),
                                         nn.Sigmoid())

        # Calculate number of pixels shaved in the forward pass
        self.forward_shave = conf.input_crop_size - self.forward(torch.FloatTensor(torch.ones([1, 3, conf.input_crop_size, conf.input_crop_size]))).shape[-1]

    def forward(self, input_tensor):
        receptive_extraction = self.first_layer(input_tensor)#([1, 3, 64, 64])-->([1, 64, 58, 58])
        features = self.feature_block(receptive_extraction)#([1, 64, 58, 58])-->([1, 64, 58, 58])
        return_param = self.final_layer(features)#([1, 64, 58, 58])-->([1, 1, 58, 58])
        return return_param

def weights_init_D(m):
    """ initialize weights of the discriminator """
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif class_name.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init_G(m):
    """ initialize weights of the generator """
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)

if __name__ == '__main__':
    input = torch.rand(6,3,64,64)
    model = Generator(10, BatchNorm=nn.BatchNorm2d)
    output = model(input)
    print('output:', output.size())
