"""
Normalized Cross-Correlation for pattern matching.
pytorch implementation

roger.bermudez@epfl.ch
CVLab EPFL 2019
"""



import logging
import torch
from torch.nn import functional as F
import torch.nn as nn
from math import log10
# ncc_logger = logging.getLogger(__name__)

# contains [NCC_github, NCC_pytorch]
def patch_mean(images, patch_shape):
    """
    Computes the local mean of an image or set of images.

    Args:
        images (Tensor): Expected size is (n_images, n_channels, *image_size). 1d, 2d, and 3d images are accepted.
        patch_shape (tuple): shape of the patch tensor (n_channels, *patch_size)

    Returns:
        Tensor same size as the image, with local means computed independently for each channel.

    Example::
        >>> images = torch.randn(4, 3, 15, 15)           # 4 images, 3 channels, 15x15 pixels each
        >>> patch_shape = 3, 5, 5                        # 3 channels, 5x5 pixels neighborhood
        >>> means = patch_mean(images, patch_shape)
        >>> expected_mean = images[3, 2, :5, :5].mean()  # mean of the third image, channel 2, top left 5x5 patch
        >>> computed_mean = means[3, 2, 5//2, 5//2]      # computed mean whose 5x5 neighborhood covers same patch
        >>> computed_mean.isclose(expected_mean).item()
        1
    """
    channels, *patch_size = patch_shape
    dimensions = len(patch_size)
    padding = tuple(side // 2 for side in patch_size)

    conv = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]

    # Convolution with these weights will effectively compute the channel-wise means
    patch_elements = torch.Tensor(patch_size).prod().item()
    weights = torch.full((channels, channels, *patch_size), fill_value=1 / patch_elements)
    weights = weights.to(images.device)

    # Make convolution operate on single channels
    channel_selector = torch.eye(channels).byte()
    weights[1 - channel_selector] = 0

    result = conv(images, weights, padding=padding, bias=None)

    return result


def patch_std(image, patch_shape):
    """
    Computes the local standard deviations of an image or set of images.

    Args:
        images (Tensor): Expected size is (n_images, n_channels, *image_size). 1d, 2d, and 3d images are accepted.
        patch_shape (tuple): shape of the patch tensor (n_channels, *patch_size)

    Returns:
        Tensor same size as the image, with local standard deviations computed independently for each channel.

    Example::
        >>> images = torch.randn(4, 3, 15, 15)           # 4 images, 3 channels, 15x15 pixels each
        >>> patch_shape = 3, 5, 5                        # 3 channels, 5x5 pixels neighborhood
        >>> stds = patch_std(images, patch_shape)
        >>> patch = images[3, 2, :5, :5]
        >>> expected_std = patch.std(unbiased=False)     # standard deviation of the third image, channel 2, top left 5x5 patch
        >>> computed_std = stds[3, 2, 5//2, 5//2]        # computed standard deviation whose 5x5 neighborhood covers same patch
        >>> computed_std.isclose(expected_std).item()
        1
    """
    return (patch_mean(image**2, patch_shape) - patch_mean(image, patch_shape)**2).sqrt()


def channel_normalize(template):
    """
    Z-normalize image channels independently.
    """
    reshaped_template = template.clone().view(template.shape[0], -1)
    reshaped_template.sub_(reshaped_template.mean(dim=-1, keepdim=True))
    reshaped_template.div_(reshaped_template.std(dim=-1, keepdim=True, unbiased=False))

    return reshaped_template.view_as(template)


class NCC_github(nn.Module):
    """
    Computes the [Zero-Normalized Cross-Correlation][1] between an image and a template.

    Example:
        >>> lena_path = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
        >>> lena_tensor = torch.Tensor(plt.imread(lena_path)).permute(2, 0, 1).cuda()
        >>> patch_center = 275, 275
        >>> y1, y2 = patch_center[0] - 25, patch_center[0] + 25
        >>> x1, x2 = patch_center[1] - 25, patch_center[1] + 25
        >>> lena_patch = lena_tensor[:, y1:y2 + 1, x1:x2 + 1]
        >>> ncc = NCC(lena_patch)
        >>> ncc_response = ncc(lena_tensor[None, ...])
        >>> ncc_response.max()
        tensor(1.0000, device='cuda:0')
        >>> np.unravel_index(ncc_response.argmax(), lena_tensor.shape)
        (0, 275, 275)

    [1]: https://en.wikipedia.org/wiki/Cross-correlation#Zero-normalized_cross-correlation_(ZNCC)
    """
    def __init__(self, template, keep_channels=False):
        super().__init__()

        self.keep_channels = keep_channels

        channels, *template_shape = template.shape
        print(template_shape)
        dimensions = len(template_shape)
        print(dimensions)
        self.padding = tuple(side // 2 for side in template_shape)

        self.conv_f = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]
        self.normalized_template = channel_normalize(template)
        ones = template.dim() * (1, )
        self.normalized_template = self.normalized_template.repeat(channels, *ones)
        # Make convolution operate on single channels
        channel_selector = torch.eye(channels).byte()
        self.normalized_template[1 - channel_selector] = 0
        # Reweight so that output is averaged
        patch_elements = torch.Tensor(template_shape).prod().item()
        self.normalized_template.div_(patch_elements)

    def forward(self, image):
        result = self.conv_f(image, self.normalized_template, padding=self.padding, bias=None)
        std = patch_std(image, self.normalized_template.shape[1:])
        result.div_(std)
        if not self.keep_channels:
            result = result.mean(dim=1)

        return result


# my NCC loss transformed from ncc tensorflow
class NCC_pytorch(nn.Module):
    def __init__(self, batch_size = 16, win=None, eps=1e-5):
        super(NCC_pytorch, self).__init__()
        self.win = win
        self.eps=eps
        ndims = 2
        
        if self.win is None:
            self.win = [9] * ndims
        self.padding = 1
        self.sum_filt = torch.ones(batch_size,3,3,3)
        self.sum_filt = self.sum_filt.cuda()
        self.win_size = 81
        
        
    def forward(self,I, J):
        I2 = I*I
        J2 = J*J
        IJ = I*J
        I_sum = F.conv2d(I, self.sum_filt, padding=self.padding, bias=None)
        J_sum = F.conv2d(J, self.sum_filt, padding=self.padding, bias=None)
        I2_sum = F.conv2d(I2, self.sum_filt, padding=self.padding)
        J2_sum = F.conv2d(J2, self.sum_filt, padding=self.padding)
        IJ_sum = F.conv2d(IJ, self.sum_filt, padding=self.padding)

        u_I = I_sum/self.win_size
        u_J = J_sum/self.win_size
        
        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*self.win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * self.win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * self.win_size
        cc = cross*cross / (I_var*J_var + self.eps)

        output = -log10(torch.mean(cc))
        
        return output
        
        
if __name__ =="__main__":
    from PIL import Image
    import numpy as np
    
    temp=torch.rand(16,3,200,200)
    input=torch.rand(16,3,200,200)
    ncc=NCC_pytorch(batch_size=16)
    output = np.array(ncc(temp,input))
    
    print(output)

    # output = Image.fromarray(np.uint8(output.squeeze().transpose(1,0)*255))
    # output.show()
    