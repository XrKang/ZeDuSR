import torch
from torch import nn
from torchvision.models.vgg import vgg16
# import contextual_loss as cl
# import contextual_loss.fuctional as F


# contains [CXloss, CoBi_Loss]
class CXLoss(nn.Module):
    def __init__(self):
        super(CXLoss, self).__init__()
        vgg = vgg16(pretrained=False)
        # vgg.load_state_dict(torch.load('./modeling/vgg16-397923af.pth'))
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def compute_cosine_distance(self, x, y):
        assert x.size() == y.size()
        N, C, H, W = x.size()  # e.g., 10 x 512 x 14 x 14. In this case, the number of points is 196 (14x14).

        y_mu = y.mean(3).mean(2).reshape(N, -1, 1, 1)
        x_centered = x - y_mu
        y_centered = y - y_mu
        x_normalized = x_centered / torch.norm(x_centered, p=2, dim=1, keepdim=True)
        y_normalized = y_centered / torch.norm(y_centered, p=2, dim=1, keepdim=True)

        # print(x_normalized.shape)
        # The equation at the bottom of page 6 in the paper
        # Vectorized computation of cosine similarity for each pair of x_i and y_j
        x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
        y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

        cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, H*W, H*W)
        d = 1 - cosine_sim  # (N, H*W, H*W)  d[n, i, j] means d_ij for n-th data
        return d

    def compute_relative_distance(self, d):
        # ?????min(cosine_sim)
        d_min, _ = torch.min(d, dim=2, keepdim=True)  # (N, H*W, 1)

        # Eq (2)
        d_tilde = d / (d_min + 1e-5)

        return d_tilde

    def compute_cx(self, d_tilde, h):
        # Eq(3)
        w = torch.exp((1 - d_tilde) / h)

        # Eq(4)
        cx_ij = w / torch.sum(w, dim=2, keepdim=True)  # (N, H*W, H*W)
        return cx_ij

    def contextual_loss(self,x, y, h=0.5):
        """Computes contextual loss between x and y.

        Args:
          x: features of shape (N, C, H, W).
          y: features of shape (N, C, H, W).

        Returns:
          cx_loss = contextual loss between x and y (Eq (1) in the paper)
        """
        d = self.compute_cosine_distance(x,y)
        d_tilde = self.compute_relative_distance(d)
        cx_feat = self.compute_cx(d_tilde,h)

        # Eq (1)
        cx = torch.mean(torch.max(cx_feat, dim=1)[0], dim=1)  # (N, )
        cx_loss = torch.mean(-torch.log(cx + 1e-5))

        return cx_loss

    def symetric_CX_loss(self,T_features, I_features):
        score = (self.contextual_loss(T_features, I_features) + self.contextual_loss(I_features, T_features)) / 2
        return score

    def forward(self, out_images, target_images):
        # Perception Loss
        perception_loss = self.symetric_CX_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        # image_loss = self.mse_loss(out_images, target_images)
        # # TV Loss
        # tv_loss = self.tv_loss(out_images)

        return  perception_loss


class CoBi_Loss(nn.Module):
    def __init__(self):
        super(CoBi_Loss,self).__init__()
        vgg = vgg16(pretrained=False)
        vgg.load_state_dict(torch.load('../../models/model_lib/vgg16-397923af.pth'))
        print('ssssssssssssssssssssssss')
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False

    def compute_cosine_distance(self, x, y):
        assert x.size() == y.size()
        N, C, H, W = x.size()  # e.g., 10 x 512 x 14 x 14. In this case, the number of points is 196 (14x14).

        y_mu = y.mean(3).mean(2).reshape(N, -1, 1, 1)
        x_centered = x - y_mu
        y_centered = y - y_mu
        x_normalized = x_centered / torch.norm(x_centered, p=2, dim=1, keepdim=True)
        y_normalized = y_centered / torch.norm(y_centered, p=2, dim=1, keepdim=True)

        # The equation at the bottom of page 6 in the paper
        # Vectorized computation of cosine similarity for each pair of x_i and y_j
        x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
        y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

        cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, H*W, H*W)
        d = 1 - cosine_sim  # (N, H*W, H*W)  d[n, i, j] means d_ij for n-th data
        return d

    def compute_relative_distance(self, d):
        # ?????min(cosine_sim)
        d_min, _ = torch.min(d, dim=2, keepdim=True)  # (N, H*W, 1)
        d_tilde = d / (d_min + 1e-5)

        return d_tilde

    def compute_cx(self, d_tilde, h=0.5):
        w = torch.exp((1 - d_tilde) / h)
        cx_ij = w / torch.sum(w, dim=2, keepdim=True)  # (N, H*W, H*W)
        return cx_ij

    def compute_meshgrid(self, shape):
        N, C, H, W = shape
        rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
        cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)

        feature_grid = torch.meshgrid(rows, cols)
        feature_grid = torch.stack(feature_grid).unsqueeze(0)
        feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)

        return feature_grid
    def compute_spatial_loss(self,x,h=0.5):
        # spatial loss
        grid = self.compute_meshgrid(x.shape)
        print(grid)
        exit()
        d = self.compute_cosine_distance(grid, grid)
        d_tilde = self.commpute_relative_distance(d)
        cx_sp = self.compute_cx(d_tilde,h)
        return cx_sp

    def compute_feat_loss(self,x,y,h=0.5):
        # feature loss
        d = self.compute_cosine_distance(x,y)
        d_tilde = self.compute_relative_distance(d)
        cx_feat = self.compute_cx(d_tilde,h)
        return cx_feat

    def cobi_vgg(self,out_images,target_images,w=0.1):
        sp_loss = self.compute_spatial_loss(vgg16(out_images),h=0.5)
        feat_loss = self.compute_feat_loss(vgg16(out_images), vgg16(target_images), h=0.5)
        combine_loss = (1 - w) * feat_loss + w * sp_loss
        return combine_loss

    def cobi_rgb(self,out_images,target_images,w=0.1):
        sp_loss = self.compute_spatial_loss(out_images,h=0.5)
        feat_loss = self.compute_feat_loss(out_images, target_images, h=0.5)
        combine_loss = (1 - w) * feat_loss + w * sp_loss
        return combine_loss

    def forward(self, out_images, target_images, w=0.1):
        loss = self.cobi_vgg(out_images,target_images) + self.cobi_rgb(out_images,target_images)
        return loss


if __name__ == "__main__":
    # cx_loss = CXLoss()
    # # print(cx_loss)
    # # CX_LOSS
    # torch.manual_seed(2)
    # x = torch.rand(10, 3, 100, 100)
    # y = torch.rand(10, 3, 100, 100)
    # cx_loss = cx_loss(x, y)
    # print(cx_loss)

    cobi_loss = CoBi_Loss()

    # input features
    img1 = torch.rand(1, 3, 45, 45)
    img2 = torch.rand(1, 3, 45, 45)

    # contextual loss
    # criterion = cobi_loss()
    loss = cobi_loss(img1, img2)

    print(loss)
    # # functional call
    # loss = F.contextual_loss(img1, img2, band_width=0.1, loss_type='cosine')
    #
    # # comparing with VGG features
    # # if `use_vgg` is set, VGG model will be created inside of the criterion
    # criterion = cl.ContextualLoss(use_vgg=True, vgg_layer='relu5_4')
    # loss = criterion(img1, img2)
