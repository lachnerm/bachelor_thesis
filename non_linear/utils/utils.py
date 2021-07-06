import torch
from pytorch_ssim import ssim
from torch import nn


def calc_pc(x, y, keep_first_dim=False):
    """
    Wrapper function for the computation of the pearson correlation coefficient for the given two tensors.

    :param x: first tensor
    :param y: second tensor
    :param keep_first_dim: whether to consider the first dimension of the tensors for the computation as well or to
                           compute the pc individually for each entry there.
    :return: pearson correlation coefficient for the provided tensors
    """
    return torch.stack([pearsonr(r, g) for r, g in zip(x, y)]) if keep_first_dim else pearsonr(x, y)


def pearsonr(x, y):
    """
    Computes the pearson correlation coefficient.

    :param x: first tensor
    :param y: second tensor
    :return: pearson correlation coefficient
    """
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))


def calc_ssim(x, y, keep_first_dim=False):
    """
    Computes the structural similarity index.

    :param x: first tensor
    :param y: second tensor
    :param keep_first_dim: whether to consider the first dimension of the tensors for the computation as well or to
                           compute the ssim individually for each entry there.
    :return: structural similarity index for the provided tensors
    """
    return torch.stack(
        [ssim(torch.unsqueeze(r, dim=0), torch.unsqueeze(g, dim=0)) for r, g in zip(x, y)]) if keep_first_dim else ssim(
        x, y)


def weights_init(m):
    """
    Initializes the weights of the given layer using a normal distribution for (transposed) convolutional layers and
    batch normalization layers, as well as all biases of batch normalization layers with 0.
    :param m: layer to init the weights for
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
