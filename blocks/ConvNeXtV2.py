
import torch
from torch import nn
from norm.LayerNorm2d import LayerNorm


class GRN(nn.Module):
    """
    GRN (Global Response Normalization) layer for B,C,H,W format
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)  # norm over H,W
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)  # normalize over C
        return self.gamma * (x * nx) + self.beta + x

class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        # depthwise conv
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="chan_last")
        # pointwise/1x1 convs, implemented with linear layers
        self.pw_conv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim),requires_grad=True
        ) if layer_scale_init_value > 0 else None
        # TODO
        # self.drop_path = DropPath(
        #     drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x0 = x
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = x0 + x
        return x
