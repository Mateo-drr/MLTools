# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 21:15:35 2026

@author: Mateo-drr
"""

import torch
from torch import nn
from torch_omni_tool.normalizations.layern_norm_2d import LayerNorm


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block, there are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    This block uses (2) as it is slightly faster in PyTorch
    """

    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6) -> None:
        """
        Build a ConvNeXt block
        Args
            self: ConvNeXt block instance
            dim: Number of input channels
            layer_scale_init_value: Initial value of LayerScale, disabled when not positive
        """
        super().__init__()
        # depthwise conv
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="chan_last")
        # pointwise/1x1 convs, implemented with linear layers
        self.pw_conv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Linear(4 * dim, dim)
        self.gamma: nn.Parameter | None = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        # TODO
        # self.drop_path = DropPath(
        #     drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the depthwise convolution, inverted bottleneck and residual connection
        Args
            self: ConvNeXt block instance
            x: Input tensor with shape [N, C, H, W]
        Returns
            torch.Tensor: Output tensor with shape [N, C, H, W]
        """
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
