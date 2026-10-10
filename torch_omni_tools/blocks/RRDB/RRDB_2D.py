# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:41:06 2024

@author: Mateo-drr
"""

import torch
import torch.nn as nn


class ResidualDenseBlock_5C(nn.Module):
    """
    Residual dense block with 5 convolutions and dense connections
    """

    def __init__(self, nf: int = 64, gc: int = 32, bias: bool = True) -> None:
        """
        Build a residual dense block
        Args
            self: Residual dense block instance
            nf: Number of feature channels
            gc: Growth channels, i.e. intermediate channels
            bias: Whether the convolutions use a bias term
        """
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convolve the input with every convolution and merge the outputs densely
        Args
            self: Residual dense block instance
            x: Input tensor with shape [B, C, H, W]
        Returns
            torch.Tensor: Scaled output tensor with the same shape as the input
        """
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5: torch.Tensor = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, nf: int, gc: int = 32) -> None:
        """
        Build a residual in residual dense block
        Args
            self: RRDB instance
            nf: Number of feature channels
            gc: Growth channels of every residual dense block
        """
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass the input through the three stacked residual dense blocks
        Args
            self: RRDB instance
            x: Input tensor with shape [B, C, H, W]
        Returns
            torch.Tensor: Scaled output tensor with the same shape as the input
        """
        out: torch.Tensor = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x
