# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 21:15:35 2026

@author: Mateo-drr
"""

import torch
from torch import nn


class GRN(nn.Module):
    """
    GRN (Global Response Normalization) layer for B,C,H,W format
    """

    def __init__(self, dim: int) -> None:
        """
        Build a GRN layer
        Args
            self: GRN layer instance
            dim: Number of channels, must match the channel dim of the input
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the spatial L2 norm of every channel across channels and rescale the input
        Args
            self: GRN layer instance
            x: Input tensor with shape [B, C, H, W]
        Returns
            torch.Tensor: Output tensor with shape [B, C, H, W]
        """
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)  # normalizations over H,W
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)  # normalize over C
        y: torch.Tensor = self.gamma * (x * nx) + self.beta + x
        return y
