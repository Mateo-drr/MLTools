# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 21:15:35 2026

@author: Mateo-drr
"""

import torch
from torch import nn


class SEBlock(nn.Module):
    """Squeeze and Excite block"""

    def __init__(self, channels: int, reduce_dim: int = 16) -> None:
        """
        Build a squeeze and excite block
        Args
            self: SE block instance
            channels: Number of input channels
            reduce_dim: Number of channels of the bottleneck inside the excitation
        """
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, reduce_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduce_dim, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool the spatial dims, compute the channel gate and rescale the input
        Args
            self: SE block instance
            x: Input tensor with shape [B, C, H, W]
        Returns
            torch.Tensor: Gated tensor with shape [B, C, H, W]
        """
        b, c, _, _ = x.size()
        y: torch.Tensor = self.squeeze(x).view(b, c)  # Squeeze: [b,c,h,w] → [b,c,1,1] → [b,c]
        y = self.excite(y)  # Excite: [b,c] → [b,r] → [b,c]
        return x * y.view(b, c, 1, 1)  # Scale: [b,c] → [b,c,1,1] then multiply
