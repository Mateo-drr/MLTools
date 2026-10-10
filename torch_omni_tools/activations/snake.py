# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 22:38:08 2026

@author: Mateo-drr
"""

import torch
from torch import nn


class Snake(nn.Module):
    """
    Sine based periodic activation function
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        """
        Build a snake activation
        Args
            self: Snake instance
            channels: Number of channels of the input, shapes the learnable frequency
            eps: Value added to the frequency to avoid dividing by zero
        """
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.a = nn.Parameter(torch.ones(1, channels, 1, 1) / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add a learnable periodic term to the input
        Args
            self: Snake instance
            x: Input tensor with shape [B, C, H, W]
        Returns
            torch.Tensor: Output tensor with shape [B, C, H, W]
        """
        x = x + (1 / (self.a + self.eps)) * torch.sin(self.a * x) ** 2
        return x
