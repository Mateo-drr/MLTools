# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:13:55 2024

@author: Mateo-drr
"""

import torch
import torch.nn as nn


class SampleNet(nn.Module):
    """
    Placeholder autoencoder style model
    """

    def __init__(self) -> None:
        """
        Build the sample model
        Args
            self: SampleNet instance
        """
        super().__init__()
        self.fc1 = nn.Linear(16, 5)
        self.fc2 = nn.Linear(5, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply both linear layers with a ReLU in between
        Args
            self: SampleNet instance
            x: Input tensor with shape [B, ..., 16]
        Returns
            torch.Tensor: Output tensor with shape [B, ..., 16]
        """
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
