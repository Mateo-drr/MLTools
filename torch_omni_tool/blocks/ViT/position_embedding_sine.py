# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 21:15:35 2026

@author: Mateo-drr
"""

import torch
from torch import nn
import math


class PositionEmbeddingSine(nn.Module):
    """
    Sine/Cosine positional encoding for 2D images.
    Works directly with tensors of shape [B, C, H, W].
    """

    def __init__(
        self,
        temperature: float = 10000,
        normalize: bool = True,
        scale: float | None = None,
    ) -> None:
        """
        Build a sine positional embedding
        Args
            self: Positional embedding instance
            temperature: Temperature of the sinusoidal embedding
            normalize: Whether the coordinate grids are normalized to [0, scale]
            scale: Scale of the coordinates, defaults to 2 * pi, only valid when normalize is True
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build the sine and cosine encoding of the spatial coordinates
        Args
            self: Positional embedding instance
            x: Input tensor with shape [B, C, H, W], used for device and channel count
        Returns
            torch.Tensor: Positional embedding with shape [B, C, H, W]
        """
        b, c, h, w = x.shape
        num_pos_feats = (c + 1) // 2  # round up so 2*num_pos_feats >= c

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)

        # Create coordinate grids
        y_embed = (
            torch.arange(1, h + 1, dtype=torch.float32, device=x.device)
            .view(1, h, 1)
            .repeat(b, 1, w)
        )
        x_embed = (
            torch.arange(1, w + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, w)
            .repeat(b, h, 1)
        )
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(
            0, 3, 1, 2
        )  # [B, 2*num_pos_feats, H, W]
        return pos[:, :c, :, :]  # slice back to exactly c channels
