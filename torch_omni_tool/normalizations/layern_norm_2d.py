# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 21:15:35 2026

@author: Mateo-drr
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Literal


class LayerNorm(nn.Module):
    """
    LayerNorm 2d
    Source: convnext
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: Literal["chan_first", "chan_last"] = "chan_first",
    ) -> None:
        """
        Build a 2d LayerNorm
        Supports two data formats: chan_first (default) or
        chan_last. The ordering of the dimensions in the inputs. chan_last
        corresponds to inputs with shape [b, h, w, c], while chan_first
        corresponds to inputs with shape [b, c, h, w].
        Args
            self: LayerNorm instance
            normalized_shape: Number of channels that are normalized together
            eps: Value added to the variance for numerical stability
            data_format: Layout of the input, either chan_first or chan_last
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["chan_last", "chan_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the channel dimension and apply the learned scale and bias
        Args
            self: LayerNorm instance
            x: Input tensor with shape [b, c, h, w] or [b, h, w, c] depending on data_format
        Returns
            torch.Tensor: Normalized tensor with the same shape as the input
        """
        if self.data_format == "chan_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "chan_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        else:
            raise NotImplementedError
