import torch
from torch import nn


class GRN(nn.Module):
    """
    GRN (Global Response Normalization) layer for B,C,H,W format
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)  # normalizations over H,W
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)  # normalize over C
        return self.gamma * (x * nx) + self.beta + x
