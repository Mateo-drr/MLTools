
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
            num_pos_feats: int,
            temperature=10000,
            normalize=True,
            scale=None
    ):
        """
        num_pos_feats (int): Number of channels // 2
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        b, _, h, w = x.shape
        # Create coordinate grids
        y_embed = torch.arange(
            1, h + 1, dtype=torch.float32, device=x.device
        ).view(1, h, 1).repeat(b, 1, w)
        x_embed = torch.arange(
            1, w + 1, dtype=torch.float32, device=x.device
        ).view(1, 1, w).repeat(b, h, 1)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=x.device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
