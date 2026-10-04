# ViT

Positional embeddings for vision transformers. Only the (DETR-style) 2D sine/cosine
encoding is implemented here; there is no attention module in this folder.

## `PositionEmbeddingSine`

`position_embedding_sine.py`

```python
PositionEmbeddingSine(
    temperature: float = 10000,
    normalize: bool = True,
    scale: float | None = None,
)
```

Takes a `[B, C, H, W]` tensor and returns a positional embedding of the **same shape** —
it uses the input only for its device and channel count, so the result can be added to a
feature map directly:

```python
import torch
from mltools.blocks.ViT.position_embedding_sine import PositionEmbeddingSine

pos = PositionEmbeddingSine()(torch.randn(2, 128, 56, 56))     # [2, 128, 56, 56]
```

How it is built:

1. Two coordinate grids are created with `torch.arange(1, H + 1)` and `torch.arange(1, W + 1)`,
   i.e. **1-based** coordinates, broadcast over the batch.
2. With `normalize=True` (default) each grid is divided by its last value and multiplied by
   `scale` (default `2 * pi`), mapping coordinates to `[0, 2 * pi]`. Passing `scale` while
   `normalize=False` raises `ValueError`.
3. Each coordinate is divided by `temperature ** (2 * (i // 2) / num_pos_feats)` and split
   into an interleaved sine/cosine pair — the standard transformer encoding, applied
   per axis.
4. The `y` encoding is concatenated with the `x` encoding along the channel axis, permuted
   back to `[B, *, H, W]` and sliced to exactly `C` channels.

## Caveats

- `num_pos_feats = (C + 1) // 2` must be even, otherwise the `sin`/`cos` stacks do not
  match up and `torch.stack` raises. Use an even channel count.
- Channels are ordered `y` first, then `x`.
- With `normalize=False` the coordinates are raw pixel indices scaled only by the
  temperature term, so they grow with image size — keep that in mind for variable
  resolutions.

## References

- Carion et al., *End-to-End Object Detection with Transformers* (DETR), the source of the
  2D sine positional encoding — [arXiv:2005.12872](https://arxiv.org/abs/2005.12872) —
  [github.com/facebookresearch/detr](https://github.com/facebookresearch/detr)
- Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at
  Scale* (ViT), for the 1D encoding this one generalizes —
  [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) —
  [github.com/facebookresearch/deit](https://github.com/facebookresearch/deit)
