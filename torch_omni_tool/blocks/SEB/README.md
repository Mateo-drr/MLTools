# SEB

Squeeze-and-Excitation channel gating, as introduced by the SENet paper.

## `SEBlock`

`SEB.py`

```python
SEBlock(channels: int, reduce_dim: int = 16)
```

Input and output are `[B, C, H, W]` with `C == channels`. Three steps:

1. **Squeeze** — `AdaptiveAvgPool2d(1)` collapses the spatial dims to `[B, C, 1, 1]`, which
   is flattened to `[B, C]`.
2. **Excite** — `Linear(C, reduce_dim, bias=False)` → `ReLU` → `Linear(reduce_dim, C,
   bias=False)` → `Sigmoid`, producing a per-channel gate in `[0, 1]`.
3. **Scale** — the gate is reshaped to `[B, C, 1, 1]` and multiplied with the input.

```python
import torch
from torch_omni_tool.blocks.SEB.SEB import SEBlock

y = SEBlock(channels=64, reduce_dim=16)(torch.randn(2, 64, 56, 56))  # [2, 64, 56, 56]
```

Notes:

- `reduce_dim` is the bottleneck width; 16 is the paper's default for ResNet-style trunks.
  Keep it positive and smaller than `channels` for the block to make sense as a bottleneck.
- The excitation MLPs are bias-free (as in the reference implementation), so a channel with
  all-zero input passes through unchanged only up to the sigmoid value, which is 0.5 at
  init.
- The block is a drop-in wrapper: it never changes the shape and can be placed after any
  convolution.

## References

- Hu, Shen, Sun, *Squeeze-and-Excitation Networks*, CVPR 2018 —
  [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)
- The block shape also matches `torchvision.ops.SqueezeExcitation` and the timm
  implementation — [github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
