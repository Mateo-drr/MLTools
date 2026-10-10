# ConvNeXtV2

2D blocks from the ConvNeXt V2 line of work: the ConvNeXt block and Global Response
Normalization (GRN).

## `ConvNeXtBlock`

`../..`

```python
ConvNeXtBlock(dim: int, layer_scale_init_value: float = 1e-6)
```

Input and output are `[B, dim, H, W]`. The block is the modern ConvNeXt meta block:

1. Depthwise 7x7 convolution (`groups=dim`) that permutes to channel-last.
2. `LayerNorm` over the channel dim, taken from
   [`../..`](../../normalizations/README.md).
3. Inverted bottleneck: `Linear(dim, 4 * dim)` → GELU → `Linear(4 * dim, dim)`.
4. Optional LayerScale: the branch is multiplied by a learnable per-channel `gamma`
   initialized to `layer_scale_init_value`.
5. Residual connection back to the input.

`layer_scale_init_value <= 0` drops LayerScale entirely, which is what the paper does for
the smaller ConvNeXt variants.

```python
import torch
from torch_omni_tool.blocks.ConvNeXtV2.ConvNeXtV2 import ConvNeXtBlock

block = ConvNeXtBlock(dim=64)
y = block(torch.randn(1, 64, 56, 56))  # [1, 64, 56, 56]
```

## `GRN`

`../..`

```python
GRN(dim: int)
```

Global Response Normalization for `[B, C, H, W]`, with `dim == C`:

1. `gx = ||x||_2` over the spatial dims `(H, W)`, kept as `[B, C, 1, 1]`.
2. `nx = gx / (mean_C(gx) + 1e-6)` — the channel norms are normalized against each other.
3. `y = gamma * (x * nx) + beta + x`.

`gamma` and `beta` are per-channel parameters initialized to zero, so a freshly built GRN is
the identity and training starts from a no-op. The block is channel-wise (no cross-channel
mixing) and works in both train and eval mode; the tested behaviour is documented in
[`tests/`](../../../tests/README.md).

```python
import torch
from torch_omni_tool.blocks.ConvNeXtV2.GRN import GRN

grn = GRN(64)
y = grn(torch.randn(2, 64, 56, 56))  # [2, 64, 56, 56]
```

Notes:

- `dim` must equal the channel count and the input must be 4D; a mismatch surfaces as a
  broadcast error from the `[1, dim, 1, 1]` parameters rather than an explicit check.
- The spatial L2 norm keeps gradients, so this is not a `torch.no_grad()` region.
- `gamma` scales multiplicatively, `beta` is a pure additive shift.
- `ConvNeXtBlock` keeps a `# TODO` for stochastic depth (`DropPath`), so `drop_path` is not
  wired up.

## References

- Liu et al., *A ConvNet for the 2020s* (ConvNeXt), [arXiv:2201.03597](https://arxiv.org/abs/2201.03597)
  — [github.com/facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
- Liu et al., *ConvNeXt V2: ConvNet Modality Masking Pretraining*,
  [arXiv:2301.03524](https://arxiv.org/abs/2301.03524)
  — [github.com/facebookresearch/ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)
