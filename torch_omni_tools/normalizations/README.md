# Normalizations

Normalization layers. Currently a single 2D `LayerNorm` that accepts both channel layouts.

## `LayerNorm`

`layern_norm_2d.py` (note the typo in the file name — it is the import path used by
[`..`](../blocks/ConvNeXtV2/README.md))

```python
LayerNorm(
    normalized_shape: int,
    eps: float = 1e-6,
    data_format: Literal["chan_first", "chan_last"] = "chan_first",
)
```

`normalized_shape` is the number of channels that are normalized together; learnable
`weight` (ones) and `bias` (zeros) are per channel.

- `data_format="chan_last"` — input `[b, h, w, c]`, delegated to `F.layer_norm` over the
  last dimension.
- `data_format="chan_first"` — input `[b, c, h, w]`, statistics computed manually over
  dim 1 and the scale/bias broadcast as `weight[:, None, None]`.

```python
import torch
from torch_omni_tools.normalizations.layern_norm_2d import LayerNorm

norm = LayerNorm(64, eps=1e-6, data_format="chan_first")
y = norm(torch.randn(2, 64, 56, 56))  # [2, 64, 56, 56]
```

An invalid `data_format` raises `NotImplementedError` both at construction and in
`forward`. There are no 1D/3D variants yet; write the statistics over the channel dim and
broadcast the parameters over the remaining spatial dims.

## References

- Ba, Kiros, Hinton, *Layer Normalization* — [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
- The `chan_last` / `chan_first` switch follows the ConvNeXt implementation —
  [arXiv:2201.03597](https://arxiv.org/abs/2201.03597) —
  [github.com/facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
