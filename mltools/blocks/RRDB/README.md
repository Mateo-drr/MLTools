# RRDB

Residual-in-Residual Dense Blocks from ESRGAN, in 2D and 3D. Useful as the trunk of
super-resolution or restoration networks, and as a plain feature extractor.

| Module | Classes | Input |
| --- | --- | --- |
| [`RRDB_2D.py`](RRDB_2D.py) | `ResidualDenseBlock_5C`, `RRDB` | `[B, C, H, W]` |
| [`RRDB_3d.py`](RRDB_3d.py) | `ResidualDenseBlock_5C_3D`, `RRDB_3D` | `[B, C, D, H, W]` |

## `ResidualDenseBlock_5C` / `ResidualDenseBlock_5C_3D`

```python
ResidualDenseBlock_5C(nf: int = 64, gc: int = 32, bias: bool = True)
ResidualDenseBlock_5C_3D(nf: int = 64, gc: int = 32, bias: bool = True)
```

Five 3x3x3 convolutions with dense (concatenating) skip connections and `LeakyReLU(0.2)`
between them. `nf` is the number of feature channels, `gc` the growth channels produced by
each convolution. Convolutions 1-4 output `gc` channels and see the concatenation of the
input with every previous output; convolution 5 maps back to `nf` channels. The result is
residual-scaled and added to the input:

```
x5 = conv5(cat(x, x1, x2, x3, x4))
out = 0.2 * x5 + x
```

The input and output shapes match, so blocks stack freely.

## `RRDB` / `RRDB_3D`

```python
RRDB(nf: int, gc: int = 32)
RRDB_3D(nf: int, gc: int = 32)
```

Three residual dense blocks in sequence, each with its own residual connection and the same
`0.2` scaling, plus a residual connection around the whole stack:

```python
import torch
from mltools.blocks.RRDB.RRDB_2D import RRDB
from mltools.blocks.RRDB.RRDB_3d import RRDB_3D

y2d = RRDB(nf=64, gc=32)(torch.randn(1, 64, 32, 32))       # [1, 64, 32, 32]
y3d = RRDB_3D(nf=32, gc=16)(torch.randn(1, 32, 8, 8, 8))    # [1, 32, 8, 8, 8]
```

## Caveats

- All convolutions are `kernel=3, stride=1, padding=1`, so spatial size is preserved and
  `nf` must stay constant across a stack.
- `nf` has no default on `RRDB` / `RRDB_3D` (only on the inner dense blocks) — pass it
  explicitly.
- The 3D variant needs a 5D input; the 2D one a 4D input.
- These are the building blocks only: no upsampling, no pixel-shuffle, no discriminator.

## References

- Wang et al., *ESRGAN: Photo-Realistic Single Image Super-Resolution Using a Generative
  Adversarial Network* — [arXiv:1809.00219](https://arxiv.org/abs/1809.00219)
- Code — [github.com/xinntao/ESRGAN](https://github.com/xinntao/ESRGAN)
