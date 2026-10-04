# MLTools

Collection of PyTorch building blocks, base structures and a small training scaffold.

## Installation

```bash
pip install -e ".[dev]"      # runtime + pytest / pylint / mypy / black
```

Requires Python >= 3.12.

## Project layout

| Path | Contents |
| --- | --- |
| `mltools/blocks/` | Reusable network blocks (ConvNeXt, RepMoDE, RRDB, SE, ViT) |
| `mltools/normalizations/` | Normalization layers |
| `mltools/config.py` | `Config` dataclass + `config` singleton |
| `mltools/model.py` | `SampleNet`, a placeholder model |
| `mltools/datasets/` | Dataloader factory and 3D augmentations |
| `mltools/training.py` | Training entry point |
| `mltools/loops.py`, `mltools/utils.py` | Train / eval loops and logging helpers |
| `tests/` | Pytest suite (currently `test_grn.py`) |

Every `__init__.py` is empty, so blocks are imported from their leaf module, e.g.
`from mltools.blocks.SEB.SEB import SEBlock`. Note the casing in the paths
(`RRDB_2D.py` vs `RRDB_3d.py`) and the typo in `normalizations/layern_norm_2d.py`.

## Blocks

### ConvNeXt

`ConvNeXtBlock(dim, layer_scale_init_value=1e-6)` — depthwise 7x7 conv, channel-last
LayerNorm, 4x inverted bottleneck with GELU, optional LayerScale, plus a residual.
Set `layer_scale_init_value <= 0` to drop LayerScale.

```python
import torch
from mltools.blocks.ConvNeXtV2.ConvNeXtV2 import ConvNeXtBlock

block = ConvNeXtBlock(dim=64)
y = block(torch.randn(1, 64, 56, 56))          # [1, 64, 56, 56]
```

### GRN (Global Response Normalization)

`GRN(dim)` for `[B, C, H, W]`. Computes the L2 norm of each channel over `H, W`,
normalizes those norms across `C`, and rescales the input with learnable per-channel
`gamma` / `beta`. Both are zero-initialized, so the block is the identity at init.

```python
from mltools.blocks.ConvNeXtV2.GRN import GRN

y = GRN(64)(torch.randn(2, 64, 56, 56))        # [2, 64, 56, 56]
```

### RepMoDE

Mixture-of-Diverse-Experts convolution, available for 3D, 2D and causal 1D data.
Class names collide across variants (`MoDEConv` exists in both the 1d and 3d
modules), so alias them when importing more than one.

2D — `MoDE` wraps the raw block and adds a "global" task id:

```python
from mltools.blocks.RepMoDE.RepMoDE_2d import MoDE

block = MoDE(in_chans=3, out_chans=64, num_tasks=2)
y, task_id = block(torch.randn(2, 3, 64, 64), torch.tensor([0, 1]))
```

1D / 3D — encoder, "bottle" and decoder blocks used as an autoencoder topology.
The encoder doubles channels and downsamples; decoders take `in_chan == 2 * out_chan`.

```python
from mltools.blocks.RepMoDE.RepMoDE_3d import (
    MoDEEncoderBlock, MoDESubNet2Conv, MoDEDecoderBlock,
)

t = torch.eye(3)[:2]                                    # one-hot task ids [B, num_tasks]
y, skip = MoDEEncoderBlock(num_experts=5, num_tasks=3, in_chan=4, out_chan=8)(x, t)
b = MoDESubNet2Conv(num_experts=5, num_tasks=3, n_in=8, n_out=16)(y, t)
d = MoDEDecoderBlock(num_experts=5, num_tasks=3, in_chan=16, out_chan=8)(b, skip, t)
```

### RRDB (Residual in Residual Dense Block)

ESRGAN-style dense block: 5 convolutions with LeakyReLU, 3 of them stacked and
residual-scaled by 0.2.

```python
from mltools.blocks.RRDB.RRDB_2D import RRDB
from mltools.blocks.RRDB.RRDB_3d import RRDB_3D

y2d = RRDB(nf=64, gc=32)(torch.randn(1, 64, 32, 32))
y3d = RRDB_3D(nf=32, gc=16)(torch.randn(1, 32, 8, 8, 8))
```

### SEB (Squeeze-and-Excite)

`SEBlock(channels, reduce_dim=16)` — global average pooling followed by
Linear-ReLU-Linear-Sigmoid channel gating.

```python
from mltools.blocks.SEB.SEB import SEBlock

y = SEBlock(channels=64, reduce_dim=16)(torch.randn(2, 64, 56, 56))
```

### ViT positional embedding

`PositionEmbeddingSine(temperature=10000, normalize=True, scale=None)` — sine/cosine
2D positional encoding that keeps the input layout `[B, C, H, W]`.

```python
from mltools.blocks.ViT.position_embedding_sine import PositionEmbeddingSine

pos = PositionEmbeddingSine()(torch.randn(2, 128, 56, 56))
```

## Normalizations

### LayerNorm2d

Adapted from ConvNeXt. Handles `chan_first` (`[b, c, h, w]`, channel statistics
computed manually) and `chan_last` (`[b, h, w, c]`, delegated to `F.layer_norm`).

```python
from mltools.normalizations.layern_norm_2d import LayerNorm

norm = LayerNorm(64, eps=1e-6, data_format="chan_first")
y = norm(torch.randn(2, 64, 56, 56))
```

## Datasets and augmentations

`CustomDataset` currently returns synthetic data (`{"data": tensor}`), and
`make_dl(config, split)` wraps it in a `DataLoader` (`split` is `"train"`,
`"valid"` or `"test"`). No files are read from disk yet.

```python
from mltools.config import config
from mltools.datasets.cstm_ds import make_dl

loader = make_dl(config, "train")
samples = next(iter(loader))       # {"data": tensor}
```

3D augmentations, applied to an image chunk and its label mask together:

```python
from mltools.datasets.augmentations_3d import cutmix, random_local_rotation

image, label = random_local_rotation(image, label, radius=16, p=0.5)
images, labels = cutmix(images, labels, mask_size=32)
```

## Configuration

`mltools/config.py` defines the `Config` dataclass and a module-level `config`
instance. Print the defaults with:

```bash
python -m mltools.config
```

Key fields: `lr`, `optimizer` / `criterion` / `scheduler` classes, `grad_clip`,
`num_epochs`, `batch`, `num_workers`, `device`, `half_p` (AMP), `wb` (Weights & Biases),
`project_name`, `modelDir`. There is no config file or CLI parsing — edit the dataclass
defaults or import `config` and override attributes before training.

## Training

```bash
python -m mltools.training
```

Run it from the repository root (or anywhere, as long as the `mltools` package is
importable). `python mltools/training.py` works as well. The training body lives in
`main()` behind an `if __name__ == "__main__"` guard, so importing the module has no
side effects.

The flow is `training.py` → `make_dl` → `SampleNet` → `loops.train_loop` /
`loops.eval_loop` → `utils.finish_epoch`, with `torch.amp.autocast` and
`GradScaler` when `config.half_p` is set. The loss is autoencoder-style
(`criterion(samples["data"], output)`), so the target is the input.

Weights & Biases logging is toggled with `config.wb`, but `wandb` (and `Pillow`,
needed by `augmentations_3d`) are imported unconditionally and are not declared in
`pyproject.toml` — install them manually if you use a clean environment.

## Development

```bash
pytest             # tests/ (testpaths set in pyproject.toml)
pylint mltools
mypy               # strict mode over mltools/
```

CI (`.github/workflows/ci.yml`) runs Python 3.12 on Ubuntu: install, pylint, mypy and
pytest. Mypy runs in strict mode over `mltools/` and is clean; pylint scores 7.6/10 and
is marked `continue-on-error` until the remaining warnings are fixed.

## Known issues

- `RepMoDE_1d.one_hot_task_embedding` references an undefined `self` and cannot be called.
- `MoDEConv(..., causal=True)` in `RepMoDE_1d.py` fails: it pads the channel dim instead of the time dim.
- `PositionEmbeddingSine` requires an even `(channels + 1) // 2`; odd values raise from `torch.stack`.
- `augmentations_3d.random_local_rotation` expects a 4D `[b, 1, H, W]` chunk and a 2D label; 5D volumes raise `RuntimeError`.
- `training.py` builds a scheduler but passes `scheduler=None` to `train_loop`, so it never steps.
- In `RepMoDE_3d.MoDEConv`, `eval()` mode uses only the first task's expert weights for the whole batch.

## License

See `LICENCE.txt`.