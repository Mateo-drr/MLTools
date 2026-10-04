# Blocks

Reusable `nn.Module` blocks. Each block lives in its own folder next to an empty
`__init__.py`, so imports go through the leaf module:

```python
from mltools.blocks.SEB.SEB import SEBlock
```

## Available blocks

| Folder | Module | Classes | Spatial dims | Original source |
| --- | --- | --- | --- | --- |
| [ConvNeXtV2/](ConvNeXtV2/README.md) | `ConvNeXtV2.py` | `ConvNeXtBlock` | 2D | [ConvNeXt V2](https://arxiv.org/abs/2301.03524) |
| [ConvNeXtV2/](ConvNeXtV2/README.md) | `GRN.py` | `GRN` | 2D | [ConvNeXt V2](https://arxiv.org/abs/2301.03524) |
| [RepMoDE/](RepMoDE/README.md) | `RepMoDE_2d.py` | `MoDE`, `MoDEConv2D` | 2D | [RepMode (CVPR 2023)](https://arxiv.org/abs/2212.10066) |
| [RepMoDE/](RepMoDE/README.md) | `RepMoDE_1d.py` | `MoDEConv`, `MoDEEncoderBlock`, `MoDESubNet2Conv`, `MoDEDecoderBlock` | 1D (causal option) | [RepMode (CVPR 2023)](https://arxiv.org/abs/2212.10066) |
| [RepMoDE/](RepMoDE/README.md) | `RepMoDE_3d.py` | `MoDEConv`, `MoDEEncoderBlock`, `MoDESubNet2Conv`, `MoDEDecoderBlock` | 3D | [RepMode (CVPR 2023)](https://arxiv.org/abs/2212.10066) |
| [RRDB/](RRDB/README.md) | `RRDB_2D.py` | `ResidualDenseBlock_5C`, `RRDB` | 2D | [ESRGAN](https://arxiv.org/abs/1809.00219) |
| [RRDB/](RRDB/README.md) | `RRDB_3d.py` | `ResidualDenseBlock_5C_3D`, `RRDB_3D` | 3D | [ESRGAN](https://arxiv.org/abs/1809.00219) |
| [SEB/](SEB/README.md) | `SEB.py` | `SEBlock` | 2D | [SENet](https://arxiv.org/abs/1709.01507) |
| [ViT/](ViT/README.md) | `position_embedding_sine.py` | `PositionEmbeddingSine` | 2D | [DETR](https://arxiv.org/abs/2005.12872) |

## Things to know

- **Pick the variant that matches your data.** `RepMoDE` and `RRDB` come in 1d / 2d / 3d
  flavours; a 2d block will not accept a 5D volume.
- **Class names collide across variants.** `MoDEConv` exists in both `RepMoDE_1d` and
  `RepMoDE_3d`, so alias them when importing more than one:

  ```python
  from mltools.blocks.RepMoDE.RepMoDE_1d import MoDEConv as MoDEConv1d
  from mltools.blocks.RepMoDE.RepMoDE_3d import MoDEConv as MoDEConv3d
  ```

- **Casing is inconsistent** (`RRDB_2D.py` vs `RRDB_3d.py`), and so is the order of the
  dimension arguments — always check the docstring of the leaf module.
- **`conv_type` / `padding` are strings, not enums.** `conv_type="normal"` adds
  InstanceNorm + Mish, anything else (e.g. `"final"`) skips them; `padding` is forwarded
  straight to `F.conv*d`.
- Nothing here registers itself globally: blocks are plain modules and can be reused across
  models and projects.
