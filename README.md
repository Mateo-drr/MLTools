# MLTools

A small PyTorch library of reusable network blocks plus a minimal training scaffold
(config, dataloaders, train/eval loops, logging helpers).

Everything is plain `nn.Module` code with type hints and no framework dependencies, so
blocks can be copied into any project or imported from here.

## Installation

```bash
pip install -e ".[dev]"      # runtime + pytest / pylint / mypy / black / types-tqdm
pip install -e ".[wandb]"    # adds the optional Weights & Biases logging dependency
```

Requires Python >= 3.12. `wandb` is optional and only imported when `Config.wb` is enabled,
so installing it is not needed to use the blocks.

## What is inside

| Path | Contents |
| --- | --- |
| [`torch_omni_tools`](torch_omni_tools/blocks/README.md) | Network blocks: ConvNeXt V2, GRN, RepMoDE, RRDB, Squeeze-and-Excite, ViT positional embeddings |
| [`torch_omni_tools`](torch_omni_tools/activations/README.md) | Activation layers: the periodic `Snake` activation |
| [`torch_omni_tools`](torch_omni_tools/normalizations/README.md) | `LayerNorm` for `[b, c, h, w]` and `[b, h, w, c]` |
| [`torch_omni_tools`](torch_omni_tools/datasets/README.md) | `Dataset` / `DataLoader` factory and 3D augmentations |
| `torch_omni_tools` | `Config` dataclass plus a module-level `config` singleton |
| `torch_omni_tools` | `SampleNet`, a placeholder autoencoder-style model |
| `torch_omni_tools` | `run_model`, `train_loop`, `eval_loop` |
| `torch_omni_tools` | Timing, metric formatting and epoch bookkeeping helpers |
| `torch_omni_tools` | Entry point that wires the pieces together |
| [`tests/`](tests/README.md) | Pytest suite (currently the GRN block) |

Each folder has its own README with the details, tensor shapes and references. Every
`__init__.py` is empty, so blocks are imported from their leaf module:

```python
from torch_omni_tools.blocks.SEB.SEB import SEBlock
from torch_omni_tools.blocks.ConvNeXtV2.GRN import GRN
```

Note the inconsistent casing in the paths (`RRDB_2D.py` vs `RRDB_3d.py`) and the typo in
`normalizations/layern_norm_2d.py`.

## Blocks at a glance

| Block | Input | Import |
| --- | --- | --- |
| `ConvNeXtBlock` | `[B, C, H, W]` | `mltools.blocks.ConvNeXtV2.ConvNeXtV2` |
| `GRN` | `[B, C, H, W]` | `mltools.blocks.ConvNeXtV2.GRN` |
| `MoDE`, `MoDEConv2D` | `[B, C, H, W]` | `mltools.blocks.RepMoDE.RepMoDE_2d` |
| `MoDEConv`, encoder/decoder blocks | `[B, C, L]` and `[B, C, D, H, W]` | `mltools.blocks.RepMoDE.RepMoDE_1d` / `RepMoDE_3d` |
| `RRDB`, `RRDB_3D` | `[B, C, H, W]`, `[B, C, D, H, W]` | `mltools.blocks.RRDB.RRDB_2D` / `RRDB_3d` |
| `SEBlock` | `[B, C, H, W]` | `mltools.blocks.SEB.SEB` |
| `PositionEmbeddingSine` | `[B, C, H, W]` | `mltools.blocks.ViT.position_embedding_sine` |
| `Snake` | `[B, C, H, W]` | `mltools.activations.snake` |

```python
import torch
from torch_omni_tools.blocks.ConvNeXtV2.GRN import GRN

y = GRN(64)(torch.randn(2, 64, 56, 56))  # [2, 64, 56, 56]
```

## Configuration and training

`Config` holds every hyperparameter and option of a run; there is no config file or CLI
parsing, so edit the dataclass defaults or override attributes on the `config` singleton.
Print the defaults with:

```bash
python -m torch_omni_tools.config
```

Run the scaffold:

```bash
python -m torch_omni_tools.training
```

The flow is `training.py` → `make_dl` → `SampleNet` → `loops.train_loop` /
`loops.eval_loop` → `utils.finish_epoch`, with `torch.amp.autocast` and `GradScaler` when
`config.half_p` is set. The loss is autoencoder-style (`criterion(samples["data"], output)`),
so the target is the input. The training body sits inside an `if __name__ == "__main__"`
guard, so importing the module has no side effects.

## Development

```bash
pytest             # tests/ (testpaths set in pyproject.toml)
mypy               # strict mode over torch_omni_tools/
pylint torch_omni_tools
```

CI (`.github/workflows/ci.yml`) runs Python 3.12 on Ubuntu: install, pylint, mypy and
pytest. Mypy is clean in strict mode; pylint is marked `continue-on-error` until the
remaining warnings are fixed.

## Known issues

- `RepMoDE_1d.one_hot_task_embedding` and `RepMoDE_3d.one_hot_task_embedding` are module
  level functions that take `self: Any` and cannot be called.
- `MoDEConv(..., causal=True)` in `RepMoDE_1d.py` pads the channel dim instead of the time
  dim.
- `PositionEmbeddingSine` needs an even `(channels + 1) // 2`; odd channel counts raise
  from `torch.stack`.
- `augmentations_3d.random_local_rotation` expects a 4D `[b, 1, H, W]` chunk and a 2D
  label; 5D volumes raise `RuntimeError`.
- `training.py` builds a scheduler but passes `scheduler=None` to `train_loop`, so it never
  steps.
- In `RepMoDE_3d.MoDEConv` (and `RepMoDE_1d.MoDEConv` outside training), `eval()` mode uses
  only the first task's expert weights for the whole batch.
- See the folder READMEs for block specific caveats.

## References

The blocks are re-implementations of published architectures. Original sources:

| Block | Source |
| --- | --- |
| `ConvNeXtBlock` | [ConvNeXt V2: ConvNet Modality Masking Pretraining](https://arxiv.org/abs/2301.03524) — [repo](https://github.com/facebookresearch/ConvNeXt-V2) |
| `GRN`, `LayerNorm` | [ConvNeXt: A ConvNet for the 2020s](https://arxiv.org/abs/2201.03597) — [repo](https://github.com/facebookresearch/ConvNeXt) |
| `MoDE*` (RepMoDE) | [RepMode: Learning to Re-parameterize Diverse Experts for Subcellular Structure Prediction (CVPR 2023)](https://arxiv.org/abs/2212.10066) — [repo](https://github.com/correr-zhou/RepMode) |
| `RRDB`, `RRDB_3D` | [ESRGAN: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1809.00219) — [repo](https://github.com/xinntao/ESRGAN) |
| `SEBlock` | [Squeeze-and-Excitation Networks (CVPR 2018)](https://arxiv.org/abs/1709.01507) |
| `PositionEmbeddingSine` | [End-to-End Object Detection with Transformers (DETR)](https://arxiv.org/abs/2005.12872) — [repo](https://github.com/facebookresearch/detr) |
| `Snake` | [Neural Networks Fail to Learn Periodic Functions and How to Fix It](https://arxiv.org/abs/2006.08195) |
| `random_local_rotation` | [Local Rotation Augmentation for 3D data](https://www.mdpi.com/2313-433X/9/2/46) |
| `cutmix` | [CutMix: Regularized Strategy for Mixup on Image Classification](https://arxiv.org/abs/1905.04899) |

## License

See `LICENSE`.
