# RepMoDE

Re-implementation of **RepMode** (Zhou et al., CVPR 2023): a Mixture-of-Diverse-Experts
(MoDE) convolution whose expert kernels are *re-parameterized* per task by a gate, so the
trained network keeps the compact topology of a plain convolution while behaving like a
task-specific expert.

One module per dimensionality:

| Module | Classes | Data |
| --- | --- | --- |
| [`RepMoDE_2d.py`](RepMoDE_2d.py) | `MoDE` (wrapper), `MoDEConv2D` | `[B, C, H, W]` |
| [`RepMoDE_1d.py`](RepMoDE_1d.py) | `MoDEConv`, `MoDEEncoderBlock`, `MoDESubNet2Conv`, `MoDEDecoderBlock` | `[B, C, L]`, optional `causal=True` |
| [`RepMoDE_3d.py`](RepMoDE_3d.py) | `MoDEConv`, `MoDEEncoderBlock`, `MoDESubNet2Conv`, `MoDEDecoderBlock` | `[B, C, D, H, W]` |

## How a MoDE convolution works

Instead of routing samples to separate expert sub-networks, every layer stores **five
shared expert kernels** and mixes them per sample and per output channel:

| Expert | Kernel | Diversity |
| --- | --- | --- |
| 0 | 5x5 conv | shape (receptive field) |
| 1 | 3x3 conv, zero-padded to `kernel_size` | shape |
| 2 | 1x1 conv, zero-padded to `kernel_size` | shape |
| 3 | 1x1 conv composed with a 3x3 average-pool kernel | kernel |
| 4 | 1x1 conv composed with a 5x5 average-pool kernel | kernel |

That covers the paper's two diversity axes: *shape* diversity (different receptive fields)
and *kernel* diversity (plain convolutions vs. smoothed ones).

The gate is a single `nn.Linear(num_tasks, num_experts * out_chan)` applied to the one-hot
task encoding, reshaped to `[B, num_experts, out_chan]` and softmaxed over the expert axis.
`routing()` then builds one mixed kernel per sample and per output channel:

```
w[b, o] = sum_e g[b, e, o] * expert_e[o]
```

giving a tensor of shape `[B, out_chan, in_chan, K, K]`, which is applied with a grouped
convolution (`groups=B`) or a per-sample convolution. Everything is differentiable: the
gradient flows through the gate into the shared expert kernels.

Shared building blocks across the three modules:

- `gen_conv_kernel(chans_out, chans_in, k_size)` — Kaiming-initialized expert kernel.
- `gen_avg_pool_kernel(kernel_size)` / `gen_avgpool_kernel` — uniform averaging kernel kept
  in a buffer.
- `trans_kernel(kernel, target_size)` — pads a small expert kernel up to `kernel_size`
  (left-only when `causal=True`).
- `routing(g, batch_size)` — the weighted expert mixture.
- `conv_type="normal"` appends `InstanceNorm + Mish`; `"final"` (or any other value)
  appends `nn.Identity`, which is what the output convolution of a network wants.

## Task encoding

Every block takes a one-hot task tensor `t` of shape `[B, num_tasks]` as its second
argument. In the paper the task id is a prior supplied per sample; here it is up to the
caller to build it, e.g. `F.one_hot(task_id, num_classes=num_tasks).float()`.

`RepMoDE_2d.MoDE` wraps `MoDEConv2D` (fixed at 5 experts, `kernel_size=5`,
`conv_type="final"`) and adds the **global task**: during training, with probability
`global_task_train_prob` (0.2 by default) a sample's task id is replaced by
`global_task_id` (0 by default), which keeps the shared experts useful for samples whose
task is unknown at inference time. `forward` returns the tensor *and* the task ids it
actually used, and `get_task_weights(task_id=None)` returns the gate weights as numpy
arrays of shape `[num_experts, out_chan]`, either for one task or for all of them.

```python
import torch
from mltools.blocks.RepMoDE.RepMoDE_2d import MoDE

block = MoDE(in_chans=3, out_chans=64, num_tasks=2)
y, task_id = block(torch.randn(2, 3, 64, 64), torch.tensor([0, 1]))
weights = block.get_task_weights()            # {task_id: [5, 64]}
```

## Encoder / decoder topology (1d and 3d)

The 1d and 3d modules ship the autoencoder topology from the paper's code:

- `MoDEEncoderBlock(in_chan, out_chan)` — `MoDESubNet2Conv` to widen the channels, then a
  strided `Conv{1,3}d(kernel=2, stride=2)` + norm + Mish to halve the resolution. Returns
  the downsampled tensor and a skip tensor with `out_chan` channels.
- `MoDESubNet2Conv(n_in, n_out)` — two `MoDEConv`s (`kernel_size=5`, `padding="same"`) at
  constant resolution.
- `MoDEDecoderBlock(in_chan, out_chan)` — `ConvTranspose{1,3}d(kernel=2, stride=2)` +
  norm + Mish to double the resolution, concatenate the matching skip, then
  `MoDESubNet2Conv`. `in_chan` is expected to be `2 * out_chan`.

```python
import torch
from mltools.blocks.RepMoDE.RepMoDE_3d import (
    MoDEEncoderBlock,
    MoDESubNet2Conv,
    MoDEDecoderBlock,
)

t = torch.eye(3)[:2]                                    # one-hot task ids [B, num_tasks]
x = torch.randn(2, 4, 16, 16, 16)
y, skip = MoDEEncoderBlock(num_experts=5, num_tasks=3, in_chan=4, out_chan=8)(x, t)
b = MoDESubNet2Conv(num_experts=5, num_tasks=3, n_in=8, n_out=16)(y, t)
d = MoDEDecoderBlock(num_experts=5, num_tasks=3, in_chan=16, out_chan=8)(b, skip, t)
```

`RepMoDE_1d.py` additionally documents a full encoder → bottle → decoder stack in a
"Sample usage" string at the top of the module.

## Caveats

- Class names collide between the 1d and 3d modules (`MoDEConv`, `MoDEEncoderBlock`, ...),
  so alias them when importing both.
- `num_experts` is effectively fixed at 5: `routing()` indexes experts `0..4` explicitly.
- `one_hot_task_embedding` exists as a module level function in both `RepMoDE_1d.py` and
  `RepMoDE_3d.py`, but it takes `self: Any` and uses `self.num_tasks` / `self.device`, so
  it cannot be called as written. Build the one-hot tensor yourself.
- `MoDEConv(..., causal=True)` in `RepMoDE_1d.py` pads the channel dim instead of the time
  dim (`F.pad(x, (pad_left, 0))` is applied after a transpose), so it does not work.
- In `eval()` mode both `MoDEConv` implementations convolve with `w[0]`, i.e. only the first
  task's mixed kernel for the whole batch. In the 1d module the non-causal path only mixes
  per sample while `self.training` is true, so multi-task batches silently collapse at
  inference.
- `MoDEConv2D` has no `causal` option and does the grouped convolution whenever
  `grouped=True` (the default); pass `grouped=False` for a per-sample loop.

## References

- Zhou et al., *RepMode: Learning to Re-parameterize Diverse Experts for Subcellular
  Structure Prediction*, CVPR 2023 — [arXiv:2212.10066](https://arxiv.org/abs/2212.10066)
- Code — [github.com/correr-zhou/RepMode](https://github.com/correr-zhou/RepMode)
