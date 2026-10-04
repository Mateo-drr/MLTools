# Activations

Activation layers. Currently a single periodic activation.

## `Snake`

`snake.py`

```python
Snake(channels: int, eps: float = 1e-6)
```

The sine-based periodic activation `snake_a(x) = x + (1 / (a + eps)) * sin(a * x) ** 2`,
applied elementwise on `[B, C, H, W]`. `a` is a learnable per-channel parameter of shape
`[1, C, 1, 1]`, initialized to `1 / 2`, so the frequency of the periodic term is learned
per channel; `eps` keeps the division finite when `a` approaches zero.

Unlike ReLU, the function is periodic *and* monotonic, which is what makes it useful for
regressing periodic signals and for implicit-neural-representation style models.

```python
import torch
from mltools.activations.snake import Snake

y = Snake(64)(torch.randn(2, 64, 56, 56))     # [2, 64, 56, 56]
```

Caveats:

- The parameter shape `[1, C, 1, 1]` means the activation only broadcasts over 4D inputs.
  For `[B, C, L]` audio-style tensors use `SnakeBeta` or reshape the parameter yourself.
- This is the plain `Snake`, not `SnakeBeta` (which adds a learnable magnitude `beta`).

## References

- Ziyin Liu, Tilman Hartwig, Masahito Ueda, *Neural Networks Fail to Learn Periodic
  Functions and How to Fix It* — [arXiv:2006.08195](https://arxiv.org/abs/2006.08195)
- Widely used implementation — NVIDIA BigVGAN,
  [github.com/NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN/blob/main/activations.py)
- PyTorch port used as a reference — [github.com/EdwardDixon/snake](https://github.com/EdwardDixon/snake)
