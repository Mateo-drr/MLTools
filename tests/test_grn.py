"""Tests for the GRN (Global Response Normalization) block."""

import pytest
import torch
from torch.testing import assert_close

from torch_omni_tool.blocks.ConvNeXtV2.GRN import GRN

BATCH, CHANNELS, HEIGHT, WIDTH = 2, 8, 5, 7
EPS = 1e-6


def _reference_grn(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Straightforward re-implementation of the GRN formula for cross-checking."""
    gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
    nx = gx / (gx.mean(dim=1, keepdim=True) + EPS)
    return gamma * (x * nx) + beta + x


def _make_input(
    batch: int = BATCH,
    channels: int = CHANNELS,
    height: int = HEIGHT,
    width: int = WIDTH,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(batch, channels, height, width, generator=generator, dtype=dtype)


@pytest.fixture
def grn() -> GRN:
    return GRN(CHANNELS)


@pytest.mark.parametrize("shape", [(1, 3, 1, 1), (2, 8, 5, 7), (4, 16, 32, 32)])
def test_output_shape_matches_input(shape: tuple[int, int, int, int]) -> None:
    out = GRN(shape[1])(_make_input(*shape))
    assert out.shape == shape


def test_gamma_and_beta_shapes_and_initialization(grn: GRN) -> None:
    names = dict(grn.named_parameters())
    assert set(names) == {"gamma", "beta"}
    assert names["gamma"].shape == (1, CHANNELS, 1, 1)
    assert names["beta"].shape == (1, CHANNELS, 1, 1)
    assert_close(names["gamma"], torch.zeros_like(names["gamma"]))
    assert_close(names["beta"], torch.zeros_like(names["beta"]))


def test_identity_at_initialization(grn: GRN) -> None:
    x = _make_input()
    assert_close(grn(x), x, rtol=0, atol=0)


def test_matches_reference_formula(grn: GRN) -> None:
    torch.manual_seed(1)
    with torch.no_grad():
        grn.gamma.normal_()
        grn.beta.normal_()

    x = _make_input()
    assert_close(grn(x), _reference_grn(x, grn.gamma, grn.beta))


def test_normalization_is_over_spatial_dims_then_channels() -> None:
    x = _make_input()
    gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
    nx = gx / (gx.mean(dim=1, keepdim=True) + EPS)

    # gx holds one L2 norm per (batch, channel) and is broadcast over H, W
    assert gx.shape == (BATCH, CHANNELS, 1, 1)
    assert_close(gx, x.pow(2).sum(dim=(2, 3), keepdim=True).sqrt())
    # the normalized response averages to ~1 across the channel dim
    assert_close(nx.mean(dim=1), torch.ones(BATCH, 1, 1), rtol=1e-4, atol=1e-4)


def test_gamma_scales_and_beta_shifts_per_channel() -> None:
    grn = GRN(CHANNELS)
    x = _make_input()
    gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
    nx = gx / (gx.mean(dim=1, keepdim=True) + EPS)

    gamma = torch.arange(1, CHANNELS + 1, dtype=torch.float32).view(1, -1, 1, 1)
    beta = torch.full((1, CHANNELS, 1, 1), 0.5)
    with torch.no_grad():
        grn.gamma.copy_(gamma)
        grn.beta.copy_(beta)

    assert_close(grn(x), x + gamma * (x * nx) + beta)


def test_beta_is_a_pure_additive_shift() -> None:
    grn = GRN(CHANNELS)
    x = _make_input()
    with torch.no_grad():
        grn.gamma.normal_()
    before = grn(x)
    with torch.no_grad():
        grn.beta.fill_(0.25)
    assert_close(grn(x), before + 0.25)


def test_positive_homogeneity_when_beta_is_zero() -> None:
    grn = GRN(CHANNELS)
    with torch.no_grad():
        grn.gamma.normal_()
    x = _make_input()
    for scale in (0.5, 2.0, 7.0):
        assert_close(grn(x * scale), grn(x) * scale, rtol=1e-5, atol=1e-5)


def test_train_and_eval_modes_agree(grn: GRN) -> None:
    x = _make_input()
    grn.train()
    train_out = grn(x)
    grn.eval()
    assert_close(grn(x), train_out, rtol=0, atol=0)


def test_gradients_flow_to_input_and_parameters(grn: GRN) -> None:
    with torch.no_grad():
        grn.gamma.normal_()
        grn.beta.normal_()

    x = _make_input().requires_grad_(True)
    grn(x).pow(2).mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    assert grn.gamma.grad is not None
    assert grn.beta.grad is not None
    assert torch.isfinite(grn.gamma.grad).all()
    assert torch.isfinite(grn.beta.grad).all()
    assert grn.gamma.grad.abs().sum() > 0
    assert grn.beta.grad.abs().sum() > 0


def test_input_gradient_matches_autograd_reference(grn: GRN) -> None:
    with torch.no_grad():
        grn.gamma.normal_()
        grn.beta.normal_()

    x = _make_input().requires_grad_(True)
    grad_out = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH, generator=torch.Generator().manual_seed(3))

    assert_close(torch.autograd.grad(grn(x), x, grad_out)[0],
                 torch.autograd.grad(_reference_grn(x, grn.gamma, grn.beta), x, grad_out)[0])


def test_parameter_gradients_match_autograd_reference(grn: GRN) -> None:
    with torch.no_grad():
        grn.gamma.normal_()
        grn.beta.normal_()

    x = _make_input()
    grad_out = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH, generator=torch.Generator().manual_seed(4))

    actual = torch.autograd.grad(grn(x), [grn.gamma, grn.beta], grad_out)
    expected = torch.autograd.grad(_reference_grn(x, grn.gamma, grn.beta), [grn.gamma, grn.beta], grad_out)
    assert_close(actual[0], expected[0])
    assert_close(actual[1], expected[1])


def test_beta_receives_constant_gradient(grn: GRN) -> None:
    x = _make_input()
    grn(x).sum().backward()
    # beta is broadcast over H, W, so summing the output gives a gradient of
    # batch * H * W for every channel
    expected = torch.full_like(grn.beta, float(BATCH * HEIGHT * WIDTH))
    assert_close(grn.beta.grad, expected)
    # gamma starts at zero, so d(out)/d(gamma) = x * nx which is generally nonzero
    assert grn.gamma.grad.abs().sum() > 0


def test_all_zero_input_does_not_produce_nan(grn: GRN) -> None:
    with torch.no_grad():
        grn.gamma.fill_(1.0)
    x = torch.zeros(BATCH, CHANNELS, HEIGHT, WIDTH)
    assert torch.isfinite(grn(x)).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_preserved(grn: GRN, dtype: torch.dtype) -> None:
    grn = GRN(CHANNELS).to(dtype)
    x = _make_input(dtype=dtype)
    assert grn(x).dtype == dtype


def test_non_contiguous_input(grn: GRN) -> None:
    x = _make_input(batch=BATCH, channels=CHANNELS, height=HEIGHT, width=WIDTH * 2)
    view = x[..., ::2]
    assert not view.is_contiguous()
    assert_close(grn(view), grn(view.contiguous()))


def test_dim_must_match_channels() -> None:
    grn = GRN(CHANNELS)
    with pytest.raises(RuntimeError):
        grn(_make_input(channels=CHANNELS + 1))


def test_rejects_non_4d_input(grn: GRN) -> None:
    with pytest.raises((IndexError, RuntimeError)):
        grn(torch.randn(BATCH, CHANNELS, HEIGHT))


def test_state_dict_roundtrip() -> None:
    grn = GRN(CHANNELS)
    with torch.no_grad():
        grn.gamma.normal_()
        grn.beta.normal_()

    clone = GRN(CHANNELS)
    clone.load_state_dict(grn.state_dict())
    x = _make_input()
    assert_close(clone(x), grn(x))