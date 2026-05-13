from __future__ import annotations

import torch

from stackformers.v1.configs import FeedForwardConfig
from stackformers.v1.feedforward.swiglu import SwiGLU

B, N, D = 2, 16, 64


def _make_ff(mult: float = 4.0) -> SwiGLU:
    return SwiGLU(FeedForwardConfig(dim=D, mult=mult))


def test_swiglu_output_shape() -> None:
    ff = _make_ff()
    x = torch.randn(B, N, D)
    out = ff(x)
    assert out.shape == (B, N, D)


def test_swiglu_no_bias() -> None:
    ff = _make_ff()
    for name, _ in ff.named_parameters():
        assert "bias" not in name, f"Unexpected bias param: {name}"


def test_swiglu_inner_dim_scaled() -> None:
    """inner_dim should be ~2/3 * dim * mult to match GELU-4x param count."""
    config = FeedForwardConfig(dim=64, mult=4.0)
    expected = int(64 * 4.0 * 2 / 3)
    assert config.inner_dim == expected


def test_swiglu_different_mult() -> None:
    ff = _make_ff(mult=8.0)
    x = torch.randn(B, N, D)
    out = ff(x)
    assert out.shape == (B, N, D)


def test_swiglu_gradients_flow() -> None:
    ff = _make_ff()
    x = torch.randn(B, N, D, requires_grad=True)
    out = ff(x)
    out.sum().backward()
    assert x.grad is not None
