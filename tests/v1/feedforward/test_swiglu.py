from __future__ import annotations

import pytest
import torch

from stackformers.v1.feedforward.config import FeedForwardConfig
from stackformers.v1.feedforward.swiglu import SwiGLU

B, N, D = 2, 16, 64


@pytest.fixture
def swiglu(device_dtype: tuple[torch.device, torch.dtype]) -> SwiGLU:
    device, dtype = device_dtype
    return SwiGLU(FeedForwardConfig(dim=D)).to(device=device, dtype=dtype)


@pytest.fixture
def x(device_dtype: tuple[torch.device, torch.dtype]) -> torch.Tensor:
    device, dtype = device_dtype
    return torch.randn(B, N, D, device=device, dtype=dtype)


def test_swiglu_output_shape(swiglu: SwiGLU, x: torch.Tensor) -> None:
    assert swiglu(x).shape == (B, N, D)


def test_swiglu_no_bias(swiglu: SwiGLU) -> None:
    for name, _ in swiglu.named_parameters():
        assert "bias" not in name, f"Unexpected bias param: {name}"


def test_swiglu_inner_dim_scaled() -> None:
    config = FeedForwardConfig(dim=64, mult=4.0)
    assert config.inner_dim == int(64 * 4.0 * 2 / 3)


def test_swiglu_different_mult(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    ff = SwiGLU(FeedForwardConfig(dim=D, mult=8.0)).to(device=device, dtype=dtype)
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    assert ff(x).shape == (B, N, D)


def test_swiglu_gradients_flow(device: torch.device) -> None:
    ff = SwiGLU(FeedForwardConfig(dim=D)).to(device=device)
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    ff(x).sum().backward()
    assert x.grad is not None
