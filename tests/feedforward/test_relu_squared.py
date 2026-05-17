from __future__ import annotations

import pytest
import torch

from stackformers.feedforward.config import ReluSquaredConfig
from stackformers.feedforward.relu_squared import ReluSquaredFF

B, N, D = 2, 16, 64


@pytest.fixture
def relu_squared(device_dtype: tuple[torch.device, torch.dtype]) -> ReluSquaredFF:
    device, dtype = device_dtype
    return ReluSquaredFF(ReluSquaredConfig(dim=D)).to(device=device, dtype=dtype)


@pytest.fixture
def x(device_dtype: tuple[torch.device, torch.dtype]) -> torch.Tensor:
    device, dtype = device_dtype
    return torch.randn(B, N, D, device=device, dtype=dtype)


def test_relu_squared_output_shape(relu_squared: ReluSquaredFF, x: torch.Tensor) -> None:
    assert relu_squared(x).shape == (B, N, D)


def test_relu_squared_no_bias(relu_squared: ReluSquaredFF) -> None:
    for name, _ in relu_squared.named_parameters():
        assert "bias" not in name, f"Unexpected bias param: {name}"


def test_relu_squared_inner_dim() -> None:
    config = ReluSquaredConfig(dim=64, mult=4.0)
    assert config.inner_dim == int(64 * 4.0)


def test_relu_squared_different_mult(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    ff = ReluSquaredFF(ReluSquaredConfig(dim=D, mult=8.0)).to(device=device, dtype=dtype)
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    assert ff(x).shape == (B, N, D)


def test_relu_squared_gradients_flow(device: torch.device) -> None:
    ff = ReluSquaredFF(ReluSquaredConfig(dim=D)).to(device=device)
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    ff(x).sum().backward()
    assert x.grad is not None
