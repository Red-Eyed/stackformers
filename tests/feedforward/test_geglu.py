from __future__ import annotations

import pytest
import torch

from stackformers.feedforward.config import GEGLUConfig
from stackformers.feedforward.geglu import GEGLU

B, N, D = 2, 16, 64


@pytest.fixture
def geglu(device_dtype: tuple[torch.device, torch.dtype]) -> GEGLU:
    device, dtype = device_dtype
    return GEGLU(GEGLUConfig(dim=D)).to(device=device, dtype=dtype)


@pytest.fixture
def x(device_dtype: tuple[torch.device, torch.dtype]) -> torch.Tensor:
    device, dtype = device_dtype
    return torch.randn(B, N, D, device=device, dtype=dtype)


def test_geglu_output_shape(geglu: GEGLU, x: torch.Tensor) -> None:
    assert geglu(x).shape == (B, N, D)


def test_geglu_no_bias(geglu: GEGLU) -> None:
    for name, _ in geglu.named_parameters():
        assert "bias" not in name, f"Unexpected bias param: {name}"


def test_geglu_inner_dim_scaled() -> None:
    config = GEGLUConfig(dim=64, mult=4.0)
    assert config.inner_dim == int(64 * 4.0 * 2 / 3)


def test_geglu_different_mult(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    ff = GEGLU(GEGLUConfig(dim=D, mult=8.0)).to(device=device, dtype=dtype)
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    assert ff(x).shape == (B, N, D)


def test_geglu_gradients_flow(device: torch.device) -> None:
    ff = GEGLU(GEGLUConfig(dim=D)).to(device=device)
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    ff(x).sum().backward()
    assert x.grad is not None


def test_geglu_differs_from_swiglu(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    """GELU and SiLU gates produce different outputs on the same weights."""
    from stackformers.feedforward.config import SwiGLUConfig
    from stackformers.feedforward.swiglu import SwiGLU

    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    g = GEGLU(GEGLUConfig(dim=D)).to(device=device, dtype=dtype)
    s = SwiGLU(SwiGLUConfig(dim=D)).to(device=device, dtype=dtype)
    assert not torch.allclose(g(x), s(x))
