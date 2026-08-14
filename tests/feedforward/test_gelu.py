"""Tests for the standard GELU feed-forward network."""

from __future__ import annotations

import pytest
import torch
from pydantic import TypeAdapter

from stackformers.feedforward.config import FeedForwardConfig, GELUConfig
from stackformers.feedforward.factory import build_ff
from stackformers.feedforward.gelu import GELUFFN

B, N, D = 2, 16, 64


@pytest.fixture
def gelu_ffn(device_dtype: tuple[torch.device, torch.dtype]) -> GELUFFN:
    """Build a GELU FFN on the active test device and dtype."""
    device, dtype = device_dtype
    return GELUFFN(GELUConfig(dim=D)).to(device=device, dtype=dtype)


@pytest.fixture
def x(device_dtype: tuple[torch.device, torch.dtype]) -> torch.Tensor:
    """Create token embeddings on the active test device and dtype."""
    device, dtype = device_dtype
    return torch.randn(B, N, D, device=device, dtype=dtype)


def test_gelu_output_shape(gelu_ffn: GELUFFN, x: torch.Tensor) -> None:
    """The FFN preserves batch, sequence, and model dimensions."""
    assert gelu_ffn(x).shape == (B, N, D)


def test_gelu_no_bias(gelu_ffn: GELUFFN) -> None:
    """Both projections remain bias-free like the other FFN variants."""
    for name, _ in gelu_ffn.named_parameters():
        assert "bias" not in name, f"Unexpected bias param: {name}"


def test_gelu_inner_dim() -> None:
    """The non-gated FFN uses the full configured width multiplier."""
    config = GELUConfig(dim=64, mult=4.0)
    assert config.inner_dim == 256


def test_gelu_uses_tanh_approximation(gelu_ffn: GELUFFN) -> None:
    """The activation uses the repository's established GELU approximation."""
    assert gelu_ffn.act.approximate == "tanh"


def test_gelu_config_round_trip() -> None:
    """The discriminated union restores GELU configs from serialized data."""
    adapter = TypeAdapter(FeedForwardConfig)
    config = GELUConfig(dim=D, mult=3.0, dropout=0.1)

    restored = adapter.validate_python(config.model_dump())

    assert restored == config


def test_gelu_gradients_flow(device: torch.device) -> None:
    """Gradients propagate from the output to input embeddings."""
    ff = GELUFFN(GELUConfig(dim=D)).to(device=device)
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    ff(x).sum().backward()
    assert x.grad is not None


def test_build_ff_constructs_gelu() -> None:
    """The feed-forward factory dispatches GELU configs to GELU FFNs."""
    assert isinstance(build_ff(GELUConfig(dim=D)), GELUFFN)
