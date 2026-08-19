"""Tests for the HardSwish-gated feed-forward network."""

from __future__ import annotations

import pytest
import torch
from pydantic import TypeAdapter

from stackformers.feedforward.config import FeedForwardConfig, HardSwishGLUConfig
from stackformers.feedforward.factory import build_ff
from stackformers.feedforward.hardswish_glu import HardSwishGLU

B, N, D = 2, 16, 64


@pytest.fixture
def hardswish_glu(device_dtype: tuple[torch.device, torch.dtype]) -> HardSwishGLU:
    """Build a HardSwish GLU on the active test device and dtype."""
    device, dtype = device_dtype
    return HardSwishGLU(HardSwishGLUConfig(dim=D)).to(device=device, dtype=dtype)


@pytest.fixture
def x(device_dtype: tuple[torch.device, torch.dtype]) -> torch.Tensor:
    """Create token embeddings on the active test device and dtype."""
    device, dtype = device_dtype
    return torch.randn(B, N, D, device=device, dtype=dtype)


def test_hardswish_glu_output_shape(hardswish_glu: HardSwishGLU, x: torch.Tensor) -> None:
    """The feed-forward transformation preserves token dimensions."""
    assert hardswish_glu(x).shape == (B, N, D)


def test_hardswish_glu_no_bias(hardswish_glu: HardSwishGLU) -> None:
    """All three projections remain bias-free like the other gated variants."""
    assert all("bias" not in name for name, _ in hardswish_glu.named_parameters())


def test_hardswish_glu_uses_parameter_matched_inner_width() -> None:
    """The gated variant retains the shared two-thirds width correction."""
    config = HardSwishGLUConfig(dim=64, mult=4.0)

    assert config.inner_dim == int(64 * 4.0 * 2 / 3)


def test_hardswish_glu_uses_torch_hardswish(hardswish_glu: HardSwishGLU) -> None:
    """The gate is HardSwish rather than silently retaining SwiGLU's SiLU."""
    assert isinstance(hardswish_glu.act, torch.nn.Hardswish)


def test_hardswish_glu_config_round_trip() -> None:
    """The discriminated union restores the new variant from serialized data."""
    adapter = TypeAdapter(FeedForwardConfig)
    config = HardSwishGLUConfig(dim=D, mult=3.0, dropout=0.1)

    restored = adapter.validate_python(config.model_dump())

    assert restored == config


def test_hardswish_glu_gradients_flow(device: torch.device) -> None:
    """Gradients propagate from output tokens through the complete gated branch."""
    ff = HardSwishGLU(HardSwishGLUConfig(dim=D)).to(device=device)
    x = torch.randn(B, N, D, device=device, requires_grad=True)

    ff(x).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_build_ff_constructs_hardswish_glu() -> None:
    """The shared factory dispatches the new discriminated configuration."""
    assert isinstance(build_ff(HardSwishGLUConfig(dim=D)), HardSwishGLU)
