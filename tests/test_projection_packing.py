"""Conversion preserves checkpoints, ownership, and parameter trainability."""

from __future__ import annotations

import io

import pytest
import torch

from stackformers.feedforward.config import SwiGLUConfig
from stackformers.feedforward.packed_swiglu import PackedSwiGLU
from stackformers.feedforward.swiglu import SwiGLU


@pytest.fixture
def source(device_dtype: tuple[torch.device, torch.dtype]) -> SwiGLU:
    """Include frozen parameters and a child with a distinct training mode."""
    device, dtype = device_dtype
    module = SwiGLU(SwiGLUConfig(dim=24, mult=4)).to(device=device, dtype=dtype).eval()
    module.w1.requires_grad_(False)
    module.w2.requires_grad_(False)
    module.dropout.train()
    return module


def test_checkpoint_roundtrip(source: SwiGLU) -> None:
    """Packed checkpoints reload and unpack to exact, independently owned weights."""
    packed = source.to_packed()
    buffer = io.BytesIO()
    torch.save(packed.state_dict(), buffer)
    buffer.seek(0)
    loaded = PackedSwiGLU(packed.config).to(source.w1.weight)
    loaded.load_state_dict(torch.load(buffer, weights_only=True))
    for name, value in packed.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
    restored = packed.unpack()
    assert restored.state_dict().keys() == source.state_dict().keys()
    for name, value in source.state_dict().items():
        actual = restored.state_dict()[name]
        torch.testing.assert_close(actual, value, rtol=0, atol=0)
        assert actual.data_ptr() != value.data_ptr()
    assert not packed.training
    assert packed.dropout.training
    assert not packed.gate_value.weight.requires_grad
    for name, parameter in restored.named_parameters():
        assert parameter.requires_grad == source.get_parameter(name).requires_grad
    with torch.no_grad():
        packed.gate_value.weight.add_(1)
    torch.testing.assert_close(restored.w1.weight, source.w1.weight, rtol=0, atol=0)


def test_mixed_trainability_rejected(source: SwiGLU) -> None:
    """A packed parameter cannot represent independently frozen row groups."""
    source.w2.requires_grad_(True)
    with pytest.raises(ValueError, match="requires_grad"):
        source.to_packed()


def test_fresh_optimizer_step(device: torch.device) -> None:
    """New optimizers update packed rows exactly as separate trainable parameters."""
    source = SwiGLU(SwiGLUConfig(dim=24)).to(device)
    packed = source.to_packed()
    ordinary_optimizer = torch.optim.SGD(source.parameters(), lr=0.01)
    packed_optimizer = torch.optim.SGD(packed.parameters(), lr=0.01)
    x = torch.randn(2, 7, 24, device=device)
    source(x).square().sum().backward()
    packed(x).square().sum().backward()
    ordinary_optimizer.step()
    packed_optimizer.step()
    restored = packed.unpack()
    for name, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


def test_tied_projection_values(source: SwiGLU) -> None:
    """Tied source weights convert by value without retaining the original alias."""
    source.w2.weight = source.w1.weight
    packed = source.to_packed()
    restored = packed.unpack()
    torch.testing.assert_close(restored.w1.weight, source.w1.weight, rtol=0, atol=0)
    torch.testing.assert_close(restored.w2.weight, source.w2.weight, rtol=0, atol=0)
    assert restored.w1.weight.data_ptr() != restored.w2.weight.data_ptr()
