"""Packed HardSwishGLU equation, gradients, and dynamic export."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from pydantic import TypeAdapter

from stackformers.feedforward.config import (
    FeedForwardConfig,
    HardSwishGLUConfig,
    PackedHardSwishGLUConfig,
)
from stackformers.feedforward.factory import build_ff
from stackformers.feedforward.hardswish_glu import HardSwishGLU
from stackformers.feedforward.packed_hardswish_glu import PackedHardSwishGLU
from tests.conftest import atol

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def source(device_dtype: tuple[torch.device, torch.dtype]) -> HardSwishGLU:
    """Build a small parameter-matched FFN."""
    device, dtype = device_dtype
    return HardSwishGLU(HardSwishGLUConfig(dim=24)).to(device=device, dtype=dtype).eval()


@pytest.mark.parametrize("shape", [(2, 7, 24), (11, 24)])
def test_packed_forward(source: HardSwishGLU, shape: tuple[int, ...]) -> None:
    """Both padded and flattened token tensors retain numerical behavior."""
    x = torch.randn(shape, device=source.w1.weight.device, dtype=source.w1.weight.dtype)
    packed = source.to_packed()
    torch.testing.assert_close(packed(x), source(x), atol=atol(x.dtype), rtol=atol(x.dtype))


def test_packed_gradients(device: torch.device) -> None:
    """Input and each parameter gradient match the separate representation."""
    source = HardSwishGLU(HardSwishGLUConfig(dim=24)).to(device)
    packed = source.to_packed()
    x = torch.randn(2, 7, 24, device=device, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    source(x).square().sum().backward()
    packed(other).square().sum().backward()
    torch.testing.assert_close(x.grad, other.grad)
    assert packed.gate_value.weight.grad is not None
    gate, value = packed.gate_value.weight.grad.chunk(2)
    torch.testing.assert_close(gate, source.w1.weight.grad)
    torch.testing.assert_close(value, source.w2.weight.grad)
    torch.testing.assert_close(packed.w3.weight.grad, source.w3.weight.grad)


def test_dynamic_export(source: HardSwishGLU) -> None:
    """A nonsingleton trace supports one token and longer sequence lengths."""
    packed = source.to_packed()
    x = torch.randn(2, 3, 24, device=source.w1.weight.device, dtype=source.w1.weight.dtype)
    program = torch.export.export(
        packed, (x,), dynamic_shapes=({1: torch.export.Dim("tokens", min=1)},)
    ).module()
    for length in (1, 2, 7):
        values = x[:, :1].expand(-1, length, -1).contiguous()
        torch.testing.assert_close(program(values), packed(values))


def test_training_dropout(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    """Training retains dropout placement when the same random mask is used."""
    device, dtype = device_dtype
    source = HardSwishGLU(HardSwishGLUConfig(dim=24, dropout=0.25)).to(device=device, dtype=dtype)
    packed = source.to_packed()
    x = torch.randn(2, 7, 24, device=device, dtype=dtype)
    torch.manual_seed(73)
    expected = source(x)
    torch.manual_seed(73)
    actual = packed(x)
    torch.testing.assert_close(actual, expected, atol=atol(dtype), rtol=atol(dtype))


def test_checkpoint_roundtrip(source: HardSwishGLU, tmp_path: Path) -> None:
    """Original weights, packed JSON, and both state-dict layouts round-trip strictly."""
    source.w1.requires_grad_(False)
    source.w2.requires_grad_(False)
    source.dropout.train()
    assert set(source.state_dict()) == {"w1.weight", "w2.weight", "w3.weight"}
    path = tmp_path / "ordinary.pt"
    torch.save(source.state_dict(), path)
    ordinary = HardSwishGLU(source.config).to(source.w1.weight)
    ordinary.load_state_dict(torch.load(path, weights_only=True), strict=True)
    packed = source.to_packed()
    assert type(packed.config) is PackedHardSwishGLUConfig
    config = TypeAdapter(FeedForwardConfig).validate_json(packed.config.model_dump_json())
    assert type(build_ff(config)) is PackedHardSwishGLU
    assert isinstance(config, PackedHardSwishGLUConfig)
    loaded = PackedHardSwishGLU(config).to(source.w1.weight)
    path = tmp_path / "packed.pt"
    torch.save(packed.state_dict(), path)
    loaded.load_state_dict(torch.load(path, weights_only=True), strict=True)
    restored = packed.unpack()
    assert type(restored.config) is HardSwishGLUConfig
    for name, value in ordinary.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
        assert restored.state_dict()[name].data_ptr() != source.state_dict()[name].data_ptr()
    for name, value in packed.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
        assert loaded.state_dict()[name].data_ptr() != value.data_ptr()
    assert not packed.training
    assert packed.dropout.training
    assert not packed.gate_value.weight.requires_grad
    assert not restored.w1.weight.requires_grad
    assert not restored.w2.weight.requires_grad
    with torch.no_grad():
        packed.gate_value.weight.add_(1)
    torch.testing.assert_close(restored.w1.weight, source.w1.weight, rtol=0, atol=0)


def test_mixed_trainability_rejected(source: HardSwishGLU) -> None:
    """Packing rejects projection groups that cannot share one trainability flag."""
    source.w1.requires_grad_(False)
    with pytest.raises(ValueError, match="requires_grad"):
        source.to_packed()


def test_config_snapshot(source: HardSwishGLU) -> None:
    """Mutating a config after construction cannot change conversion geometry."""
    packed = source.to_packed()
    source.config.mult = 1.0
    assert packed.config.mult == 4.0
    config = PackedHardSwishGLUConfig(dim=24)
    direct = PackedHardSwishGLU(config)
    config.mult = 1.0
    assert direct.config.mult == 4.0
