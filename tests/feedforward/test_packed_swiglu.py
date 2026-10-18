"""Packed SwiGLU equation, gradients, and dynamic export."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import pytest
import torch
from pydantic import BaseModel

from stackformers.feedforward.config import SwiGLUConfig
from stackformers.feedforward.swiglu import SwiGLU
from tests.conftest import atol


@pytest.fixture
def source(device_dtype: tuple[torch.device, torch.dtype]) -> SwiGLU:
    """Build a small parameter-matched FFN."""
    device, dtype = device_dtype
    return SwiGLU(SwiGLUConfig(dim=24)).to(device=device, dtype=dtype).eval()


@pytest.mark.parametrize("shape", [(2, 7, 24), (11, 24)])
def test_packed_forward(source: SwiGLU, shape: tuple[int, ...]) -> None:
    """Both padded and flattened token tensors retain numerical behavior."""
    x = torch.randn(shape, device=source.w1.weight.device, dtype=source.w1.weight.dtype)
    packed = source.to_packed()
    torch.testing.assert_close(packed(x), source(x), atol=atol(x.dtype), rtol=atol(x.dtype))


def test_packed_gradients(device: torch.device) -> None:
    """Input and each parameter gradient match the separate representation."""
    source = SwiGLU(SwiGLUConfig(dim=24)).to(device)
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


def test_dynamic_export(source: SwiGLU) -> None:
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
    source = SwiGLU(SwiGLUConfig(dim=24, dropout=0.25)).to(device=device, dtype=dtype)
    packed = source.to_packed()
    x = torch.randn(2, 7, 24, device=device, dtype=dtype)
    torch.manual_seed(73)
    expected = source(x)
    torch.manual_seed(73)
    actual = packed(x)
    torch.testing.assert_close(actual, expected, atol=atol(dtype), rtol=atol(dtype))


class LegacySwiGLUCheckpoint(BaseModel, frozen=True):
    """Frozen checkpoint geometry, outputs, and gradients captured from the 4.6.0 wheel."""

    version: Literal["4.6.0"]
    dim: int
    mult: float
    state_shapes: dict[str, list[int]]
    output: list[list[list[float]]]
    input_gradient: list[list[list[float]]]


@pytest.fixture
def legacy_reference() -> LegacySwiGLUCheckpoint:
    """Validate the released implementation's observations independently of current code."""
    path = Path(__file__).with_name("legacy_swiglu_4_6.json")
    return LegacySwiGLUCheckpoint.model_validate_json(path.read_text())


@pytest.fixture
def legacy_source(legacy_reference: LegacySwiGLUCheckpoint, tmp_path: Path) -> SwiGLU:
    """Load an old-layout checkpoint through the unchanged ordinary loading contract."""
    state = {
        name: torch.linspace(-0.15, 0.15, math.prod(shape), dtype=torch.float64)
        .reshape(shape)
        .add(index * 0.01)
        for index, (name, shape) in enumerate(sorted(legacy_reference.state_shapes.items()))
    }
    path = tmp_path / "legacy_swiglu.pt"
    torch.save(state, path)
    source = SwiGLU(SwiGLUConfig(dim=legacy_reference.dim, mult=legacy_reference.mult)).double()
    source.load_state_dict(torch.load(path, weights_only=True), strict=True)
    return source


@pytest.mark.parametrize("pack_projections", [False, True])
def test_old_checkpoint_transition(
    legacy_source: SwiGLU,
    legacy_reference: LegacySwiGLUCheckpoint,
    pack_projections: bool,
) -> None:
    """The ordinary → packed → ordinary transition preserves the released equation."""
    x = torch.linspace(-0.8, 0.9, 6, dtype=torch.float64).reshape(1, 2, 3).requires_grad_()
    module = legacy_source.to_packed() if pack_projections else legacy_source
    output = module(x)
    torch.testing.assert_close(
        output, torch.tensor(legacy_reference.output, dtype=torch.float64), rtol=1e-10, atol=1e-12
    )
    output.square().mean().backward()
    torch.testing.assert_close(
        x.grad,
        torch.tensor(legacy_reference.input_gradient, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    )
    restored = legacy_source.to_packed().unpack()
    for name, value in legacy_source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
