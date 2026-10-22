"""All-valid attention admission and preservation of generic fully excluded rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from typing_extensions import override

import stackformers as sf
from tests.conftest import atol

if TYPE_CHECKING:
    from pathlib import Path


class MarkerAttention(torch.nn.Module):
    """Expose validity as a root argument to isolate marker serialization from legacy records."""

    def __init__(self, attention: sf.SelfAttn) -> None:
        """Retain the attention module under test without copying its parameters."""
        super().__init__()
        self.attention = attention

    @override
    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | sf.AllValid, positions: torch.Tensor
    ) -> torch.Tensor:
        """Adapt flat arguments to the library's dense record inside the exported graph."""
        return self.attention(sf.PaddedInput(x, mask, positions))


@pytest.fixture(params=[False, True], ids=["ordinary", "packed"])
def attention(request: pytest.FixtureRequest) -> sf.SelfAttention | sf.PackedSelfAttention:
    """Build reproducible global attention with QK normalization and nonzero weights."""
    with torch.random.fork_rng():
        torch.manual_seed(29)
        module = sf.SelfAttention(
            sf.SelfAttentionConfig(dim=16, heads=2, dim_head=8, qk_norm=True),
            sf.RotaryEmbedding1D(sf.RoPE1DConfig(dim_head=8)),
        ).eval()
    return module.to_packed() if request.param else module


@pytest.fixture
def input() -> sf.PaddedInput:
    """Supply fixed values and coordinates without constructing a runtime mask."""
    return sf.PaddedInput(
        torch.linspace(-1, 1, 48).reshape(1, 3, 16),
        sf.AllValid(),
        torch.arange(3, dtype=torch.float32).reshape(1, 3, 1),
    )


@pytest.fixture
def marker_attention(attention: sf.SelfAttention | sf.PackedSelfAttention) -> MarkerAttention:
    """Provide an export wrapper with the marker directly in its input tree."""
    return MarkerAttention(attention)


def test_all_valid_and_empty_rows(
    attention: sf.SelfAttention | sf.PackedSelfAttention,
    input: sf.PaddedInput,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """The new record matches generic valid attention; generic empty rows still return zero."""
    device, dtype = device_dtype
    attention.to(device=device, dtype=dtype)
    input = sf.PaddedInput(
        input.x.to(device=device, dtype=dtype), sf.AllValid(), input.abs_positions.to(device)
    )
    mask = torch.ones(1, 3, dtype=torch.bool, device=device)
    padded = sf.PaddedInput(input.x, mask, input.abs_positions)
    torch.testing.assert_close(
        attention(input), attention(padded), atol=atol(dtype), rtol=atol(dtype)
    )
    excluded = padded._replace(mask=torch.zeros_like(mask))
    torch.testing.assert_close(attention(excluded), torch.zeros_like(input.x), atol=0, rtol=0)


def test_reject_pair_masks_and_windows(
    attention: sf.SelfAttention | sf.PackedSelfAttention,
    input: sf.PaddedInput,
) -> None:
    """All-valid execution cannot silently discard restrictions or fully masked rows."""
    with pytest.raises(ValueError, match="pair masks"):
        attention(input, self_attn_mask=torch.eye(3, dtype=torch.bool))
    match attention:
        case sf.SelfAttention():
            attention.config = attention.config.model_copy(update={"window_size": 2})
        case sf.PackedSelfAttention():
            attention.config = attention.config.model_copy(update={"window_size": 2})
    with pytest.raises(ValueError, match="global attention"):
        attention(input)


def test_reject_learned_bias(
    attention: sf.SelfAttention | sf.PackedSelfAttention,
    input: sf.PaddedInput,
) -> None:
    """A bias collaborator is rejected explicitly, preserving the generic fallback contract."""
    attention.attn_bias = sf.RelativeDistanceBias(sf.DistanceBiasConfig(heads=2, r_max=10))
    with pytest.raises(ValueError, match="NoAttnBias"):
        attention(input)


def test_marker_export_roundtrip(
    marker_attention: MarkerAttention,
    input: sf.PaddedInput,
    tmp_path: Path,
) -> None:
    """The static validity marker survives export serialization without a tensor input."""
    args = (input.x, input.mask, input.abs_positions)
    program = torch.export.export(marker_attention, args)
    path = tmp_path / "attention.pt2"
    torch.export.save(program, path)
    restored = torch.export.load(path).module()
    torch.testing.assert_close(restored(*args), marker_attention(*args))


def test_materialize_only_at_tensor_boundary(input: sf.PaddedInput) -> None:
    """Tensor masks retain identity; explicit markers materialize correctly shaped validity."""
    mask = sf.materialize_padding_mask(input)
    assert mask.dtype == torch.bool
    assert mask.shape == input.x.shape[:2]
    assert mask.all()
    tensor_input = input._replace(mask=mask)
    assert sf.materialize_padding_mask(tensor_input) is mask
    packed = sf.padded_to_packed(input)
    torch.testing.assert_close(packed.x, input.x.flatten(0, 1))
    torch.testing.assert_close(packed.abs_positions, input.abs_positions.flatten(0, 1))
