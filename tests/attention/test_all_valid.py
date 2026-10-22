"""All-valid attention admission and preservation of generic fully excluded rows."""

from __future__ import annotations

import pytest
import torch

import stackformers as sf
from tests.conftest import atol


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
def input() -> sf.AllValidInput:
    """Supply fixed values and coordinates without constructing a runtime mask."""
    return sf.AllValidInput(
        torch.linspace(-1, 1, 48).reshape(1, 3, 16),
        torch.arange(3, dtype=torch.float32).reshape(1, 3, 1),
    )


def test_all_valid_and_empty_rows(
    attention: sf.SelfAttention | sf.PackedSelfAttention,
    input: sf.AllValidInput,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """The new record matches generic valid attention; generic empty rows still return zero."""
    device, dtype = device_dtype
    attention.to(device=device, dtype=dtype)
    input = sf.AllValidInput(input.x.to(device=device, dtype=dtype), input.abs_positions.to(device))
    mask = torch.ones(1, 3, dtype=torch.bool, device=device)
    padded = sf.PaddedInput(input.x, mask, input.abs_positions)
    torch.testing.assert_close(
        attention(input), attention(padded), atol=atol(dtype), rtol=atol(dtype)
    )
    excluded = padded._replace(mask=torch.zeros_like(mask))
    torch.testing.assert_close(attention(excluded), torch.zeros_like(input.x), atol=0, rtol=0)


def test_reject_pair_masks_and_windows(
    attention: sf.SelfAttention | sf.PackedSelfAttention,
    input: sf.AllValidInput,
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
    input: sf.AllValidInput,
) -> None:
    """A bias collaborator is rejected explicitly, preserving the generic fallback contract."""
    attention.attn_bias = sf.RelativeDistanceBias(sf.DistanceBiasConfig(heads=2, r_max=10))
    with pytest.raises(ValueError, match="NoAttnBias"):
        attention(input)
