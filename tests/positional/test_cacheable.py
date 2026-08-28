"""Positional parity tests for the composed cached cross-attention executor."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from stackformers.attention.cached import CachedCrossAttentionWrapper
from stackformers.attention.config import CrossAttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.positional.config import (
    LearnedPosEncodingConfig,
    NoPosEncodingConfig,
    PosEncodingConfig,
    RoPE1DConfig,
    RoPE2DConfig,
    RoPENDConfig,
)
from stackformers.positional.factory import build_pos_encoding
from stackformers.sequence import PaddedInput, PaddedSequence

DIM = 64
HEADS = 2
DIM_HEAD = DIM // HEADS
CASES: tuple[tuple[PosEncodingConfig, int], ...] = (
    (NoPosEncodingConfig(), 1),
    (LearnedPosEncodingConfig(dim_head=DIM_HEAD, max_seq_len=8), 1),
    (RoPE1DConfig(dim_head=DIM_HEAD), 1),
    (RoPE2DConfig(dim_head=DIM_HEAD), 2),
    (RoPENDConfig(dim_head=DIM_HEAD, coords=2, r_min=0.1, r_max=10.0), 2),
)


def _positions(batch: int, tokens: int, coords: int) -> Tensor:
    """Create valid deterministic coordinates for every built-in encoding."""
    values = torch.arange(tokens, dtype=torch.float32).view(1, tokens, 1)
    return values.expand(batch, tokens, coords).clone()


@pytest.mark.parametrize(("config", "coords"), CASES)
def test_cached_cross_attention_matches_joint_positional_encoding(
    config: PosEncodingConfig,
    coords: int,
) -> None:
    """Zero-token counterpart calls preserve every built-in positional equation."""
    torch.manual_seed(0)
    target = PaddedInput(
        x=torch.randn(2, 3, DIM),
        mask=torch.ones(2, 3, dtype=torch.bool),
        abs_positions=_positions(2, 3, coords),
    )
    context = PaddedInput(
        x=torch.randn(2, 5, DIM),
        mask=torch.ones(2, 5, dtype=torch.bool),
        abs_positions=_positions(2, 5, coords),
    )
    attention = CrossAttention(
        CrossAttentionConfig(dim=DIM, heads=HEADS, dim_head=DIM_HEAD),
        build_pos_encoding(config),
    ).eval()
    cached = CachedCrossAttentionWrapper(attention)

    with torch.no_grad():
        expected = attention(target, context)
        cache = cached.build_cache(context)
        actual = cached(target, cache, PaddedSequence(mask=context.mask))

    torch.testing.assert_close(actual, expected)
