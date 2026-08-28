"""Tests for reusable padded cross-attention context projections."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from stackformers.attention.cached import CachedCrossAttentionWrapper
from stackformers.attention.config import CrossAttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.positional.config import RoPE1DConfig
from stackformers.positional.none import NoPosEncoding
from stackformers.positional.rope1d import RotaryEmbedding1D
from stackformers.sequence import PaddedInput, PaddedSequence, make_padded_input

B, N, S, D, H, DH = 2, 8, 12, 64, 4, 16


@pytest.fixture
def target_context(
    device_dtype: tuple[torch.device, torch.dtype],
) -> tuple[PaddedInput, PaddedInput]:
    """Build padded target and context inputs with non-trivial masks."""
    device, dtype = device_dtype
    target_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    target_mask[0, -2:] = False
    context_mask = torch.ones(B, S, dtype=torch.bool, device=device)
    context_mask[1, -3:] = False
    target = make_padded_input(torch.randn(B, N, D, device=device, dtype=dtype), target_mask)
    context = make_padded_input(
        torch.randn(B, S, D, device=device, dtype=dtype),
        context_mask,
    )
    return target, context


@pytest.mark.parametrize("qk_norm", [False, True])
def test_cached_cross_attention_matches_ordinary_forward(
    target_context: tuple[PaddedInput, PaddedInput],
    device_dtype: tuple[torch.device, torch.dtype],
    qk_norm: bool,
) -> None:
    """Cached and ordinary padded paths implement the same attention equation."""
    device, dtype = device_dtype
    target, context = target_context
    config = CrossAttentionConfig(
        dim=D,
        heads=H,
        dim_head=DH,
        kv_heads=2,
        qk_norm=qk_norm,
    )
    attention = CrossAttention(config, NoPosEncoding()).to(device=device, dtype=dtype).eval()
    cached = CachedCrossAttentionWrapper(attention)

    with torch.no_grad():
        expected = attention(target, context)
        cache = cached.build_cache(context)
        actual = cached(
            target,
            cache,
            PaddedSequence(mask=context.mask),
        )

    torch.testing.assert_close(actual, expected)


def test_cache_retains_kv_heads(
    target_context: tuple[PaddedInput, PaddedInput],
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """GQA caches do not materialize one K/V copy per query head."""
    device, dtype = device_dtype
    _, context = target_context
    config = CrossAttentionConfig(dim=D, heads=H, dim_head=DH, kv_heads=2)
    attention = CrossAttention(config, NoPosEncoding()).to(device=device, dtype=dtype)
    cached = CachedCrossAttentionWrapper(attention)

    cache = cached.build_cache(context)

    assert cache.k.shape == (B, 2, S, DH)
    assert cache.v.shape == (B, 2, S, DH)


def test_cached_calls_do_not_repeat_context_projections(
    target_context: tuple[PaddedInput, PaddedInput],
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """Repeated cached decoding reuses K/V while continuing to project each query."""
    device, dtype = device_dtype
    target, context = target_context
    config = CrossAttentionConfig(dim=D, heads=H, dim_head=DH)
    attention = CrossAttention(config, NoPosEncoding()).to(device=device, dtype=dtype).eval()
    cached = CachedCrossAttentionWrapper(attention)
    q_calls = Mock(return_value=None)
    k_calls = Mock(return_value=None)
    v_calls = Mock(return_value=None)
    attention.to_q.register_forward_hook(q_calls)
    attention.to_k.register_forward_hook(k_calls)
    attention.to_v.register_forward_hook(v_calls)

    with torch.no_grad():
        cache = cached.build_cache(context)
        for _ in range(2):
            cached(
                target,
                cache,
                PaddedSequence(mask=context.mask),
            )

    assert q_calls.call_count == 2
    assert k_calls.call_count == 1
    assert v_calls.call_count == 1


def test_cached_cross_attention_matches_ordinary_forward_with_rope(
    target_context: tuple[PaddedInput, PaddedInput],
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """Context K and repeated Q positioning preserve RoPE attention exactly."""
    device, dtype = device_dtype
    target, context = target_context
    config = CrossAttentionConfig(dim=D, heads=H, dim_head=DH)
    position = RotaryEmbedding1D(RoPE1DConfig(dim_head=DH))
    attention = CrossAttention(config, position).to(device=device, dtype=dtype)
    cached = CachedCrossAttentionWrapper(attention)

    with torch.no_grad():
        expected = attention(target, context)
        cache = cached.build_cache(context)
        actual = cached(target, cache, PaddedSequence(mask=context.mask))

    torch.testing.assert_close(actual, expected)
