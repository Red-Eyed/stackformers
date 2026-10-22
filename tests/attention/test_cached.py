"""Cache adapters admit exact built-in equations instead of guessing subclass behavior."""

from __future__ import annotations

import pytest

from stackformers.attention.bias import NoAttnBias
from stackformers.attention.cache_validation import (
    cacheable_cross_attention,
    cacheable_self_attention,
    supported_self_attention,
)
from stackformers.attention.cached import CachedCrossAttentionWrapper, CachedSelfAttentionWrapper
from stackformers.attention.config import (
    CrossAttentionConfig,
    PackedCrossAttentionConfig,
    PackedSelfAttentionConfig,
    SelfAttentionConfig,
)
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.packed_cross_attn import PackedCrossAttention
from stackformers.attention.packed_self_attn import PackedSelfAttention
from stackformers.attention.self_attn import SelfAttention
from stackformers.positional.none import NoPosEncoding
from stackformers.result import Err


class CustomSelfAttention(SelfAttention):
    """Represent a subclass whose compatibility with the cached equation is unknown."""


class CustomPackedSelfAttention(PackedSelfAttention):
    """Represent custom behavior around a packed QKV projection."""


class CustomCrossAttention(CrossAttention):
    """Represent custom behavior around ordinary cross-attention projections."""


class CustomPackedCrossAttention(PackedCrossAttention):
    """Represent custom behavior around a packed KV projection."""


class CustomNoAttnBias(NoAttnBias):
    """Represent a bias subclass that caching must not assume is a no-op."""


@pytest.fixture(params=[False, True], ids=["separate", "packed"])
def custom_bias_attention(request: pytest.FixtureRequest) -> SelfAttention | PackedSelfAttention:
    """Keep the built-in attention type while customizing its injected bias equation."""
    attention = SelfAttention(
        SelfAttentionConfig(dim=64, heads=1, dim_head=64, causal=True),
        NoPosEncoding(),
        CustomNoAttnBias(),
    )
    return attention.to_packed() if request.param else attention


@pytest.fixture(params=[False, True], ids=["separate", "packed"])
def custom_self_attention(request: pytest.FixtureRequest) -> SelfAttention | PackedSelfAttention:
    """Build a subclass with an otherwise supported global causal configuration."""
    if request.param:
        return CustomPackedSelfAttention(
            PackedSelfAttentionConfig(dim=64, heads=1, dim_head=64, causal=True), NoPosEncoding()
        )
    return CustomSelfAttention(
        SelfAttentionConfig(dim=64, heads=1, dim_head=64, causal=True), NoPosEncoding()
    )


@pytest.fixture(params=[False, True], ids=["separate", "packed"])
def custom_cross_attention(request: pytest.FixtureRequest) -> CrossAttention | PackedCrossAttention:
    """Build a subclass whose projection geometry would otherwise be supported."""
    if request.param:
        return CustomPackedCrossAttention(
            PackedCrossAttentionConfig(dim=64, heads=1, dim_head=64), NoPosEncoding()
        )
    return CustomCrossAttention(CrossAttentionConfig(dim=64, heads=1, dim_head=64), NoPosEncoding())


def test_self_attention_subclass_rejected(
    custom_self_attention: SelfAttention | PackedSelfAttention,
) -> None:
    """Both the Result core and direct public adapter reject unsupported subclasses."""
    outcome = cacheable_self_attention(custom_self_attention)
    assert isinstance(outcome, Err)
    supported = supported_self_attention(custom_self_attention)
    assert isinstance(supported, Err)
    assert supported.error.args == outcome.error.args
    with pytest.raises(TypeError) as caught:
        CachedSelfAttentionWrapper(custom_self_attention)
    assert caught.value.args == outcome.error.args


def test_cross_attention_subclass_rejected(
    custom_cross_attention: CrossAttention | PackedCrossAttention,
) -> None:
    """Cross-cache construction cannot silently replace a subclass's attention equation."""
    outcome = cacheable_cross_attention(custom_cross_attention)
    assert isinstance(outcome, Err)
    with pytest.raises(TypeError) as caught:
        CachedCrossAttentionWrapper(custom_cross_attention)
    assert caught.value.args == outcome.error.args


def test_bias_subclass_rejected(custom_bias_attention: SelfAttention | PackedSelfAttention) -> None:
    """A subclass of the no-op bias cannot silently lose its custom behavior in the cache path."""
    outcome = supported_self_attention(custom_bias_attention)
    assert isinstance(outcome, Err)
    with pytest.raises(NotImplementedError) as caught:
        CachedSelfAttentionWrapper(custom_bias_attention)
    assert caught.value.args == outcome.error.args
