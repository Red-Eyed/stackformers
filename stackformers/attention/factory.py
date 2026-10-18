"""Construct attention implementations and biases from their configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from stackformers.attention.bias import NoAttnBias
from stackformers.attention.config import (
    AttnBiasConfig,
    CrossAttentionConfig,
    DistanceBiasConfig,
    NoAttnBiasConfig,
    PackedCrossAttentionConfig,
    PackedSelfAttentionConfig,
    SelfAttentionConfig,
)
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.distance_bias import RelativeDistanceBias
from stackformers.attention.packed_cross_attn import PackedCrossAttention
from stackformers.attention.packed_self_attn import PackedSelfAttention
from stackformers.attention.self_attn import SelfAttention

if TYPE_CHECKING:
    from stackformers.attention.config import (
        CrossAttentionVariantConfig,
        SelfAttentionVariantConfig,
    )
    from stackformers.attention.protocols import AttnBias, CrossAttn, SelfAttn
    from stackformers.positional.protocols import PosEncoding


def build_attn_bias(config: AttnBiasConfig) -> AttnBias:
    match config:
        case NoAttnBiasConfig():
            return NoAttnBias()
        case DistanceBiasConfig():
            return RelativeDistanceBias(config)
        case _:
            assert_never(config)


def build_self_attn(
    config: SelfAttentionVariantConfig,
    pos_encoding: PosEncoding,
    attn_bias: AttnBias | None = None,
) -> SelfAttn:
    """Construct the configured QKV layout directly, without an intermediate conversion."""
    match config:
        case PackedSelfAttentionConfig():
            return PackedSelfAttention(config, pos_encoding, attn_bias)
        case SelfAttentionConfig():
            return SelfAttention(config, pos_encoding, attn_bias)
        case _:
            assert_never(config)


def build_cross_attn(config: CrossAttentionVariantConfig, pos_encoding: PosEncoding) -> CrossAttn:
    """Construct the configured context KV layout directly."""
    match config:
        case PackedCrossAttentionConfig():
            return PackedCrossAttention(config, pos_encoding)
        case CrossAttentionConfig():
            return CrossAttention(config, pos_encoding)
        case _:
            assert_never(config)
