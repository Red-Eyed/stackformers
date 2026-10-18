"""Attention variant dispatch and compatibility of tagged and legacy configuration JSON."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from stackformers.attention.config import (
    CrossAttentionConfig,
    CrossAttentionVariantConfig,
    PackedCrossAttentionConfig,
    PackedSelfAttentionConfig,
    SelfAttentionConfig,
    SelfAttentionVariantConfig,
)
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.factory import build_cross_attn, build_self_attn
from stackformers.attention.packed_cross_attn import PackedCrossAttention
from stackformers.attention.packed_self_attn import PackedSelfAttention
from stackformers.attention.self_attn import SelfAttention
from stackformers.positional.none import NoPosEncoding


@pytest.fixture(params=["separate", "packed"])
def self_config(request: pytest.FixtureRequest) -> SelfAttentionVariantConfig:
    """Provide both QKV layouts with identical geometry."""
    match request.param:
        case "separate":
            return SelfAttentionConfig(dim=32, heads=4, dim_head=8)
        case "packed":
            return PackedSelfAttentionConfig(dim=32, heads=4, dim_head=8)
        case _:
            pytest.fail("unknown self-attention test variant")


@pytest.fixture(params=["separate", "packed"])
def cross_config(request: pytest.FixtureRequest) -> CrossAttentionVariantConfig:
    """Provide both context KV layouts with identical geometry."""
    match request.param:
        case "separate":
            return CrossAttentionConfig(dim=32, heads=4, dim_head=8)
        case "packed":
            return PackedCrossAttentionConfig(dim=32, heads=4, dim_head=8)
        case _:
            pytest.fail("unknown cross-attention test variant")


@pytest.fixture
def position() -> NoPosEncoding:
    """Supply a positional collaborator without parameters."""
    return NoPosEncoding()


def test_self_variant_roundtrip(
    self_config: SelfAttentionVariantConfig, position: NoPosEncoding
) -> None:
    """A saved discriminator reconstructs the selected module directly."""
    restored = TypeAdapter(SelfAttentionVariantConfig).validate_json(self_config.model_dump_json())
    assert type(restored) is type(self_config)
    match self_config:
        case SelfAttentionConfig():
            assert type(build_self_attn(restored, position)) is SelfAttention
        case PackedSelfAttentionConfig():
            assert type(build_self_attn(restored, position)) is PackedSelfAttention


def test_cross_variant_roundtrip(
    cross_config: CrossAttentionVariantConfig, position: NoPosEncoding
) -> None:
    """A saved discriminator distinguishes ordinary and packed context projections."""
    restored = TypeAdapter(CrossAttentionVariantConfig).validate_json(
        cross_config.model_dump_json()
    )
    assert type(restored) is type(cross_config)
    match cross_config:
        case CrossAttentionConfig():
            assert type(build_cross_attn(restored, position)) is CrossAttention
        case PackedCrossAttentionConfig():
            assert type(build_cross_attn(restored, position)) is PackedCrossAttention


def test_legacy_attention_json(position: NoPosEncoding) -> None:
    """Untagged old configs retain separate projection and checkpoint layouts."""
    payload = '{"dim":32,"heads":4,"dim_head":8}'
    self_config = TypeAdapter(SelfAttentionVariantConfig).validate_json(payload)
    cross_config = TypeAdapter(CrossAttentionVariantConfig).validate_json(payload)
    assert type(build_self_attn(self_config, position)) is SelfAttention
    assert type(build_cross_attn(cross_config, position)) is CrossAttention


def test_conversion_config_variants(position: NoPosEncoding) -> None:
    """Checkpoint transitions retain concrete configs for the destination layout."""
    ordinary_self = SelfAttention(SelfAttentionConfig(dim=32, heads=4, dim_head=8), position)
    ordinary_cross = CrossAttention(CrossAttentionConfig(dim=32, heads=4, dim_head=8), position)
    packed_self = ordinary_self.to_packed()
    packed_cross = ordinary_cross.to_packed()
    assert type(packed_self.config) is PackedSelfAttentionConfig
    assert type(packed_cross.config) is PackedCrossAttentionConfig
    assert type(packed_self.unpack().config) is SelfAttentionConfig
    assert type(packed_cross.unpack().config) is CrossAttentionConfig


@pytest.mark.parametrize("payload", ['{"kind":"unknown"}', '{"kind":"packed_swiglu"}'])
def test_unknown_attention_kind(payload: str) -> None:
    """Invalid discriminator values fail instead of falling back to the ordinary layout."""
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        TypeAdapter(SelfAttentionVariantConfig).validate_json(payload)
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        TypeAdapter(CrossAttentionVariantConfig).validate_json(payload)
