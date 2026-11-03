"""Attention rounding resolves immutable geometry before module construction."""

from typing import cast

import pytest
import torch
from pydantic import ValidationError

from stackformers import DimensionRoundingConfig
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
from stackformers.sequence import AllValid, PaddedTensor, make_padded_input

AttentionConfigType = (
    type[SelfAttentionConfig]
    | type[PackedSelfAttentionConfig]
    | type[CrossAttentionConfig]
    | type[PackedCrossAttentionConfig]
)


@pytest.fixture(
    params=[
        SelfAttentionConfig,
        PackedSelfAttentionConfig,
        CrossAttentionConfig,
        PackedCrossAttentionConfig,
    ]
)
def config_type(request: pytest.FixtureRequest) -> AttentionConfigType:
    """Exercise every attention layout and source of keys and values."""
    return cast("AttentionConfigType", request.param)


@pytest.mark.parametrize(("mode", "expected"), [("floor", 64), ("ceil", 128)])
def test_rounded_geometry(config_type: AttentionConfigType, mode: str, expected: int) -> None:
    """Store effective head width while preserving model width and GQA grouping."""
    rounding = DimensionRoundingConfig.model_validate({"multiple_of": 64, "rounding_mode": mode})
    config = config_type(dim=256, heads=4, kv_heads=2, dim_head=70, dimension_rounding=rounding)
    assert config.dim == 256
    assert config.dim_head == expected
    assert config.groups == 2
    assert config_type.model_validate_json(config.model_dump_json()) == config
    assert config_type.model_validate(config).dim_head == expected


def test_default_head_width_is_rounded(config_type: AttentionConfigType) -> None:
    """Omitting dim_head still applies the explicitly chosen rounding policy."""
    config = config_type(dim=512, dimension_rounding=DimensionRoundingConfig(multiple_of=128))
    assert config.dim_head == 128


def test_disabled_rounding(config_type: AttentionConfigType) -> None:
    """An absent policy preserves legacy geometry and diagnostics."""
    with pytest.warns(UserWarning, match="dim_head=70"):
        config = config_type(dim=560, dim_head=70)
    assert config.dim_head == 70


def test_floor_rejects_zero(config_type: AttentionConfigType) -> None:
    """Invalid rounded geometry fails during Pydantic admission."""
    with pytest.raises(ValidationError, match="positive width"):
        config_type(
            dim=512,
            dim_head=32,
            dimension_rounding=DimensionRoundingConfig(multiple_of=64, rounding_mode="floor"),
        )


@pytest.fixture
def tokens(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedTensor:
    """Provide a short valid sequence for each supported device and dtype."""
    device, dtype = device_dtype
    return make_padded_input(torch.randn(2, 3, 256, device=device, dtype=dtype), AllValid())


@pytest.mark.parametrize("mode", ["floor", "ceil"])
def test_rounded_forward(config_type: AttentionConfigType, mode: str, tokens: PaddedTensor) -> None:
    """Rounded geometry constructs the same projections as an explicit head width."""
    rounding = DimensionRoundingConfig.model_validate({"multiple_of": 64, "rounding_mode": mode})
    config = config_type(dim=256, heads=4, kv_heads=2, dim_head=70, dimension_rounding=rounding)
    explicit = config_type(dim=256, heads=4, kv_heads=2, dim_head=config.dim_head)
    match config, explicit:
        case SelfAttentionConfig(), SelfAttentionConfig():
            module = SelfAttention(config, NoPosEncoding())
            reference = SelfAttention(explicit, NoPosEncoding())
        case PackedSelfAttentionConfig(), PackedSelfAttentionConfig():
            module = PackedSelfAttention(config, NoPosEncoding())
            reference = PackedSelfAttention(explicit, NoPosEncoding())
        case CrossAttentionConfig(), CrossAttentionConfig():
            module = CrossAttention(config, NoPosEncoding())
            reference = CrossAttention(explicit, NoPosEncoding())
        case PackedCrossAttentionConfig(), PackedCrossAttentionConfig():
            module = PackedCrossAttention(config, NoPosEncoding())
            reference = PackedCrossAttention(explicit, NoPosEncoding())
        case _:
            pytest.fail("config variants differ")
    module.to(device=tokens.x.device, dtype=tokens.x.dtype)
    reference.to(device=tokens.x.device, dtype=tokens.x.dtype)
    reference.load_state_dict(module.state_dict())
    match module, reference:
        case (
            (CrossAttention() | PackedCrossAttention()),
            (CrossAttention() | PackedCrossAttention()),
        ):
            actual = module(tokens, tokens)
            expected = reference(tokens, tokens)
        case ((SelfAttention() | PackedSelfAttention()), (SelfAttention() | PackedSelfAttention())):
            actual = module(tokens)
            expected = reference(tokens)
        case _:
            pytest.fail("module variants differ")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
