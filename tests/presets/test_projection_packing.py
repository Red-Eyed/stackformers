"""Preset construction and checkpoint reloading from explicit projection-layout configs."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, assert_never

import pytest
import torch
from torch import Tensor

from stackformers import (
    CrossAttender,
    CrossAttenderConfig,
    CrossAttention,
    CrossAttentionConfig,
    PackedCrossAttention,
    PackedCrossAttentionConfig,
    PackedSelfAttention,
    PackedSelfAttentionConfig,
    PackedSwiGLU,
    PackedSwiGLUConfig,
    SelfAttention,
    SelfAttentionConfig,
    SwiGLU,
    TransformerDecoder,
    TransformerDecoderConfig,
    TransformerEncoder,
    TransformerEncoderConfig,
    VariableWidthTransformerEncoder,
    VariableWidthTransformerEncoderConfig,
    plain_cross_attender_config,
    plain_decoder_config,
    plain_encoder_config,
    variable_width_encoder_config,
)
from stackformers.feedforward.config import SwiGLUConfig
from stackformers.sequence import PaddedInput, make_padded_input
from tests.conftest import atol

if TYPE_CHECKING:
    from pathlib import Path

    from stackformers.norm.config import NormPlacement

PresetConfig: TypeAlias = (
    TransformerEncoderConfig
    | TransformerDecoderConfig
    | CrossAttenderConfig
    | VariableWidthTransformerEncoderConfig
)
Preset: TypeAlias = (
    TransformerEncoder | TransformerDecoder | CrossAttender | VariableWidthTransformerEncoder
)


@pytest.fixture(params=["pre", "post", "sandwich", "reordered"])
def placement(request: pytest.FixtureRequest) -> NormPlacement:
    """Exercise every existing residual/normalization topology."""
    match request.param:
        case "pre" | "post" | "sandwich" | "reordered" as value:
            return value
        case _:
            pytest.fail("unknown normalization placement")


@pytest.fixture(params=["encoder", "decoder", "cross", "variable"])
def config(
    request: pytest.FixtureRequest,
    placement: NormPlacement,
    packed_attention: bool,
    packed_ff: bool,
) -> PresetConfig:
    """Select concrete packed records independently for attention and SwiGLU."""
    match request.param:
        case "encoder":
            config = plain_encoder_config(32, 4, 1, norm_placement=placement)
        case "decoder":
            config = plain_decoder_config(32, 4, 1, norm_placement=placement)
        case "cross":
            config = plain_cross_attender_config(32, 4, 1, norm_placement=placement)
        case "variable":
            config = variable_width_encoder_config(
                d_models=[32, 64], dim_heads=8, norm_placement=placement
            )
        case _:
            pytest.fail("unknown preset")
    match config:
        case TransformerEncoderConfig():
            if packed_attention:
                config.attn = PackedSelfAttentionConfig.model_validate(
                    config.attn.model_dump(exclude={"kind"})
                )
        case TransformerDecoderConfig():
            if packed_attention:
                config.self_attn = PackedSelfAttentionConfig.model_validate(
                    config.self_attn.model_dump(exclude={"kind"})
                )
                config.cross_attn = PackedCrossAttentionConfig.model_validate(
                    config.cross_attn.model_dump(exclude={"kind"})
                )
        case CrossAttenderConfig():
            if packed_attention:
                config.attn = PackedCrossAttentionConfig.model_validate(
                    config.attn.model_dump(exclude={"kind"})
                )
        case VariableWidthTransformerEncoderConfig():
            for layer in config.layers:
                if packed_attention:
                    layer.attn = PackedSelfAttentionConfig.model_validate(
                        layer.attn.model_dump(exclude={"kind"})
                    )
                layer.ff = (
                    PackedSwiGLUConfig(dim=layer.attn.dim)
                    if packed_ff
                    else SwiGLUConfig(dim=layer.attn.dim)
                )
            return config
        case _:
            assert_never(config)
    config.ff = PackedSwiGLUConfig(dim=32) if packed_ff else SwiGLUConfig(dim=32)
    return config


@pytest.fixture
def models(
    config: PresetConfig, device_dtype: tuple[torch.device, torch.dtype], tmp_path: Path
) -> tuple[Preset, Preset]:
    """Reconstruct both the concrete config and the checkpoint without recursive conversion."""
    restored_config = type(config).model_validate_json(config.model_dump_json())
    match config, restored_config:
        case TransformerEncoderConfig(), TransformerEncoderConfig():
            original = TransformerEncoder(config)
            restored = TransformerEncoder(restored_config)
        case TransformerDecoderConfig(), TransformerDecoderConfig():
            original = TransformerDecoder(config)
            restored = TransformerDecoder(restored_config)
        case CrossAttenderConfig(), CrossAttenderConfig():
            original = CrossAttender(config)
            restored = CrossAttender(restored_config)
        case VariableWidthTransformerEncoderConfig(), VariableWidthTransformerEncoderConfig():
            original = VariableWidthTransformerEncoder(config)
            restored = VariableWidthTransformerEncoder(restored_config)
        case _:
            pytest.fail("config roundtrip changed the preset type")
    device, dtype = device_dtype
    original.to(device=device, dtype=dtype).eval()
    restored.to(device=device, dtype=dtype).eval()
    path = tmp_path / "model.pt"
    torch.save(original.state_dict(), path)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    return original, restored


@pytest.fixture
def inputs(device_dtype: tuple[torch.device, torch.dtype]) -> tuple[PaddedInput, PaddedInput]:
    """Use independent dynamic query/context lengths on every supported device and dtype."""
    device, dtype = device_dtype
    query = make_padded_input(
        torch.randn(2, 3, 32, device=device, dtype=dtype),
        torch.ones(2, 3, device=device, dtype=torch.bool),
    )
    context = make_padded_input(
        torch.randn(2, 5, 32, device=device, dtype=dtype),
        torch.ones(2, 5, device=device, dtype=torch.bool),
    )
    return query, context


def _run(model: Preset, query: PaddedInput, context: PaddedInput) -> Tensor:
    """Apply each preset's existing input contract."""
    match model:
        case TransformerEncoder() | VariableWidthTransformerEncoder():
            return model(query)
        case TransformerDecoder() | CrossAttender():
            return model(query, context)
        case _:
            assert_never(model)


@pytest.mark.parametrize(
    ("packed_attention", "packed_ff"), [(False, False), (True, False), (False, True), (True, True)]
)
def test_preset_checkpoint_roundtrip(
    models: tuple[Preset, Preset],
    inputs: tuple[PaddedInput, PaddedInput],
    packed_attention: bool,
    packed_ff: bool,
) -> None:
    """Factories honor independent variant selections and packed checkpoints reload directly."""
    original, restored = models
    for module in restored.modules():
        match module:
            case SelfAttention() | PackedSelfAttention():
                assert type(module) is (PackedSelfAttention if packed_attention else SelfAttention)
            case CrossAttention() | PackedCrossAttention():
                assert type(module) is (
                    PackedCrossAttention if packed_attention else CrossAttention
                )
            case SwiGLU() | PackedSwiGLU():
                assert type(module) is (PackedSwiGLU if packed_ff else SwiGLU)
    query, context = inputs
    torch.testing.assert_close(
        _run(restored, query, context),
        _run(original, query, context),
        atol=atol(query.x.dtype),
        rtol=atol(query.x.dtype),
    )


@pytest.mark.parametrize(("packed_attention", "packed_ff"), [(False, False)])
def test_legacy_preset_json(config: PresetConfig) -> None:
    """Old nested attention payloads without a kind still select ordinary modules."""
    match config:
        case TransformerEncoderConfig() | CrossAttenderConfig():
            payload = config.model_dump_json(exclude={"attn": {"kind"}})
        case TransformerDecoderConfig():
            payload = config.model_dump_json(
                exclude={"self_attn": {"kind"}, "cross_attn": {"kind"}}
            )
        case VariableWidthTransformerEncoderConfig():
            payload = config.model_dump_json(exclude={"layers": {"__all__": {"attn": {"kind"}}}})
        case _:
            assert_never(config)
    restored = type(config).model_validate_json(payload)
    match restored:
        case TransformerEncoderConfig():
            assert type(restored.attn) is SelfAttentionConfig
        case TransformerDecoderConfig():
            assert type(restored.self_attn) is SelfAttentionConfig
            assert type(restored.cross_attn) is CrossAttentionConfig
        case CrossAttenderConfig():
            assert type(restored.attn) is CrossAttentionConfig
        case VariableWidthTransformerEncoderConfig():
            assert all(type(layer.attn) is SelfAttentionConfig for layer in restored.layers)
        case _:
            assert_never(restored)
