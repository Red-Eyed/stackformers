"""Tests for the package-level public component API."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import stackformers
import stackformers as sf
import stackformers.attention as attention
import stackformers.decoder_cache as decoder_cache
import stackformers.feedforward as feedforward
import stackformers.norm as norm
import stackformers.positional as positional
import stackformers.presets as presets
import stackformers.presets.cross_attender as cross_attender
import stackformers.presets.decoder as decoder
import stackformers.presets.encoder as encoder
import stackformers.result as result
import stackformers.sequence as sequence

if TYPE_CHECKING:
    from types import ModuleType

STABLE_4_6_PUBLIC_NAMES = (
    "__version__",
    "PaddedSequence",
    "PackedSequence",
    "SequenceInfo",
    "PaddedInput",
    "PackedInput",
    "SequenceInput",
    "make_padded",
    "make_packed",
    "make_padded_input",
    "make_packed_input",
    "to_seq_info",
    "lengths_to_cu_seqlens",
    "position_ids_from_packed",
    "PosEncoding",
    "SelfAttn",
    "CrossAttn",
    "FeedForward",
    "Norm",
    "EncoderLike",
    "MaskingStrategy",
    "ReconstructionHead",
    "SelfAttentionConfig",
    "CrossAttentionConfig",
    "FeedForwardConfig",
    "LayerConfig",
    "EncoderConfig",
    "DecoderConfig",
    "NormPlacement",
    "RMSNormConfig",
    "LayerNormConfig",
    "NormConfig",
    "RoPE1DConfig",
    "RoPE2DConfig",
    "NoPosEncodingConfig",
    "PosEncodingConfig",
    "NoPosEncoding",
    "RotaryEmbedding1D",
    "RotaryEmbedding2D",
    "SelfAttention",
    "CrossAttention",
    "SwiGLU",
    "build_norm",
    "build_ff",
    "build_pos_encoding",
    "TransformerLayerBase",
    "TransformerLayer",
    "PostNormTransformerLayer",
    "SandwichNormTransformerLayer",
    "ReorderedNormTransformerLayer",
    "Encoder",
    "DecoderLayerBase",
    "DecoderLayer",
    "PostNormDecoderLayer",
    "SandwichNormDecoderLayer",
    "ReorderedNormDecoderLayer",
    "Decoder",
    "CrossAttenderLayerBase",
    "CrossAttenderLayer",
    "PostNormCrossAttenderLayer",
    "SandwichNormCrossAttenderLayer",
    "ReorderedNormCrossAttenderLayer",
    "CrossAttenderStack",
    "TransformerEncoderConfig",
    "TransformerEncoder",
    "plain_encoder_config",
    "windowed_encoder_config",
    "TransformerDecoderConfig",
    "TransformerDecoder",
    "plain_decoder_config",
    "CrossAttenderConfig",
    "CrossAttender",
    "plain_cross_attender_config",
    "MLMWrapperConfig",
    "RandomMasking",
    "RegressionHead",
    "CosineHead",
    "MLMWrapper",
    "MLMOutput",
)

PUBLIC_COMPONENT_NAMES = [
    "PackedTensor",
    "PaddedTensor",
    "AttnBiasConfig",
    "CachedCrossAttentionWrapper",
    "CachedSelfAttentionWrapper",
    "CrossAttentionKVCache",
    "DecoderCrossAttentionCache",
    "DecoderStepOutput",
    "DecoderCrossAttentionCacheBuilder",
    "CachedDecoderWrapper",
    "DistanceBiasConfig",
    "NoAttnBiasConfig",
    "FeedForwardConfig",
    "SwiGLUConfig",
    "HardSwishGLUConfig",
    "GEGLUConfig",
    "GELUConfig",
    "ReluSquaredConfig",
    "SwiGLU",
    "HardSwishGLU",
    "GEGLU",
    "GELUFFN",
    "ReluSquaredFF",
    "LearnedPosEncodingConfig",
    "RoPE2DConfig",
    "RoPENDConfig",
    "YaRNConfig",
    "LearnedPosEncoding",
    "RotaryEmbeddingND",
    "VariableWidthEncoderLayerConfig",
    "node_encoder_config",
]


def test_stable_4_6_public_api_remains_available() -> None:
    """Every symbol published by the latest non-beta release remains importable."""
    missing = set(STABLE_4_6_PUBLIC_NAMES).difference(stackformers.__all__)
    assert not missing
    assert all(hasattr(stackformers, name) for name in STABLE_4_6_PUBLIC_NAMES)


@pytest.mark.parametrize("name", PUBLIC_COMPONENT_NAMES)
def test_top_level_exports_public_component(name: str) -> None:
    """Every configurable component is importable from the package root."""
    assert name in stackformers.__all__
    assert hasattr(stackformers, name)


@pytest.mark.parametrize("name", ["VariableWidthEncoderLayerConfig", "node_encoder_config"])
def test_preset_package_exports_public_component(name: str) -> None:
    """Preset-specific configuration entry points remain available from the subpackage."""
    assert name in presets.__all__
    assert hasattr(presets, name)


@pytest.mark.parametrize("name", sf.__all__)
def test_declared_root_export_resolves(name: str) -> None:
    """Every declared public name is accessible through the sf namespace."""
    assert hasattr(sf, name)


@pytest.mark.parametrize("module", [attention, feedforward, norm, positional, presets, result])
def test_subpackage_exports_available_at_root(module: ModuleType) -> None:
    """Subpackage public APIs retain object identity through the package facade."""
    for name in module.__all__:
        assert name in sf.__all__
        assert getattr(sf, name) is getattr(module, name)


@pytest.mark.parametrize(
    ("module", "names"),
    [
        (
            sequence,
            (
                "packed_batch_size",
                "packed_to_padded",
                "padded_to_key_padding_mask",
                "padded_to_packed",
            ),
        ),
        (encoder, ("TransformerEncoderBase",)),
        (decoder, ("TransformerDecoderBase",)),
        (cross_attender, ("CrossAttenderBase",)),
        (
            decoder_cache,
            (
                "CachedDecoderLayerBase",
                "CachedDecoderLayer",
                "CachedPostNormDecoderLayer",
                "CachedSandwichNormDecoderLayer",
                "CachedReorderedNormDecoderLayer",
            ),
        ),
    ],
)
def test_public_helpers_and_extension_classes(module: ModuleType, names: tuple[str, ...]) -> None:
    """Root imports expose the original helpers and subclassing contracts."""
    for name in names:
        assert name in sf.__all__
        assert getattr(sf, name) is getattr(module, name)
