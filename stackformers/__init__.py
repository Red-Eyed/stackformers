"""stackformers public API."""

from importlib.metadata import version
from typing import TYPE_CHECKING

__version__ = version("stackformers")

from stackformers.all_valid_decoder import (
    AllValidCachedDecoderWrapper,
    AllValidCrossAttentionCacheBuilder,
    AllValidDecoderStepOutput,
)
from stackformers.attention import (
    AttnBias,
    AttnBiasConfig,
    CachedCrossAttentionWrapper,
    CachedSelfAttentionWrapper,
    CrossAttention,
    CrossAttentionConfig,
    CrossAttentionKVCache,
    CrossAttentionVariantConfig,
    CrossAttn,
    DecoderCrossAttentionCache,
    DecoderStepOutput,
    DistanceBiasConfig,
    NoAttnBias,
    NoAttnBiasConfig,
    PackedCrossAttention,
    PackedCrossAttentionConfig,
    PackedSelfAttention,
    PackedSelfAttentionConfig,
    RelativeDistanceBias,
    SelfAttention,
    SelfAttentionConfig,
    SelfAttentionVariantConfig,
    SelfAttn,
    build_attn_bias,
    build_cross_attn,
    build_self_attn,
)
from stackformers.attention.masking import CrossAttentionMask, SelfAttentionMask
from stackformers.config import DecoderConfig, EncoderConfig, LayerConfig
from stackformers.cross_attender import (
    CrossAttenderLayer,
    CrossAttenderLayerBase,
    CrossAttenderStack,
    PostNormCrossAttenderLayer,
    ReorderedNormCrossAttenderLayer,
    SandwichNormCrossAttenderLayer,
)
from stackformers.decoder import (
    Decoder,
    DecoderLayer,
    DecoderLayerBase,
    PostNormDecoderLayer,
    ReorderedNormDecoderLayer,
    SandwichNormDecoderLayer,
)
from stackformers.decoder_cache import (
    CachedDecoderLayer,
    CachedDecoderLayerBase,
    CachedDecoderWrapper,
    CachedPostNormDecoderLayer,
    CachedReorderedNormDecoderLayer,
    CachedSandwichNormDecoderLayer,
    DecoderCrossAttentionCacheBuilder,
)
from stackformers.encoder import Encoder
from stackformers.feedforward import (
    GEGLU,
    GELUFFN,
    FeedForward,
    FeedForwardConfig,
    GEGLUConfig,
    GELUConfig,
    HardSwishGLU,
    HardSwishGLUConfig,
    PackedGEGLU,
    PackedGEGLUConfig,
    PackedHardSwishGLU,
    PackedHardSwishGLUConfig,
    PackedSwiGLU,
    PackedSwiGLUConfig,
    ReluSquaredConfig,
    ReluSquaredFF,
    SwiGLU,
    SwiGLUConfig,
    build_ff,
)
from stackformers.layers import (
    PostNormTransformerLayer,
    ReorderedNormTransformerLayer,
    SandwichNormTransformerLayer,
    TransformerLayer,
    TransformerLayerBase,
)
from stackformers.mlm.config import MLMWrapperConfig
from stackformers.mlm.head import RegressionHead
from stackformers.mlm.head_cosine import CosineHead
from stackformers.mlm.masking import RandomMasking
from stackformers.mlm.protocols import EncoderLike, MaskingStrategy, ReconstructionHead
from stackformers.mlm.wrapper import MLMOutput, MLMWrapper
from stackformers.norm.config import LayerNormConfig, NormPlacement, RMSNormConfig
from stackformers.norm.factory import NormConfig, build_norm
from stackformers.norm.protocols import Norm
from stackformers.packing import to_packed, to_unpacked
from stackformers.positional import (
    LearnedPosEncoding,
    LearnedPosEncodingConfig,
    NoPosEncoding,
    NoPosEncodingConfig,
    PosEncoding,
    PosEncodingConfig,
    RoPE1DConfig,
    RoPE2DConfig,
    RoPENDConfig,
    RotaryEmbedding1D,
    RotaryEmbedding2D,
    RotaryEmbeddingND,
    YaRNConfig,
    build_pos_encoding,
)
from stackformers.preallocated_decoder import (
    PreallocatedCachedDecoderWrapper,
    PreallocatedDecoderCache,
    PreallocatedDecoderStepOutput,
)
from stackformers.presets.cross_attender import (
    CrossAttender,
    CrossAttenderBase,
    CrossAttenderConfig,
    plain_cross_attender_config,
)
from stackformers.presets.decoder import (
    TransformerDecoder,
    TransformerDecoderBase,
    TransformerDecoderConfig,
    plain_decoder_config,
)
from stackformers.presets.encoder import (
    TransformerEncoder,
    TransformerEncoderBase,
    TransformerEncoderConfig,
    node_encoder_config,
    plain_encoder_config,
    windowed_encoder_config,
)
from stackformers.presets.variable_width_encoder import (
    VariableWidthEncoderLayerConfig,
    VariableWidthTransformerEncoder,
    VariableWidthTransformerEncoderConfig,
    variable_width_encoder_config,
)
from stackformers.result import Err, Ok, Result
from stackformers.sequence import (
    AllValid,
    PackedSequence,
    PackedTensor,
    PaddedSequence,
    PaddedTensor,
    SequenceInfo,
    SequenceInput,
    lengths_to_cu_seqlens,
    make_packed,
    make_packed_input,
    make_padded,
    make_padded_input,
    materialize_padding_mask,
    packed_batch_size,
    packed_to_padded,
    padded_to_key_padding_mask,
    padded_to_packed,
    position_ids_from_packed,
    to_seq_info,
)

__all__ = [
    "AllValid",
    "materialize_padding_mask",
    "AllValidCachedDecoderWrapper",
    "AllValidCrossAttentionCacheBuilder",
    "AllValidDecoderStepOutput",
    "PreallocatedCachedDecoderWrapper",
    "PreallocatedDecoderCache",
    "PreallocatedDecoderStepOutput",
    "to_packed",
    "to_unpacked",
    "SelfAttentionMask",
    "CrossAttentionMask",
    "__version__",
    "Result",
    "Ok",
    "Err",
    # sequences
    "PaddedSequence",
    "PackedSequence",
    "SequenceInfo",
    "PaddedTensor",
    "PackedTensor",
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
    "packed_batch_size",
    "packed_to_padded",
    "padded_to_key_padding_mask",
    "padded_to_packed",
    # protocols
    "PosEncoding",
    "SelfAttn",
    "CrossAttn",
    "CachedCrossAttentionWrapper",
    "CachedSelfAttentionWrapper",
    "AttnBias",
    "CrossAttentionKVCache",
    "DecoderCrossAttentionCache",
    "DecoderStepOutput",
    "FeedForward",
    "Norm",
    "EncoderLike",
    "MaskingStrategy",
    "ReconstructionHead",
    # configs — attention
    "SelfAttentionConfig",
    "CrossAttentionConfig",
    "AttnBiasConfig",
    "NoAttnBiasConfig",
    "DistanceBiasConfig",
    # configs — ff / layer / encoder / decoder
    "FeedForwardConfig",
    "SwiGLUConfig",
    "HardSwishGLUConfig",
    "GEGLUConfig",
    "GELUConfig",
    "ReluSquaredConfig",
    "LayerConfig",
    "EncoderConfig",
    "DecoderConfig",
    "NormPlacement",
    # configs — norm
    "RMSNormConfig",
    "LayerNormConfig",
    "NormConfig",
    # configs — positional
    "RoPE1DConfig",
    "RoPE2DConfig",
    "RoPENDConfig",
    "LearnedPosEncodingConfig",
    "YaRNConfig",
    "NoPosEncodingConfig",
    "PosEncodingConfig",
    # positional
    "NoPosEncoding",
    "RotaryEmbedding1D",
    "RotaryEmbedding2D",
    "RotaryEmbeddingND",
    "LearnedPosEncoding",
    # attention
    "PackedSelfAttentionConfig",
    "PackedCrossAttentionConfig",
    "SelfAttentionVariantConfig",
    "CrossAttentionVariantConfig",
    "PackedSelfAttention",
    "PackedCrossAttention",
    "SelfAttention",
    "CrossAttention",
    "NoAttnBias",
    "RelativeDistanceBias",
    # feedforward
    "HardSwishGLU",
    "PackedGEGLU",
    "PackedGEGLUConfig",
    "PackedHardSwishGLU",
    "PackedHardSwishGLUConfig",
    "PackedSwiGLUConfig",
    "PackedSwiGLU",
    "SwiGLU",
    "GEGLU",
    "GELUFFN",
    "ReluSquaredFF",
    # builders
    "build_norm",
    "build_ff",
    "build_pos_encoding",
    "build_attn_bias",
    "build_cross_attn",
    "build_self_attn",
    # transformer blocks
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
    "DecoderCrossAttentionCacheBuilder",
    "CachedDecoderWrapper",
    "CachedDecoderLayerBase",
    "CachedDecoderLayer",
    "CachedPostNormDecoderLayer",
    "CachedSandwichNormDecoderLayer",
    "CachedReorderedNormDecoderLayer",
    "CrossAttenderLayerBase",
    "CrossAttenderLayer",
    "PostNormCrossAttenderLayer",
    "SandwichNormCrossAttenderLayer",
    "ReorderedNormCrossAttenderLayer",
    "CrossAttenderStack",
    # presets
    "TransformerEncoderBase",
    "TransformerDecoderBase",
    "CrossAttenderBase",
    "TransformerEncoderConfig",
    "TransformerEncoder",
    "plain_encoder_config",
    "windowed_encoder_config",
    "node_encoder_config",
    "VariableWidthEncoderLayerConfig",
    "VariableWidthTransformerEncoderConfig",
    "VariableWidthTransformerEncoder",
    "variable_width_encoder_config",
    "TransformerDecoderConfig",
    "TransformerDecoder",
    "plain_decoder_config",
    "CrossAttenderConfig",
    "CrossAttender",
    "plain_cross_attender_config",
    # mlm
    "MLMWrapperConfig",
    "RandomMasking",
    "RegressionHead",
    "CosineHead",
    "MLMWrapper",
    "MLMOutput",
]


if TYPE_CHECKING:
    PackedInput = PackedTensor
    PaddedInput = PaddedTensor


def __getattr__(name: str) -> type[PackedTensor] | type[PaddedTensor]:
    """Resolve legacy input aliases with the sequence module's deprecation guidance."""
    from stackformers import sequence

    if name in ("PackedInput", "PaddedInput"):
        return sequence.__getattr__(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
