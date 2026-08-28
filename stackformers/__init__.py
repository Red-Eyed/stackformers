"""stackformers public API."""

from importlib.metadata import version

__version__ = version("stackformers")

from stackformers.attention import (
    AttnBias,
    AttnBiasConfig,
    CachedCrossAttentionWrapper,
    CachedSelfAttentionWrapper,
    CrossAttention,
    CrossAttentionConfig,
    CrossAttentionKVCache,
    CrossAttn,
    DecoderCrossAttentionCache,
    DecoderStepOutput,
    DistanceBiasConfig,
    NoAttnBias,
    NoAttnBiasConfig,
    RelativeDistanceBias,
    SelfAttention,
    SelfAttentionConfig,
    SelfAttn,
    build_attn_bias,
)
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
from stackformers.decoder_cache import CachedDecoderWrapper, DecoderCrossAttentionCacheBuilder
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
from stackformers.presets.cross_attender import (
    CrossAttender,
    CrossAttenderConfig,
    plain_cross_attender_config,
)
from stackformers.presets.decoder import (
    TransformerDecoder,
    TransformerDecoderConfig,
    plain_decoder_config,
)
from stackformers.presets.encoder import (
    TransformerEncoder,
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
from stackformers.sequence import (
    PackedInput,
    PackedSequence,
    PaddedInput,
    PaddedSequence,
    SequenceInfo,
    SequenceInput,
    lengths_to_cu_seqlens,
    make_packed,
    make_packed_input,
    make_padded,
    make_padded_input,
    position_ids_from_packed,
    to_seq_info,
)

__all__ = [
    "__version__",
    # sequences
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
    "SelfAttention",
    "CrossAttention",
    "NoAttnBias",
    "RelativeDistanceBias",
    # feedforward
    "HardSwishGLU",
    "SwiGLU",
    "GEGLU",
    "GELUFFN",
    "ReluSquaredFF",
    # builders
    "build_norm",
    "build_ff",
    "build_pos_encoding",
    "build_attn_bias",
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
    "CrossAttenderLayerBase",
    "CrossAttenderLayer",
    "PostNormCrossAttenderLayer",
    "SandwichNormCrossAttenderLayer",
    "ReorderedNormCrossAttenderLayer",
    "CrossAttenderStack",
    # presets
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
