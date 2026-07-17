"""stackformers public API."""

from importlib.metadata import version

__version__ = version("stackformers")

from stackformers.attention.config import CrossAttentionConfig, SelfAttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.protocols import CrossAttn, SelfAttn
from stackformers.attention.self_attn import SelfAttention
from stackformers.config import DecoderConfig, EncoderConfig, LayerConfig
from stackformers.cross_attender import CrossAttenderLayer, CrossAttenderStack
from stackformers.decoder import Decoder, DecoderLayer
from stackformers.encoder import Encoder
from stackformers.feedforward.config import FeedForwardConfig
from stackformers.feedforward.factory import build_ff
from stackformers.feedforward.protocols import FeedForward
from stackformers.feedforward.swiglu import SwiGLU
from stackformers.layers import TransformerLayer
from stackformers.mlm.config import MLMWrapperConfig
from stackformers.mlm.head import RegressionHead
from stackformers.mlm.head_cosine import CosineHead
from stackformers.mlm.masking import RandomMasking
from stackformers.mlm.protocols import EncoderLike, MaskingStrategy, ReconstructionHead
from stackformers.mlm.wrapper import MLMOutput, MLMWrapper
from stackformers.norm.config import LayerNormConfig, RMSNormConfig
from stackformers.norm.factory import NormConfig, build_norm
from stackformers.norm.protocols import Norm
from stackformers.positional.config import (
    NoPosEncodingConfig,
    PosEncodingConfig,
    RoPE1DConfig,
    RoPE2DConfig,
)
from stackformers.positional.factory import build_pos_encoding
from stackformers.positional.none import NoPosEncoding
from stackformers.positional.protocols import PosEncoding
from stackformers.positional.rope1d import RotaryEmbedding1D
from stackformers.positional.rope2d import RotaryEmbedding2D
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
    plain_encoder_config,
    windowed_encoder_config,
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
    "FeedForward",
    "Norm",
    "EncoderLike",
    "MaskingStrategy",
    "ReconstructionHead",
    # configs — attention
    "SelfAttentionConfig",
    "CrossAttentionConfig",
    # configs — ff / layer / encoder / decoder
    "FeedForwardConfig",
    "LayerConfig",
    "EncoderConfig",
    "DecoderConfig",
    # configs — norm
    "RMSNormConfig",
    "LayerNormConfig",
    "NormConfig",
    # configs — positional
    "RoPE1DConfig",
    "RoPE2DConfig",
    "NoPosEncodingConfig",
    "PosEncodingConfig",
    # positional
    "NoPosEncoding",
    "RotaryEmbedding1D",
    "RotaryEmbedding2D",
    # attention
    "SelfAttention",
    "CrossAttention",
    # feedforward
    "SwiGLU",
    # builders
    "build_norm",
    "build_ff",
    "build_pos_encoding",
    # transformer blocks
    "TransformerLayer",
    "Encoder",
    "DecoderLayer",
    "Decoder",
    "CrossAttenderLayer",
    "CrossAttenderStack",
    # presets
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
    # mlm
    "MLMWrapperConfig",
    "RandomMasking",
    "RegressionHead",
    "CosineHead",
    "MLMWrapper",
    "MLMOutput",
]
