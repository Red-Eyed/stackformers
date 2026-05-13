"""stackformers public API."""

from stackformers.attention.bias import ALiBiBuilder, NoBiasBuilder
from stackformers.attention.config import AttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.kernels import (
    SDPAKernel,
    VarlenSDPAKernel,
    VarlenWindowedSDPAKernel,
    WindowedSDPAKernel,
)
from stackformers.attention.protocols import AttnBiasBuilder, AttnKernel, CrossAttn, SelfAttn
from stackformers.attention.self_attn import SelfAttention
from stackformers.config import DecoderConfig, EncoderConfig, LayerConfig
from stackformers.decoder import Decoder, DecoderLayer
from stackformers.encoder import Encoder
from stackformers.feedforward.config import FeedForwardConfig
from stackformers.feedforward.protocols import FeedForward
from stackformers.feedforward.swiglu import SwiGLU
from stackformers.layers import TransformerLayer
from stackformers.norm.protocols import Norm
from stackformers.norm.rms import RMSNorm
from stackformers.positional.none import NoPosEncoding
from stackformers.positional.protocols import PackedPosEncoding, PosEncoding
from stackformers.positional.rope1d import RotaryEmbedding1D
from stackformers.positional.rope2d import RotaryEmbedding2D
from stackformers.presets.encoder import TransformerEncoder, TransformerEncoderConfig
from stackformers.presets.encoder_cross import (
    TransformerEncoderCross,
    TransformerEncoderCrossConfig,
)
from stackformers.sequence import (
    PackedSequence,
    PaddedSequence,
    SequenceInfo,
    lengths_to_cu_seqlens,
    make_packed,
    make_padded,
    position_ids_from_packed,
)

__all__ = [
    # sequences
    "PaddedSequence",
    "PackedSequence",
    "SequenceInfo",
    "make_padded",
    "make_packed",
    "lengths_to_cu_seqlens",
    "position_ids_from_packed",
    # protocols
    "PosEncoding",
    "PackedPosEncoding",
    "AttnBiasBuilder",
    "AttnKernel",
    "SelfAttn",
    "CrossAttn",
    "FeedForward",
    "Norm",
    # configs
    "AttentionConfig",
    "FeedForwardConfig",
    "LayerConfig",
    "EncoderConfig",
    "DecoderConfig",
    # norm
    "RMSNorm",
    # positional
    "NoPosEncoding",
    "RotaryEmbedding1D",
    "RotaryEmbedding2D",
    # attention
    "NoBiasBuilder",
    "ALiBiBuilder",
    "SDPAKernel",
    "VarlenSDPAKernel",
    "VarlenWindowedSDPAKernel",
    "WindowedSDPAKernel",
    "SelfAttention",
    "CrossAttention",
    # feedforward
    "SwiGLU",
    # transformer blocks
    "TransformerLayer",
    "Encoder",
    "DecoderLayer",
    "Decoder",
    # presets
    "TransformerEncoderConfig",
    "TransformerEncoder",
    "TransformerEncoderCrossConfig",
    "TransformerEncoderCross",
]
