"""stackformers v1 public API."""

from stackformers.v1.attention.bias import ALiBiBuilder, NoBiasBuilder
from stackformers.v1.attention.config import AttentionConfig
from stackformers.v1.attention.cross_attn import CrossAttention
from stackformers.v1.attention.kernels import (
    SDPAKernel,
    VarlenSDPAKernel,
    VarlenWindowedSDPAKernel,
    WindowedSDPAKernel,
)
from stackformers.v1.attention.protocols import AttnBiasBuilder, AttnKernel, CrossAttn, SelfAttn
from stackformers.v1.attention.self_attn import SelfAttention
from stackformers.v1.config import DecoderConfig, EncoderConfig, LayerConfig
from stackformers.v1.decoder import Decoder, DecoderLayer
from stackformers.v1.encoder import Encoder
from stackformers.v1.factories import build_decoder, build_encoder, build_gpt
from stackformers.v1.feedforward.config import FeedForwardConfig
from stackformers.v1.feedforward.protocols import FeedForward
from stackformers.v1.feedforward.swiglu import SwiGLU
from stackformers.v1.layers import TransformerLayer
from stackformers.v1.norm.protocols import Norm
from stackformers.v1.norm.rms import RMSNorm
from stackformers.v1.positional.none import NoPosEncoding
from stackformers.v1.positional.protocols import PackedPosEncoding, PosEncoding
from stackformers.v1.positional.rope1d import RotaryEmbedding1D
from stackformers.v1.positional.rope2d import RotaryEmbedding2D
from stackformers.v1.sequence import (
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
    # transformer
    "TransformerLayer",
    "Encoder",
    "DecoderLayer",
    "Decoder",
    # factories
    "build_encoder",
    "build_decoder",
    "build_gpt",
]
