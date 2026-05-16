"""stackformers public API."""

from importlib.metadata import version

__version__ = version("stackformers")

from stackformers.attention.config import AttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.kernels import (
    SDPAKernel,
    VarlenSDPAKernel,
    VarlenWindowedSDPAKernel,
    WindowedSDPAKernel,
)
from stackformers.attention.kernels.config import (
    KernelConfig,
    SDPAKernelConfig,
    VarlenSDPAKernelConfig,
    VarlenWindowedSDPAKernelConfig,
    WindowedSDPAKernelConfig,
)
from stackformers.attention.kernels.factory import build_kernel
from stackformers.attention.protocols import AttnKernel, CrossAttn, SelfAttn
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
from stackformers.norm.config import LayerNormConfig, RMSNormConfig
from stackformers.norm.factory import NormConfig, build_norm
from stackformers.norm.protocols import Norm
from stackformers.norm.rms import RMSNorm
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
    packed_cross_attender_config,
    plain_cross_attender_config,
)
from stackformers.presets.decoder import TransformerDecoder, TransformerDecoderConfig
from stackformers.presets.encoder import (
    TransformerEncoder,
    TransformerEncoderConfig,
    packed_encoder_config,
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
    "AttnKernel",
    "SelfAttn",
    "CrossAttn",
    "FeedForward",
    "Norm",
    # configs — attention / ff
    "AttentionConfig",
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
    # norm
    "RMSNorm",
    # positional
    "NoPosEncoding",
    "RotaryEmbedding1D",
    "RotaryEmbedding2D",
    # attention
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
    "CrossAttenderLayer",
    "CrossAttenderStack",
    # presets — builders
    "NormConfig",
    "build_norm",
    "build_ff",
    "build_pos_encoding",
    "KernelConfig",
    "SDPAKernelConfig",
    "WindowedSDPAKernelConfig",
    "VarlenSDPAKernelConfig",
    "VarlenWindowedSDPAKernelConfig",
    "build_kernel",
    # presets
    "TransformerEncoderConfig",
    "TransformerEncoder",
    "plain_encoder_config",
    "windowed_encoder_config",
    "packed_encoder_config",
    "TransformerDecoderConfig",
    "TransformerDecoder",
    "CrossAttenderConfig",
    "CrossAttender",
    "plain_cross_attender_config",
    "packed_cross_attender_config",
]
