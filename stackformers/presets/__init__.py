from stackformers.attention.bias_config import ALiBiConfig, BiasBuilderConfig, NoBiasConfig
from stackformers.attention.bias_factory import build_bias_builder
from stackformers.attention.kernels.config import (
    KernelConfig,
    SDPAKernelConfig,
    VarlenSDPAKernelConfig,
    VarlenWindowedSDPAKernelConfig,
    WindowedSDPAKernelConfig,
)
from stackformers.attention.kernels.factory import build_kernel
from stackformers.feedforward.factory import build_ff
from stackformers.norm.factory import NormConfig, build_norm
from stackformers.positional.factory import build_pos_encoding
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
    PackedTransformerEncoder,
    TransformerEncoder,
    TransformerEncoderConfig,
    packed_encoder_config,
    plain_encoder_config,
    windowed_encoder_config,
)

__all__ = [
    # builders
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
    "BiasBuilderConfig",
    "NoBiasConfig",
    "ALiBiConfig",
    "build_bias_builder",
    # preset config factories
    "plain_encoder_config",
    "windowed_encoder_config",
    "packed_encoder_config",
    "plain_decoder_config",
    "plain_cross_attender_config",
    # presets
    "TransformerEncoderConfig",
    "TransformerEncoder",
    "PackedTransformerEncoder",
    "TransformerDecoderConfig",
    "TransformerDecoder",
    "CrossAttenderConfig",
    "CrossAttender",
]
