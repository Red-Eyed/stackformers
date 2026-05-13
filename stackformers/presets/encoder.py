from __future__ import annotations

import torch.nn as nn
from pydantic import BaseModel, Field
from torch import Tensor

from stackformers.attention.bias_config import BiasBuilderConfig, NoBiasConfig
from stackformers.attention.bias_factory import build_bias_builder
from stackformers.attention.config import AttentionConfig
from stackformers.attention.kernels.config import KernelConfig, SDPAKernelConfig
from stackformers.attention.kernels.factory import build_kernel
from stackformers.attention.self_attn import SelfAttention
from stackformers.encoder import Encoder
from stackformers.feedforward.config import FeedForwardConfig
from stackformers.feedforward.factory import build_ff
from stackformers.layers import TransformerLayer
from stackformers.norm.factory import NormConfig, build_norm
from stackformers.positional.config import PosEncodingConfig
from stackformers.positional.factory import build_pos_encoding
from stackformers.sequence import SequenceInput


class TransformerEncoderConfig(BaseModel):
    attn: AttentionConfig
    ff: FeedForwardConfig
    norm: NormConfig
    pos_encoding: PosEncodingConfig
    kernel: KernelConfig = SDPAKernelConfig()
    bias: BiasBuilderConfig = NoBiasConfig()
    num_layers: int = Field(gt=0)


class TransformerEncoder(nn.Module):
    """Opinionated encoder preset: norm, ff, pos-encoding, kernel, and bias builder from config."""

    def __init__(self, config: TransformerEncoderConfig) -> None:
        super().__init__()
        self.config = config

        pos = build_pos_encoding(config.pos_encoding)

        self._encoder = Encoder(
            layers=[
                TransformerLayer(
                    self_attn=SelfAttention(
                        config=config.attn,
                        pos_encoding=pos,
                        bias_builder=build_bias_builder(
                            config.bias, config.attn.heads, config.attn.causal
                        ),
                        kernel=build_kernel(config.kernel, config.attn.causal, config.attn.dropout),
                    ),
                    ff=build_ff(config.ff),
                    norm_attn=build_norm(config.norm),
                    norm_ff=build_norm(config.norm),
                )
                for _ in range(config.num_layers)
            ],
            final_norm=build_norm(config.norm),
        )

    def forward(self, input: SequenceInput) -> Tensor:
        return self._encoder(input)
