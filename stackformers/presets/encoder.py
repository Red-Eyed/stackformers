from __future__ import annotations

import torch.nn as nn
from pydantic import BaseModel, Field
from torch import Tensor

from stackformers.attention.bias import NoBiasBuilder
from stackformers.attention.config import AttentionConfig
from stackformers.attention.kernels import SDPAKernel
from stackformers.attention.self_attn import SelfAttention
from stackformers.encoder import Encoder
from stackformers.feedforward.config import FeedForwardConfig
from stackformers.layers import TransformerLayer
from stackformers.positional.config import PosEncodingConfig
from stackformers.presets.configs import NormConfig, build_ff, build_norm, build_pos_encoding
from stackformers.sequence import SequenceInput


class TransformerEncoderConfig(BaseModel):
    attn: AttentionConfig
    ff: FeedForwardConfig
    norm: NormConfig
    pos_encoding: PosEncodingConfig
    num_layers: int = Field(gt=0)


class TransformerEncoder(nn.Module):
    """Opinionated encoder preset: norm, ff, pos-encoding, and SDPA kernel from config."""

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
                        bias_builder=NoBiasBuilder(),
                        kernel=SDPAKernel(causal=config.attn.causal, dropout=config.attn.dropout),
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
