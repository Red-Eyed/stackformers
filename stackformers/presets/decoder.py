from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch.nn as nn
from pydantic import BaseModel, Field
from torch import Tensor

from stackformers.attention.bias import NoAttnBias
from stackformers.attention.config import CrossAttentionConfig, SelfAttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.protocols import AttnBias
from stackformers.attention.self_attn import SelfAttention
from stackformers.decoder import Decoder, DecoderLayer
from stackformers.feedforward.config import FeedForwardConfig, SwiGLUConfig
from stackformers.feedforward.factory import build_ff
from stackformers.feedforward.protocols import FeedForward
from stackformers.norm.config import RMSNormConfig
from stackformers.norm.factory import NormConfig, build_norm
from stackformers.norm.protocols import Norm
from stackformers.positional.config import NoPosEncodingConfig, PosEncodingConfig, RoPE1DConfig
from stackformers.positional.factory import build_pos_encoding
from stackformers.positional.none import NoPosEncoding
from stackformers.positional.protocols import PosEncoding
from stackformers.sequence import SequenceInput

C = TypeVar("C")


class TransformerDecoderBase(nn.Module, Generic[C], ABC):
    """Abstract decoder: causal self-attn → cross-attn → ff per layer.

    Subclass with any config type C.  Implement build_layers and build_norm;
    __init__ wires them into a Decoder and nothing else.
    """

    def __init__(self, config: C) -> None:
        super().__init__()
        self.config = config
        self._decoder = Decoder(
            layers=self.build_layers(config),
            final_norm=self.build_norm(config),
        )

    @abstractmethod
    def build_layers(self, config: C) -> list[DecoderLayer]: ...

    @abstractmethod
    def build_norm(self, config: C) -> Norm: ...

    def forward(self, x_input: SequenceInput, ctx_input: SequenceInput) -> Tensor:
        return self._decoder(x_input, ctx_input)


class TransformerDecoderConfig(BaseModel):
    self_attn: SelfAttentionConfig
    cross_attn: CrossAttentionConfig
    ff: FeedForwardConfig
    norm: NormConfig
    pos_encoding: PosEncodingConfig  # applies to self-attention only
    num_layers: int = Field(gt=0)


def plain_decoder_config(
    dim: int,
    heads: int,
    num_layers: int,
    *,
    ff_mult: float = 4.0,
    dropout: float = 0.0,
) -> TransformerDecoderConfig:
    """Causal self-attn + cross-attn decoder with RoPE-1D, RMSNorm, SwiGLU FF."""
    dim_head = dim // heads
    return TransformerDecoderConfig(
        self_attn=SelfAttentionConfig(
            dim=dim, heads=heads, dim_head=dim_head, causal=True, dropout=dropout
        ),
        cross_attn=CrossAttentionConfig(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
        ff=SwiGLUConfig(dim=dim, mult=ff_mult, dropout=dropout),
        norm=RMSNormConfig(dim=dim),
        pos_encoding=RoPE1DConfig(dim_head=dim_head),
        num_layers=num_layers,
    )


class TransformerDecoder(TransformerDecoderBase[TransformerDecoderConfig]):
    """Concrete decoder for TransformerDecoderConfig.

    Override build_self_pos_encoding, build_self_attn_bias, build_ff, or build_norm
    to customise individual collaborators while keeping the rest of the defaults.
    """

    def build_layers(self, config: TransformerDecoderConfig) -> list[DecoderLayer]:
        self_pos = self.build_self_pos_encoding(config)
        self_bias = self.build_self_attn_bias(config)
        return [
            DecoderLayer(
                self_attn=SelfAttention(
                    config=config.self_attn,
                    pos_encoding=self_pos,
                    attn_bias=self_bias,
                ),
                cross_attn=CrossAttention(
                    config=config.cross_attn,
                    pos_encoding=NoPosEncoding(NoPosEncodingConfig()),
                ),
                ff=self.build_ff(config),
                norm_self=build_norm(config.norm),
                norm_cross=build_norm(config.norm),
                norm_ff=build_norm(config.norm),
            )
            for _ in range(config.num_layers)
        ]

    def build_self_pos_encoding(self, config: TransformerDecoderConfig) -> PosEncoding:
        return build_pos_encoding(config.pos_encoding)

    def build_self_attn_bias(self, _: TransformerDecoderConfig) -> AttnBias:
        return NoAttnBias()

    def build_ff(self, config: TransformerDecoderConfig) -> FeedForward:
        return build_ff(config.ff)

    def build_norm(self, config: TransformerDecoderConfig) -> Norm:
        return build_norm(config.norm)
