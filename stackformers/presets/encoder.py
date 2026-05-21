from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch.nn as nn
from pydantic import BaseModel, Field
from torch import Tensor

from stackformers.attention.bias import NoAttnBias
from stackformers.attention.config import SelfAttentionConfig
from stackformers.attention.protocols import AttnBias
from stackformers.attention.self_attn import SelfAttention
from stackformers.encoder import Encoder
from stackformers.feedforward.config import FeedForwardConfig, SwiGLUConfig
from stackformers.feedforward.factory import build_ff
from stackformers.feedforward.protocols import FeedForward
from stackformers.layers import TransformerLayer
from stackformers.norm.config import RMSNormConfig
from stackformers.norm.factory import NormConfig, build_norm
from stackformers.norm.protocols import Norm
from stackformers.positional.config import PosEncodingConfig, RoPE1DConfig
from stackformers.positional.factory import build_pos_encoding
from stackformers.positional.protocols import PosEncoding
from stackformers.sequence import SequenceInput

C = TypeVar("C")


class TransformerEncoderBase(nn.Module, Generic[C], ABC):
    """Abstract encoder: self-attn → ff per layer.

    Subclass with any config type C.  Implement build_layers and build_norm;
    __init__ wires them into an Encoder and nothing else.
    """

    def __init__(self, config: C) -> None:
        super().__init__()
        self.config = config
        self._encoder = Encoder(
            layers=self.build_layers(config),
            final_norm=self.build_norm(config),
        )

    @abstractmethod
    def build_layers(self, config: C) -> list[TransformerLayer]: ...

    @abstractmethod
    def build_norm(self, config: C) -> Norm: ...

    def forward(self, input: SequenceInput) -> Tensor:
        return self._encoder(input)


class TransformerEncoderConfig(BaseModel):
    attn: SelfAttentionConfig
    ff: FeedForwardConfig
    norm: NormConfig
    pos_encoding: PosEncodingConfig
    num_layers: int = Field(gt=0)


def plain_encoder_config(
    dim: int,
    heads: int,
    num_layers: int,
    *,
    causal: bool = False,
    ff_mult: float = 4.0,
    dropout: float = 0.0,
) -> TransformerEncoderConfig:
    """Global SDPA encoder with RoPE-1D, RMSNorm, and SwiGLU FF.

    Pass PaddedInput for inference, PackedInput for training — same model.
    """
    dim_head = dim // heads
    return TransformerEncoderConfig(
        attn=SelfAttentionConfig(
            dim=dim, heads=heads, dim_head=dim_head, causal=causal, dropout=dropout
        ),
        ff=SwiGLUConfig(dim=dim, mult=ff_mult, dropout=dropout),
        norm=RMSNormConfig(dim=dim),
        pos_encoding=RoPE1DConfig(dim_head=dim_head),
        num_layers=num_layers,
    )


def windowed_encoder_config(
    dim: int,
    heads: int,
    num_layers: int,
    window_size: int,
    *,
    causal: bool = False,
    ff_mult: float = 4.0,
    dropout: float = 0.0,
) -> TransformerEncoderConfig:
    """Sliding-window encoder — O(n·w) attention for long sequences.

    Pass PaddedInput for inference, PackedInput for training — same model.
    """
    dim_head = dim // heads
    return TransformerEncoderConfig(
        attn=SelfAttentionConfig(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            causal=causal,
            dropout=dropout,
            window_size=window_size,
        ),
        ff=SwiGLUConfig(dim=dim, mult=ff_mult, dropout=dropout),
        norm=RMSNormConfig(dim=dim),
        pos_encoding=RoPE1DConfig(dim_head=dim_head),
        num_layers=num_layers,
    )


class TransformerEncoder(TransformerEncoderBase[TransformerEncoderConfig]):
    """Concrete encoder for TransformerEncoderConfig.

    Pass PaddedInput for inference, PackedInput for training.
    Override build_pos_encoding, build_attn_bias, build_ff, or build_norm to customise
    individual collaborators while keeping the rest of the defaults.
    """

    def build_layers(self, config: TransformerEncoderConfig) -> list[TransformerLayer]:
        pos = self.build_pos_encoding(config)
        bias = self.build_attn_bias(config)
        return [
            TransformerLayer(
                self_attn=SelfAttention(config=config.attn, pos_encoding=pos, attn_bias=bias),
                ff=self.build_ff(config),
                norm_attn=build_norm(config.norm),
                norm_ff=build_norm(config.norm),
            )
            for _ in range(config.num_layers)
        ]

    def build_pos_encoding(self, config: TransformerEncoderConfig) -> PosEncoding:
        return build_pos_encoding(config.pos_encoding)

    def build_attn_bias(self, _: TransformerEncoderConfig) -> AttnBias:
        return NoAttnBias()

    def build_ff(self, config: TransformerEncoderConfig) -> FeedForward:
        return build_ff(config.ff)

    def build_norm(self, config: TransformerEncoderConfig) -> Norm:
        return build_norm(config.norm)
