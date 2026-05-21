from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch.nn as nn
from pydantic import BaseModel, Field
from torch import Tensor

from stackformers.attention.config import CrossAttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.cross_attender import CrossAttenderLayer, CrossAttenderStack
from stackformers.feedforward.config import FeedForwardConfig, SwiGLUConfig
from stackformers.feedforward.factory import build_ff
from stackformers.feedforward.protocols import FeedForward
from stackformers.norm.config import RMSNormConfig
from stackformers.norm.factory import NormConfig, build_norm
from stackformers.norm.protocols import Norm
from stackformers.positional.config import NoPosEncodingConfig, PosEncodingConfig
from stackformers.positional.factory import build_pos_encoding
from stackformers.positional.protocols import PosEncoding
from stackformers.sequence import SequenceInput

C = TypeVar("C")


class CrossAttenderBase(nn.Module, Generic[C], ABC):
    """Abstract cross-attender: queries from x attend to context, no self-attention.

    Subclass with any config type C.  Implement build_layers and build_norm;
    __init__ wires them into a CrossAttenderStack and nothing else.
    """

    def __init__(self, config: C) -> None:
        super().__init__()
        self.config = config
        self._stack = CrossAttenderStack(
            layers=self.build_layers(config),
            final_norm=self.build_norm(config),
        )

    @abstractmethod
    def build_layers(self, config: C) -> list[CrossAttenderLayer]: ...

    @abstractmethod
    def build_norm(self, config: C) -> Norm: ...

    def forward(self, x_input: SequenceInput, ctx_input: SequenceInput) -> Tensor:
        return self._stack(x_input, ctx_input)


class CrossAttenderConfig(BaseModel):
    attn: CrossAttentionConfig
    ff: FeedForwardConfig
    norm: NormConfig
    pos_encoding: PosEncodingConfig = NoPosEncodingConfig()
    num_layers: int = Field(gt=0)


def plain_cross_attender_config(
    dim: int,
    heads: int,
    num_layers: int,
    *,
    ff_mult: float = 4.0,
    dropout: float = 0.0,
) -> CrossAttenderConfig:
    """Global SDPA cross-attender with RMSNorm and SwiGLU FF, no positional encoding.

    Pass PaddedInput for inference, PackedInput for training — same model.
    """
    dim_head = dim // heads
    return CrossAttenderConfig(
        attn=CrossAttentionConfig(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
        ff=SwiGLUConfig(dim=dim, mult=ff_mult, dropout=dropout),
        norm=RMSNormConfig(dim=dim),
        num_layers=num_layers,
    )


class CrossAttender(CrossAttenderBase[CrossAttenderConfig]):
    """Concrete cross-attender for CrossAttenderConfig.

    Pass PaddedInput for inference, PackedInput for training.
    Override build_pos_encoding, build_ff, or build_norm to customise individual
    collaborators while keeping the rest of the defaults.
    """

    def build_layers(self, config: CrossAttenderConfig) -> list[CrossAttenderLayer]:
        pos = self.build_pos_encoding(config)
        return [
            CrossAttenderLayer(
                cross_attn=CrossAttention(config=config.attn, pos_encoding=pos),
                ff=self.build_ff(config),
                norm_cross=build_norm(config.norm),
                norm_ff=build_norm(config.norm),
            )
            for _ in range(config.num_layers)
        ]

    def build_pos_encoding(self, config: CrossAttenderConfig) -> PosEncoding:
        return build_pos_encoding(config.pos_encoding)

    def build_ff(self, config: CrossAttenderConfig) -> FeedForward:
        return build_ff(config.ff)

    def build_norm(self, config: CrossAttenderConfig) -> Norm:
        return build_norm(config.norm)
