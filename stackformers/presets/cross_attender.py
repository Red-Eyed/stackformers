"""Configurable cross-attender presets and their component builders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import torch.nn as nn
from pydantic import BaseModel, Field
from torch import Tensor
from typing_extensions import override

from stackformers.attention.config import CrossAttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.cross_attender import (
    CrossAttenderLayer,
    CrossAttenderLayerBase,
    CrossAttenderStack,
    PostNormCrossAttenderLayer,
    ReorderedNormCrossAttenderLayer,
    SandwichNormCrossAttenderLayer,
)
from stackformers.feedforward.config import FeedForwardConfig, SwiGLUConfig
from stackformers.feedforward.factory import build_ff
from stackformers.norm.config import NormPlacement, RMSNormConfig
from stackformers.norm.factory import NormConfig, build_norm
from stackformers.positional.config import NoPosEncodingConfig, PosEncodingConfig
from stackformers.positional.factory import build_pos_encoding

if TYPE_CHECKING:
    from collections.abc import Sequence

    from stackformers.feedforward.protocols import FeedForward
    from stackformers.norm.protocols import Norm
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
    def build_layers(self, config: C) -> Sequence[CrossAttenderLayerBase]: ...

    @abstractmethod
    def build_norm(self, config: C) -> Norm: ...

    @override
    def forward(self, x_input: SequenceInput, ctx_input: SequenceInput) -> Tensor:
        return self._stack(x_input, ctx_input)

    if TYPE_CHECKING:
        __call__ = forward


class CrossAttenderConfig(BaseModel):
    attn: CrossAttentionConfig
    ff: FeedForwardConfig
    norm: NormConfig
    pos_encoding: PosEncodingConfig = NoPosEncodingConfig()
    num_layers: int = Field(gt=0)
    norm_placement: NormPlacement = "pre"


def plain_cross_attender_config(
    dim: int,
    heads: int,
    num_layers: int,
    *,
    ff_mult: float = 4.0,
    dropout: float = 0.0,
    norm_placement: NormPlacement = "pre",
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
        norm_placement=norm_placement,
    )


class CrossAttender(CrossAttenderBase[CrossAttenderConfig]):
    """Concrete cross-attender for CrossAttenderConfig.

    Pass PaddedInput for inference, PackedInput for training.
    Override build_pos_encoding, build_ff, or build_norm to customise individual
    collaborators while keeping the rest of the defaults.
    """

    def build_layers(self, config: CrossAttenderConfig) -> list[CrossAttenderLayerBase]:
        pos = self.build_pos_encoding(config)
        return [self._build_layer(config, pos) for _ in range(config.num_layers)]

    def _build_layer(
        self,
        config: CrossAttenderConfig,
        pos: PosEncoding,
    ) -> CrossAttenderLayerBase:
        """Construct the cross-attender layer selected by the norm topology."""
        cross_attn = CrossAttention(config=config.attn, pos_encoding=pos)
        ff = self.build_ff(config)
        match config.norm_placement:
            case "pre":
                return CrossAttenderLayer(
                    cross_attn,
                    ff,
                    build_norm(config.norm),
                    build_norm(config.norm),
                )
            case "post":
                return PostNormCrossAttenderLayer(
                    cross_attn,
                    ff,
                    build_norm(config.norm),
                    build_norm(config.norm),
                )
            case "sandwich":
                return SandwichNormCrossAttenderLayer(
                    cross_attn,
                    ff,
                    build_norm(config.norm),
                    build_norm(config.norm),
                    build_norm(config.norm),
                    build_norm(config.norm),
                )
            case "reordered":
                return ReorderedNormCrossAttenderLayer(
                    cross_attn,
                    ff,
                    build_norm(config.norm),
                    build_norm(config.norm),
                )

    def build_pos_encoding(self, config: CrossAttenderConfig) -> PosEncoding:
        return build_pos_encoding(config.pos_encoding)

    def build_ff(self, config: CrossAttenderConfig) -> FeedForward:
        return build_ff(config.ff)

    def build_norm(self, config: CrossAttenderConfig) -> Norm:
        return build_norm(config.norm)
