from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch.nn as nn
from pydantic import BaseModel, Field, model_validator
from torch import Tensor

from stackformers.attention.config import (
    AttnBiasConfig,
    DistanceBiasConfig,
    NoAttnBiasConfig,
    SelfAttentionConfig,
)
from stackformers.attention.factory import build_attn_bias
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
from stackformers.positional.config import (
    NoPosEncodingConfig,
    PosEncodingConfig,
    RoPE1DConfig,
)
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
    attn_bias: AttnBiasConfig = NoAttnBiasConfig()

    @model_validator(mode="after")
    def _check_bias_heads(self) -> TransformerEncoderConfig:
        """A head-count mismatch would otherwise surface as a broadcast error inside SDPA."""
        if (
            isinstance(self.attn_bias, DistanceBiasConfig)
            and self.attn_bias.heads != self.attn.heads
        ):
            raise ValueError(
                f"attn_bias.heads ({self.attn_bias.heads}) must equal attn.heads"
                f" ({self.attn.heads}) — the bias contributes one logit per query head."
            )
        return self


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


def node_encoder_config(
    dim: int,
    heads: int,
    num_layers: int,
    r_max: float,
    *,
    num_rbf: int = 32,
    ff_mult: float = 4.0,
    dropout: float = 0.0,
) -> TransformerEncoderConfig:
    """Encoder for geometric node sets, where attention is driven by relative distance.

    Carries no positional encoding: geometry enters only through a learned function of
    ‖p_i − p_j‖, so the model is invariant to global translation and rotation of the nodes.
    Choose this when the node coordinates have no meaningful frame — a scene that arrives
    rotated is the same scene. When up and right *do* mean something (an image plane, a map
    with north up), prefer RoPE-2D, which keeps direction and stays compatible with varlen.

    `abs_positions` must carry the node coordinates (c=2), not sequence indices.
    Set `r_max` from the data: a high percentile of the pairwise distance distribution.

    Attention runs on the padded SDPA path — an attention bias has no varlen_attn slot.
    """
    dim_head = dim // heads
    return TransformerEncoderConfig(
        attn=SelfAttentionConfig(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
        ff=SwiGLUConfig(dim=dim, mult=ff_mult, dropout=dropout),
        norm=RMSNormConfig(dim=dim),
        pos_encoding=NoPosEncodingConfig(),
        attn_bias=DistanceBiasConfig(heads=heads, r_max=r_max, num_rbf=num_rbf),
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

    def build_attn_bias(self, config: TransformerEncoderConfig) -> AttnBias:
        return build_attn_bias(config.attn_bias)

    def build_ff(self, config: TransformerEncoderConfig) -> FeedForward:
        return build_ff(config.ff)

    def build_norm(self, config: TransformerEncoderConfig) -> Norm:
        return build_norm(config.norm)
