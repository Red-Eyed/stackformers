from __future__ import annotations

from typing import Generic, TypeVar

import torch.nn as nn
from jaxtyping import Float
from pydantic import BaseModel, Field
from torch import Tensor

from stackformers.attention.bias import NoBiasBuilder
from stackformers.attention.config import AttentionConfig
from stackformers.attention.kernels import SDPAKernel
from stackformers.attention.self_attn import SelfAttention
from stackformers.encoder import Encoder
from stackformers.feedforward.config import FeedForwardConfig
from stackformers.feedforward.swiglu import SwiGLU
from stackformers.layers import TransformerLayer
from stackformers.norm.rms import RMSNorm
from stackformers.positional.rope1d import RotaryEmbedding1D
from stackformers.sequence import SequenceInfo


class TransformerEncoderConfig(BaseModel):
    dim: int = Field(gt=0)
    heads: int = Field(default=8, gt=0)
    dim_head: int = Field(default=64, gt=0)
    num_layers: int = Field(gt=0)
    ff_mult: float = Field(default=4.0, gt=0.0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    causal: bool = False


ConfigT = TypeVar("ConfigT", bound=TransformerEncoderConfig)


class TransformerEncoder(nn.Module, Generic[ConfigT]):
    """Opinionated encoder preset: RMSNorm + SwiGLU + RoPE-1D + SDPA.

    Extend by subclassing with a richer config bound to ConfigT.
    """

    def __init__(self, config: ConfigT) -> None:
        super().__init__()
        self._config = config

        attn_cfg = AttentionConfig(
            dim=config.dim,
            heads=config.heads,
            dim_head=config.dim_head,
            causal=config.causal,
        )
        ff_cfg = FeedForwardConfig(dim=config.dim, mult=config.ff_mult)
        pos = RotaryEmbedding1D(dim_head=config.dim_head)

        self._encoder = Encoder(
            layers=[
                TransformerLayer(
                    self_attn=SelfAttention(
                        config=attn_cfg,
                        pos_encoding=pos,
                        bias_builder=NoBiasBuilder(),
                        kernel=SDPAKernel(dropout=config.dropout),
                    ),
                    ff=SwiGLU(ff_cfg),
                    norm_attn=RMSNorm(config.dim),
                    norm_ff=RMSNorm(config.dim),
                )
                for _ in range(config.num_layers)
            ],
            final_norm=RMSNorm(config.dim),
        )

    @property
    def config(self) -> ConfigT:
        return self._config

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        seq_info: SequenceInfo,
    ) -> Float[Tensor, "b n d"]:
        return self._encoder(x, seq_info)
