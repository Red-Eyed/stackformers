from __future__ import annotations

from typing import Generic, TypeVar

import torch.nn as nn
from jaxtyping import Float
from pydantic import BaseModel, Field
from torch import Tensor

from stackformers.attention.bias import NoBiasBuilder
from stackformers.attention.config import AttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.kernels import SDPAKernel
from stackformers.attention.self_attn import SelfAttention
from stackformers.decoder import Decoder, DecoderLayer
from stackformers.feedforward.config import FeedForwardConfig
from stackformers.feedforward.swiglu import SwiGLU
from stackformers.norm.rms import RMSNorm
from stackformers.positional.none import NoPosEncoding
from stackformers.positional.rope1d import RotaryEmbedding1D
from stackformers.sequence import SequenceInfo


class TransformerEncoderCrossConfig(BaseModel):
    """Config for an encoder/decoder block with self-attention and cross-attention.

    context must have the same dim as x. Use causal=True for decoder-style (GPT),
    causal=False for encoder-style (BERT cross-attending to context).
    """

    dim: int = Field(gt=0)
    heads: int = Field(default=8, gt=0)
    dim_head: int = Field(default=64, gt=0)
    num_layers: int = Field(gt=0)
    ff_mult: float = Field(default=4.0, gt=0.0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    causal: bool = False


ConfigT = TypeVar("ConfigT", bound=TransformerEncoderCrossConfig)


class TransformerEncoderCross(nn.Module, Generic[ConfigT]):
    """Opinionated cross-attention preset: RMSNorm + SwiGLU + RoPE-1D + SDPA.

    Each layer: pre-norm self-attn → pre-norm cross-attn → pre-norm feed-forward.
    Self-attention uses RoPE; cross-attention uses no positional encoding.
    context_dim must equal config.dim.

    Extend by subclassing with a richer config bound to ConfigT.
    """

    def __init__(self, config: ConfigT) -> None:
        super().__init__()
        self._config = config

        self_attn_cfg = AttentionConfig(
            dim=config.dim,
            heads=config.heads,
            dim_head=config.dim_head,
            causal=config.causal,
        )
        cross_attn_cfg = AttentionConfig(
            dim=config.dim,
            heads=config.heads,
            dim_head=config.dim_head,
        )
        ff_cfg = FeedForwardConfig(dim=config.dim, mult=config.ff_mult)
        pos = RotaryEmbedding1D(dim_head=config.dim_head)

        self._decoder = Decoder(
            layers=[
                DecoderLayer(
                    self_attn=SelfAttention(
                        config=self_attn_cfg,
                        pos_encoding=pos,
                        bias_builder=NoBiasBuilder(),
                        kernel=SDPAKernel(dropout=config.dropout),
                    ),
                    cross_attn=CrossAttention(
                        config=cross_attn_cfg,
                        pos_encoding=NoPosEncoding(),
                        bias_builder=NoBiasBuilder(),
                        kernel=SDPAKernel(dropout=config.dropout),
                    ),
                    ff=SwiGLU(ff_cfg),
                    norm_self=RMSNorm(config.dim),
                    norm_cross=RMSNorm(config.dim),
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
        context: Float[Tensor, "b s d"],
        seq_info: SequenceInfo,
        ctx_seq_info: SequenceInfo | None = None,
    ) -> Float[Tensor, "b n d"]:
        return self._decoder(x, context, seq_info, ctx_seq_info)
