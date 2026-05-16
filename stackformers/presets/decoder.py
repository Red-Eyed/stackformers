from __future__ import annotations

import torch.nn as nn
from pydantic import BaseModel, Field
from torch import Tensor

from stackformers.attention.bias_config import BiasBuilderConfig, NoBiasConfig
from stackformers.attention.bias_factory import build_bias_builder
from stackformers.attention.config import AttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.kernels.config import SDPAKernelConfig
from stackformers.attention.kernels.factory import build_kernel
from stackformers.attention.self_attn import SelfAttention
from stackformers.decoder import Decoder, DecoderLayer
from stackformers.feedforward.config import FeedForwardConfig, SwiGLUConfig
from stackformers.feedforward.factory import build_ff
from stackformers.norm.config import RMSNormConfig
from stackformers.norm.factory import NormConfig, build_norm
from stackformers.positional.config import NoPosEncodingConfig, PosEncodingConfig, RoPE1DConfig
from stackformers.positional.factory import build_pos_encoding
from stackformers.positional.none import NoPosEncoding
from stackformers.sequence import SequenceInput


class TransformerDecoderConfig(BaseModel):
    self_attn: AttentionConfig
    cross_attn: AttentionConfig
    ff: FeedForwardConfig
    norm: NormConfig
    pos_encoding: PosEncodingConfig  # applies to self-attention only
    self_attn_bias: BiasBuilderConfig = NoBiasConfig()
    cross_attn_bias: BiasBuilderConfig = NoBiasConfig()
    num_layers: int = Field(gt=0)


def plain_decoder_config(
    dim: int,
    heads: int,
    num_layers: int,
    *,
    ff_mult: float = 4.0,
    dropout: float = 0.0,
) -> TransformerDecoderConfig:
    """Padded SDPA decoder with RoPE-1D self-attention, no-pos cross-attention, RMSNorm, SwiGLU FF.

    Self-attention is always causal. Cross-attention uses no positional encoding.
    Both self- and cross-attention share dim and heads; context must have the same dim.
    """
    dim_head = dim // heads
    attn = AttentionConfig(
        dim=dim, heads=heads, dim_head=dim_head, dropout=dropout, kernel=SDPAKernelConfig()
    )
    return TransformerDecoderConfig(
        self_attn=attn,
        cross_attn=attn,
        ff=SwiGLUConfig(dim=dim, mult=ff_mult, dropout=dropout),
        norm=RMSNormConfig(dim=dim),
        pos_encoding=RoPE1DConfig(dim_head=dim_head),
        num_layers=num_layers,
    )


class TransformerDecoder(nn.Module):
    """Opinionated decoder preset: causal self-attn → cross-attn → feed-forward per layer.

    Self-attention is always causal. Cross-attention uses NoPosEncoding.
    context_dim must equal self_attn.dim.
    """

    def __init__(self, config: TransformerDecoderConfig) -> None:
        super().__init__()
        self.config = config

        self_attn_cfg = config.self_attn.model_copy(update={"causal": True})
        self_pos = build_pos_encoding(config.pos_encoding)

        self._decoder = Decoder(
            layers=[
                DecoderLayer(
                    self_attn=SelfAttention(
                        config=self_attn_cfg,
                        pos_encoding=self_pos,
                        bias_builder=build_bias_builder(config.self_attn_bias, self_attn_cfg.heads),
                        kernel=build_kernel(self_attn_cfg),
                    ),
                    cross_attn=CrossAttention(
                        config=config.cross_attn,
                        pos_encoding=NoPosEncoding(NoPosEncodingConfig()),
                        bias_builder=build_bias_builder(
                            config.cross_attn_bias, config.cross_attn.heads
                        ),
                        kernel=build_kernel(config.cross_attn),
                    ),
                    ff=build_ff(config.ff),
                    norm_self=build_norm(config.norm),
                    norm_cross=build_norm(config.norm),
                    norm_ff=build_norm(config.norm),
                )
                for _ in range(config.num_layers)
            ],
            final_norm=build_norm(config.norm),
        )

    def forward(self, x_input: SequenceInput, ctx_input: SequenceInput) -> Tensor:
        return self._decoder(x_input, ctx_input)
