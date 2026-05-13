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
from stackformers.cross_attender import CrossAttenderLayer, CrossAttenderStack
from stackformers.feedforward.config import FeedForwardConfig
from stackformers.feedforward.factory import build_ff
from stackformers.norm.config import RMSNormConfig
from stackformers.norm.factory import NormConfig, build_norm
from stackformers.positional.config import NoPosEncodingConfig
from stackformers.positional.none import NoPosEncoding
from stackformers.sequence import SequenceInput


class CrossAttenderConfig(BaseModel):
    attn: AttentionConfig  # causal is always False; kernel defaults to SDPA
    ff: FeedForwardConfig
    norm: NormConfig
    bias: BiasBuilderConfig = NoBiasConfig()
    num_layers: int = Field(gt=0)


def plain_cross_attender_config(
    dim: int,
    heads: int,
    num_layers: int,
    *,
    ff_mult: float = 4.0,
    dropout: float = 0.0,
) -> CrossAttenderConfig:
    """Padded SDPA cross-attender with RMSNorm and SwiGLU FF.

    Queries from x attend to context with no positional encoding or self-attention.
    Context must have the same dim. Suitable for Perceiver-style or slot-attention architectures.
    """
    dim_head = dim // heads
    return CrossAttenderConfig(
        attn=AttentionConfig(
            dim=dim, heads=heads, dim_head=dim_head, dropout=dropout, kernel=SDPAKernelConfig()
        ),
        ff=FeedForwardConfig(dim=dim, mult=ff_mult, dropout=dropout),
        norm=RMSNormConfig(dim=dim),
        bias=NoBiasConfig(),
        num_layers=num_layers,
    )


class CrossAttender(nn.Module):
    """Opinionated cross-attender preset: queries from x attend to context, no self-attention.

    Each layer: pre-norm cross-attn → pre-norm feed-forward.
    context_dim must equal attn.dim.
    """

    def __init__(self, config: CrossAttenderConfig) -> None:
        super().__init__()
        self.config = config

        cross_attn_cfg = config.attn.model_copy(update={"causal": False})

        self._stack = CrossAttenderStack(
            layers=[
                CrossAttenderLayer(
                    cross_attn=CrossAttention(
                        config=cross_attn_cfg,
                        pos_encoding=NoPosEncoding(NoPosEncodingConfig()),
                        bias_builder=build_bias_builder(config.bias, cross_attn_cfg.heads),
                        kernel=build_kernel(cross_attn_cfg),
                    ),
                    ff=build_ff(config.ff),
                    norm_cross=build_norm(config.norm),
                    norm_ff=build_norm(config.norm),
                )
                for _ in range(config.num_layers)
            ],
            final_norm=build_norm(config.norm),
        )

    def forward(self, x_input: SequenceInput, ctx_input: SequenceInput) -> Tensor:
        return self._stack(x_input, ctx_input)
