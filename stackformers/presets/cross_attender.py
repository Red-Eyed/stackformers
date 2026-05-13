from __future__ import annotations

import torch.nn as nn
from pydantic import BaseModel, Field
from torch import Tensor

from stackformers.attention.bias import NoBiasBuilder
from stackformers.attention.config import AttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.kernels import SDPAKernel
from stackformers.cross_attender import CrossAttenderLayer, CrossAttenderStack
from stackformers.feedforward.config import FeedForwardConfig
from stackformers.positional.config import NoPosEncodingConfig
from stackformers.positional.none import NoPosEncoding
from stackformers.presets.configs import NormConfig, build_ff, build_norm
from stackformers.sequence import SequenceInput


class CrossAttenderConfig(BaseModel):
    attn: AttentionConfig  # dim, heads, dim_head, dropout; causal is always False here
    ff: FeedForwardConfig
    norm: NormConfig
    num_layers: int = Field(gt=0)


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
                        bias_builder=NoBiasBuilder(),
                        kernel=SDPAKernel(causal=False, dropout=config.attn.dropout),
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
