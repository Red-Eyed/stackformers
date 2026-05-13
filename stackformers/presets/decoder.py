from __future__ import annotations

import torch.nn as nn
from pydantic import BaseModel, Field
from torch import Tensor

from stackformers.attention.bias import NoBiasBuilder
from stackformers.attention.config import AttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.kernels import SDPAKernel
from stackformers.attention.self_attn import SelfAttention
from stackformers.decoder import Decoder, DecoderLayer
from stackformers.feedforward.config import FeedForwardConfig
from stackformers.positional.config import NoPosEncodingConfig, PosEncodingConfig
from stackformers.positional.none import NoPosEncoding
from stackformers.presets.configs import NormConfig, build_ff, build_norm, build_pos_encoding
from stackformers.sequence import SequenceInput


class TransformerDecoderConfig(BaseModel):
    self_attn: AttentionConfig
    cross_attn: AttentionConfig
    ff: FeedForwardConfig
    norm: NormConfig
    pos_encoding: PosEncodingConfig  # applies to self-attention only
    num_layers: int = Field(gt=0)


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
                        bias_builder=NoBiasBuilder(),
                        kernel=SDPAKernel(causal=True, dropout=config.self_attn.dropout),
                    ),
                    cross_attn=CrossAttention(
                        config=config.cross_attn,
                        pos_encoding=NoPosEncoding(NoPosEncodingConfig()),
                        bias_builder=NoBiasBuilder(),
                        kernel=SDPAKernel(causal=False, dropout=config.cross_attn.dropout),
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
