from __future__ import annotations

from pydantic import BaseModel, Field

from stackformers.v1.attention.config import AttentionConfig
from stackformers.v1.feedforward.config import FeedForwardConfig


class LayerConfig(BaseModel):
    attn: AttentionConfig
    ff: FeedForwardConfig
    pre_norm: bool = True


class EncoderConfig(BaseModel):
    layer: LayerConfig
    num_layers: int = Field(gt=0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)


class DecoderConfig(BaseModel):
    self_attn: AttentionConfig
    cross_attn: AttentionConfig
    ff: FeedForwardConfig
    num_layers: int = Field(gt=0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
