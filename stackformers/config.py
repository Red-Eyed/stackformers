from __future__ import annotations

from pydantic import BaseModel, Field

from stackformers.attention.config import CrossAttentionConfig, SelfAttentionConfig
from stackformers.feedforward.config import FeedForwardConfig


class LayerConfig(BaseModel):
    attn: SelfAttentionConfig
    ff: FeedForwardConfig
    pre_norm: bool = True


class EncoderConfig(BaseModel):
    layer: LayerConfig
    num_layers: int = Field(gt=0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)


class DecoderConfig(BaseModel):
    self_attn: SelfAttentionConfig
    cross_attn: CrossAttentionConfig
    ff: FeedForwardConfig
    num_layers: int = Field(gt=0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
