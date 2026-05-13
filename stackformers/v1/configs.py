from __future__ import annotations

from pydantic import BaseModel, field_validator, model_validator


class AttentionConfig(BaseModel):
    dim: int
    heads: int = 8
    dim_head: int = 64
    kv_heads: int | None = None  # None → same as heads (MHA); set for GQA/MQA
    dropout: float = 0.0
    causal: bool = False

    @field_validator("dim")
    @classmethod
    def dim_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"dim must be positive, got {v}")
        return v

    @field_validator("heads")
    @classmethod
    def heads_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"heads must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def heads_divides_dim(self) -> "AttentionConfig":
        if self.dim % self.heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by heads ({self.heads})")
        kv_heads = self.kv_heads if self.kv_heads is not None else self.heads
        if self.heads % kv_heads != 0:
            raise ValueError(f"heads ({self.heads}) must be divisible by kv_heads ({kv_heads})")
        return self

    @property
    def effective_kv_heads(self) -> int:
        return self.kv_heads if self.kv_heads is not None else self.heads

    @property
    def groups(self) -> int:
        return self.heads // self.effective_kv_heads


class FeedForwardConfig(BaseModel):
    dim: int
    mult: float = 4.0
    dropout: float = 0.0

    @field_validator("dim")
    @classmethod
    def dim_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"dim must be positive, got {v}")
        return v

    @field_validator("mult")
    @classmethod
    def mult_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"mult must be positive, got {v}")
        return v

    @property
    def inner_dim(self) -> int:
        # SwiGLU: two gate matrices, so scale down to match param count with GELU-4x
        return int(self.dim * self.mult * 2 / 3)


class LayerConfig(BaseModel):
    attn: AttentionConfig
    ff: FeedForwardConfig
    pre_norm: bool = True  # pre-norm (recommended) vs post-norm


class EncoderConfig(BaseModel):
    layer: LayerConfig
    num_layers: int
    dropout: float = 0.0

    @field_validator("num_layers")
    @classmethod
    def num_layers_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"num_layers must be positive, got {v}")
        return v


class DecoderConfig(BaseModel):
    self_attn: AttentionConfig
    cross_attn: AttentionConfig
    ff: FeedForwardConfig
    num_layers: int
    dropout: float = 0.0

    @field_validator("num_layers")
    @classmethod
    def num_layers_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"num_layers must be positive, got {v}")
        return v
