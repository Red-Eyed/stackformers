from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class AttentionConfig(BaseModel):
    dim: int = Field(gt=0)
    heads: int = Field(default=8, gt=0)
    dim_head: int = Field(default=64, gt=0)
    kv_heads: int | None = None  # None → same as heads (MHA); set for GQA/MQA
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    causal: bool = False

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
