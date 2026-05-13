from __future__ import annotations

import warnings

from pydantic import BaseModel, Field, model_validator

_ALIGN = 64  # tensor-core alignment for FP16/BF16


class AttentionConfig(BaseModel):
    dim: int = Field(gt=0)
    heads: int = Field(default=8, gt=0)
    dim_head: int = Field(default=64, gt=0)
    kv_heads: int | None = None  # None → same as heads (MHA); set for GQA/MQA
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    causal: bool = False

    @model_validator(mode="after")
    def _validate(self) -> "AttentionConfig":
        if self.dim % self.heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by heads ({self.heads})")
        kv_heads = self.kv_heads if self.kv_heads is not None else self.heads
        if self.heads % kv_heads != 0:
            raise ValueError(f"heads ({self.heads}) must be divisible by kv_heads ({kv_heads})")

        if self.dim_head % _ALIGN != 0:
            warnings.warn(
                f"dim_head={self.dim_head} is not a multiple of {_ALIGN}. "
                "Unaligned head dimension reduces GPU throughput on tensor-core hardware. "
                f"Nearest aligned values: {(self.dim_head // _ALIGN) * _ALIGN} or "
                f"{(self.dim_head // _ALIGN + 1) * _ALIGN}.",
                UserWarning,
                stacklevel=2,
            )

        projected = self.heads * self.dim_head
        if projected != self.dim:
            warnings.warn(
                f"heads * dim_head = {self.heads} * {self.dim_head} = {projected}, "
                f"which does not equal dim={self.dim}. "
                "The Q/K/V projections are non-square — this is valid but may not be intentional. "
                "Standard transformers use dim_head = dim // heads = "
                f"{self.dim // self.heads}.",
                UserWarning,
                stacklevel=2,
            )

        return self

    @property
    def effective_kv_heads(self) -> int:
        return self.kv_heads if self.kv_heads is not None else self.heads

    @property
    def groups(self) -> int:
        return self.heads // self.effective_kv_heads
