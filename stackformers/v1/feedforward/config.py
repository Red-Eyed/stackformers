from __future__ import annotations

from pydantic import BaseModel, Field


class FeedForwardConfig(BaseModel):
    dim: int = Field(gt=0)
    mult: float = Field(default=4.0, gt=0.0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def inner_dim(self) -> int:
        # SwiGLU: two gate matrices, so scale down to match param count with GELU-4x
        return int(self.dim * self.mult * 2 / 3)
