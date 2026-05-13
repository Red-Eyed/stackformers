from __future__ import annotations

from pydantic import BaseModel, Field


class RMSNormConfig(BaseModel):
    dim: int = Field(gt=0)


class LayerNormConfig(BaseModel):
    dim: int = Field(gt=0)
    eps: float = 1e-5
