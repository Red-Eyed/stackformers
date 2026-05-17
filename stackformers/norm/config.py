from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class RMSNormConfig(BaseModel):
    kind: Literal["rms"] = "rms"
    dim: int = Field(gt=0)
    eps: float = Field(default=1e-6, gt=0)


class LayerNormConfig(BaseModel):
    kind: Literal["layer_norm"] = "layer_norm"
    dim: int = Field(gt=0)
    eps: float = 1e-5


NormConfig = Annotated[RMSNormConfig | LayerNormConfig, Field(discriminator="kind")]
