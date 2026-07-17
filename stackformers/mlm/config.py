from __future__ import annotations

from pydantic import BaseModel, Field


class MLMWrapperConfig(BaseModel):
    """Config for MLMWrapper: token dimension and corruption ratio."""

    dim: int = Field(gt=0)
    mask_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
