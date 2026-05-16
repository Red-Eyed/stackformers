from __future__ import annotations

import warnings
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class _FFBase(BaseModel):
    """Shared geometry and alignment check for gated feed-forward variants."""

    dim: int = Field(gt=0)
    mult: float = Field(default=4.0, gt=0.0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)

    _ALIGN = 64  # tensor-core alignment for FP16/BF16

    @model_validator(mode="after")
    def _check_inner_dim_alignment(self) -> _FFBase:
        d = self.inner_dim
        if d % self._ALIGN != 0:
            warnings.warn(
                f"inner_dim={d} is not a multiple of {self._ALIGN}. "
                "Unaligned dimensions reduce GPU throughput on tensor-core hardware. "
                f"Nearest aligned values: {(d // self._ALIGN) * self._ALIGN} or "
                f"{(d // self._ALIGN + 1) * self._ALIGN}. "
                "Adjust dim or mult so that int(dim * mult * 2/3) is a multiple of "
                f"{self._ALIGN}.",
                UserWarning,
                stacklevel=2,
            )
        return self

    @property
    def inner_dim(self) -> int:
        # Two gate matrices → scale inner dim down to match param count with GELU-4x FFN.
        return int(self.dim * self.mult * 2 / 3)


class SwiGLUConfig(_FFBase):
    """Config for the SwiGLU feed-forward network (Noam Shazeer, 2020)."""

    kind: Literal["swiglu"] = "swiglu"


class GEGLUConfig(_FFBase):
    """Config for the GEGLU feed-forward network (Noam Shazeer, 2020)."""

    kind: Literal["geglu"] = "geglu"


FeedForwardConfig = Annotated[SwiGLUConfig | GEGLUConfig, Field(discriminator="kind")]
