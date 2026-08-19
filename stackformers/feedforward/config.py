"""Validated configuration types for feed-forward implementations."""

from __future__ import annotations

import warnings
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class _FFBase(BaseModel):
    """Provide shared geometry and alignment checks for feed-forward variants."""

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
                f"Adjust dim or mult so that inner_dim is a multiple of {self._ALIGN}.",
                UserWarning,
                stacklevel=2,
            )
        return self

    @property
    def inner_dim(self) -> int:
        """Return the parameter-matched hidden width for gated variants."""
        # Two gate matrices → scale inner dim down to match param count with GELU-4x FFN.
        return int(self.dim * self.mult * 2 / 3)


class SwiGLUConfig(_FFBase):
    """Config for the SwiGLU feed-forward network (Noam Shazeer, 2020)."""

    kind: Literal["swiglu"] = "swiglu"


class HardSwishGLUConfig(_FFBase):
    """Config for a parameter-matched GLU with a HardSwish gate."""

    kind: Literal["hardswish_glu"] = "hardswish_glu"


class GEGLUConfig(_FFBase):
    """Config for the GEGLU feed-forward network (Noam Shazeer, 2020)."""

    kind: Literal["geglu"] = "geglu"


class GELUConfig(_FFBase):
    """Config for the standard GELU feed-forward network."""

    kind: Literal["gelu"] = "gelu"

    @property
    def inner_dim(self) -> int:
        """Return the hidden width for the non-gated network."""
        return int(self.dim * self.mult)


class ReluSquaredConfig(_FFBase):
    """Config for the ReLU² feed-forward network.

    Non-gated: inner_dim = dim * mult (no 2/3 factor).
    Paper: "Primer: Searching for Efficient Transformers" — https://arxiv.org/abs/2109.08668
    """

    kind: Literal["relu_squared"] = "relu_squared"

    @property
    def inner_dim(self) -> int:
        """Return the hidden width for the non-gated network."""
        return int(self.dim * self.mult)


FeedForwardConfig = Annotated[
    SwiGLUConfig | HardSwishGLUConfig | GEGLUConfig | GELUConfig | ReluSquaredConfig,
    Field(discriminator="kind"),
]
