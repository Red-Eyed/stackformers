from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class SDPAKernelConfig(BaseModel):
    kind: Literal["sdpa"] = "sdpa"


class WindowedSDPAKernelConfig(BaseModel):
    kind: Literal["windowed_sdpa"] = "windowed_sdpa"
    window_size: int = Field(gt=0)
    mode: Literal["mask", "unfold"] = "mask"


class VarlenSDPAKernelConfig(BaseModel):
    kind: Literal["varlen_sdpa"] = "varlen_sdpa"


class VarlenWindowedSDPAKernelConfig(BaseModel):
    kind: Literal["varlen_windowed_sdpa"] = "varlen_windowed_sdpa"
    window_size: int = Field(gt=0)


KernelConfig = Annotated[
    SDPAKernelConfig
    | WindowedSDPAKernelConfig
    | VarlenSDPAKernelConfig
    | VarlenWindowedSDPAKernelConfig,
    Field(discriminator="kind"),
]
