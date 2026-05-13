from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class NoBiasConfig(BaseModel):
    kind: Literal["none"] = "none"


class ALiBiConfig(BaseModel):
    kind: Literal["alibi"] = "alibi"


BiasBuilderConfig = Annotated[NoBiasConfig | ALiBiConfig, Field(discriminator="kind")]
