from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class YaRNConfig(BaseModel):
    """NTK-by-parts RoPE frequency scaling (Peng et al., 2023).

    Extends RoPE to longer contexts by scaling low-frequency dimensions
    linearly while leaving high-frequency dimensions unchanged.  The
    inv_freq buffer is modified once at construction time so forward()
    stays export-traceable.

    scale:               target_max_seq_len / original_max_seq_len
    original_max_seq_len: context length the base model was trained on
    beta_fast:            high-freq threshold (wavelength = original / beta_fast)
    beta_slow:            low-freq threshold  (wavelength = original / beta_slow)
    """

    scale: float = Field(gt=1.0)
    original_max_seq_len: int = Field(gt=0)
    beta_fast: float = Field(default=32.0, gt=0.0)
    beta_slow: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def _check_beta_ordering(self) -> YaRNConfig:
        if self.beta_slow >= self.beta_fast:
            raise ValueError(
                f"beta_slow ({self.beta_slow}) must be less than beta_fast ({self.beta_fast}). "
                "Swapped values invert the high/low-frequency partition and silently break scaling."
            )
        return self


class RoPE1DConfig(BaseModel):
    dim_head: int = Field(gt=0)
    base: int = Field(default=10_000, gt=0)
    yarn: YaRNConfig | None = None


class NoPosEncodingConfig(BaseModel):
    pass  # null object — no parameters by design


PosEncodingConfig = RoPE1DConfig | NoPosEncodingConfig
