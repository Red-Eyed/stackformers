from __future__ import annotations

from typing import Annotated, Literal

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
    kind: Literal["rope1d"] = "rope1d"
    dim_head: int = Field(gt=0)
    base: int = Field(default=10_000, gt=0)
    yarn: YaRNConfig | None = None


class RoPE2DConfig(BaseModel):
    kind: Literal["rope2d"] = "rope2d"
    dim_head: int = Field(gt=0)
    base: int = Field(default=10_000, gt=0)


class RoPENDConfig(BaseModel):
    """Rotary encoding over c continuous spatial dimensions.

    Takes no ``base``. RoPE's ``base`` ladder pins its fastest band at a wavelength of 2π
    *coordinate units* regardless of the base, so it only lands correctly when tokens sit one
    unit apart — true for text and patch grids, meaningless for continuous coordinates. The
    band range is set by the data instead: r_min fixes the fast end, r_max the slow end, and
    each is a half turn over the scale it names (ω = π / scale). Nothing else is free. The
    ladder's shape then depends only on the ratio r_max / r_min, so the units the coordinates
    happen to be expressed in stop mattering.
    """

    kind: Literal["rope_nd"] = "rope_nd"
    dim_head: int = Field(gt=0)
    coords: int = Field(
        default=2, gt=0, description="Number of spatial dimensions (c) in abs_positions."
    )
    r_min: float = Field(
        gt=0.0,
        description=(
            "Finest separation between two nodes that the model must tell apart. Gets a half"
            " turn on the fastest band, so the shortest wavelength is 2*r_min — the Nyquist"
            " limit, below which distinct offsets alias onto the same rotation. Measure it as a"
            " low percentile of the nearest-neighbour distance, not the minimum, which is noise."
        ),
    )
    r_max: float = Field(
        gt=0.0,
        description=(
            "Diameter of the domain — the largest offset that must stay distinguishable."
            " Attention sees signed offsets spanning [-r_max, +r_max], so the slowest band is"
            " given a half turn over that full width of 2*r_max: it never wraps, which leaves it"
            " monotone in the offset. Measure it as a high percentile of the pairwise distance"
            " distribution, not the maximum, which is an outlier."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> RoPENDConfig:
        pairs = 2 * self.coords
        if self.dim_head % pairs != 0:
            raise ValueError(
                f"dim_head ({self.dim_head}) must be divisible by 2 * coords ({pairs}):"
                f" dim_head is split into {self.coords} per-axis blocks, and each block is"
                " rotated in pairs of channels."
            )
        if self.bands_per_axis < 2:
            raise ValueError(
                f"dim_head ({self.dim_head}) leaves {self.bands_per_axis} band per axis; at"
                f" least 2 bands are needed, so dim_head must be at least {4 * self.coords}."
                " A lone band lands on the fast end of the ladder and r_max is never reached,"
                " leaving the encoding periodic with period 2 * r_min across the whole domain:"
                " every offset an integer number of periods apart becomes indistinguishable."
            )
        if self.r_max <= self.r_min:
            raise ValueError(f"r_max ({self.r_max}) must exceed r_min ({self.r_min}).")
        return self

    @property
    def bands_per_axis(self) -> int:
        return self.dim_head // (2 * self.coords)


class NoPosEncodingConfig(BaseModel):
    kind: Literal["none"] = "none"


class LearnedPosEncodingConfig(BaseModel):
    """Config for learned absolute position embeddings (Vaswani et al., 2017)."""

    kind: Literal["learned"] = "learned"
    dim_head: int = Field(gt=0)
    max_seq_len: int = Field(gt=0)


PosEncodingConfig = Annotated[
    RoPE1DConfig | RoPE2DConfig | RoPENDConfig | NoPosEncodingConfig | LearnedPosEncodingConfig,
    Field(discriminator="kind"),
]
