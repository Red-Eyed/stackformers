from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from stackformers.attention.config import _validate_attn_dims
from stackformers.norm.config import NormConfig, RMSNormConfig


class NoKVReductionConfig(BaseModel):
    kind: Literal["none"] = "none"


class ConvKVReductionConfig(BaseModel):
    """Strided-conv spatial reduction of the key/value context.

    Paper: "Pyramid Vision Transformer" (SRA), Wang et al., 2021 —
    https://arxiv.org/abs/2102.12122
    """

    kind: Literal["conv"] = "conv"
    dim: int = Field(gt=0, description="Channel width of the tokens being reduced.")
    reduction: int = Field(
        gt=1,
        description="Spatial stride r: an H×W grid is pooled to (H/r)×(W/r). Must divide H and W.",
    )
    norm: NormConfig = Field(
        default_factory=lambda: RMSNormConfig(dim=1),
        description="Norm applied to the reduced tokens. Defaults to RMSNorm, matching the "
        "rest of the library; set to LayerNormConfig for the original PVTv2 behaviour.",
    )

    @model_validator(mode="after")
    def _sync_norm_dim(self) -> ConvKVReductionConfig:
        # The norm always acts on `dim` channels, so derive its width rather than making
        # the caller repeat it (and risk a mismatch).
        self.norm = self.norm.model_copy(update={"dim": self.dim})
        return self


KVReductionConfig = Annotated[
    NoKVReductionConfig | ConvKVReductionConfig,
    Field(discriminator="kind"),
]


class _SpatialAttnBase(BaseModel):
    """Shared geometry for 2-D attention variants (mirrors SelfAttentionConfig)."""

    dim: int = Field(gt=0, description="Model dimension — input/output width of the sublayer.")
    heads: int = Field(default=8, gt=0, description="Number of query heads.")
    dim_head: int = Field(default=64, gt=0, description="Dimension per head.")
    kv_heads: int | None = Field(
        default=None, description="Key/value heads. None = heads (MHA); a divisor of heads = GQA."
    )
    dropout: float = Field(default=0.0, ge=0.0, le=1.0, description="Output-projection dropout.")
    qk_norm: bool = Field(default=False, description="RMSNorm queries and keys before the product.")

    @model_validator(mode="after")
    def _validate_dims(self) -> _SpatialAttnBase:
        _validate_attn_dims(self.dim, self.heads, self.dim_head, self.kv_heads)
        return self

    @property
    def effective_kv_heads(self) -> int:
        return self.kv_heads if self.kv_heads is not None else self.heads

    @property
    def groups(self) -> int:
        return self.heads // self.effective_kv_heads


class WindowAttention2DConfig(_SpatialAttnBase):
    """Non-overlapping w×w window attention over the grid — O(n·w²), local.

    Distinct from SelfAttentionConfig.window_size, which is a 1-D sliding window over a
    flat sequence; this partitions the 2-D grid into square tiles.

    Paper: "Swin Transformer" (window partitioning), Liu et al., 2021 —
    https://arxiv.org/abs/2103.14030
    """

    kind: Literal["window2d"] = "window2d"
    window: int = Field(gt=0, description="Window side length w; must divide grid H and W.")


class SpatialReductionAttentionConfig(_SpatialAttnBase):
    """Global attention with spatially-reduced K/V — O(n·s).

    Paper: "Pyramid Vision Transformer" (SRA), Wang et al., 2021 —
    https://arxiv.org/abs/2102.12122
    """

    kind: Literal["spatial_reduction"] = "spatial_reduction"
    reduction: int = Field(
        default=1,
        ge=1,
        description="K/V spatial stride r. 1 = full global attention (no reduction).",
    )
    kv_norm: NormConfig = Field(
        default_factory=lambda: RMSNormConfig(dim=1),
        description="Norm applied to reduced K/V tokens (ignored when reduction == 1).",
    )

    @model_validator(mode="after")
    def _sync_kv_norm_dim(self) -> SpatialReductionAttentionConfig:
        self.kv_norm = self.kv_norm.model_copy(update={"dim": self.dim})
        return self


SpatialAttnConfig = Annotated[
    WindowAttention2DConfig | SpatialReductionAttentionConfig,
    Field(discriminator="kind"),
]


class PatchMergingConfig(BaseModel):
    """Overlapping strided-conv downsample between pyramid stages (defaults k3 s2 p1).

    Paper: "PVTv2: Improved Baselines with Pyramid Vision Transformer" (overlapping patch
    embedding), Wang et al., 2022 — https://arxiv.org/abs/2106.13797
    """

    in_dim: int = Field(gt=0, description="Input channel width.")
    out_dim: int = Field(gt=0, description="Output channel width after merging.")
    kernel: int = Field(default=3, gt=0, description="Conv kernel size.")
    stride: int = Field(default=2, gt=0, description="Conv stride — the downsample factor.")
    padding: int = Field(default=1, ge=0, description="Conv padding.")
    norm: NormConfig = Field(
        default_factory=lambda: RMSNormConfig(dim=1),
        description="Norm applied to merged tokens; dim is synced to out_dim.",
    )

    @model_validator(mode="after")
    def _sync_norm_dim(self) -> PatchMergingConfig:
        self.norm = self.norm.model_copy(update={"dim": self.out_dim})
        return self
