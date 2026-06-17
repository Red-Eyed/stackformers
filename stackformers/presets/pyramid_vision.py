from __future__ import annotations

import torch.nn as nn
from einops import rearrange
from jaxtyping import Float
from pydantic import BaseModel, Field, model_validator
from torch import Tensor

from stackformers.feedforward.config import FeedForwardConfig, SwiGLUConfig
from stackformers.feedforward.factory import build_ff
from stackformers.norm.config import NormConfig, RMSNormConfig
from stackformers.norm.factory import build_norm
from stackformers.positional.config import PosEncodingConfig, RoPE2DConfig
from stackformers.positional.factory import build_pos_encoding
from stackformers.spatial.config import (
    PatchMergingConfig,
    SpatialAttnConfig,
    SpatialReductionAttentionConfig,
    WindowAttention2DConfig,
)
from stackformers.spatial.factory import build_patch_merging, build_spatial_attn
from stackformers.spatial.input import SpatialInput
from stackformers.spatial.layer import SpatialTransformerLayer


class PyramidStageConfig(BaseModel):
    """One pyramid stage: `depth` spatial-attention layers at a fixed grid and width."""

    attn: SpatialAttnConfig
    ff: FeedForwardConfig
    norm: NormConfig
    pos_encoding: PosEncodingConfig
    depth: int = Field(gt=0)

    @model_validator(mode="after")
    def _dims_agree(self) -> PyramidStageConfig:
        if not self.attn.dim == self.ff.dim == self.norm.dim:
            raise ValueError(
                f"stage dims disagree: attn={self.attn.dim}, ff={self.ff.dim}, norm={self.norm.dim}"
            )
        return self


class PyramidVisionConfig(BaseModel):
    """A multi-scale vision backbone: stages joined by patch-merging downsamples.

    inter_stage[i] merges the output of stage i into the input of stage i+1, so there is
    exactly one fewer merge than there are stages.
    """

    stages: list[PyramidStageConfig] = Field(min_length=1)
    inter_stage: list[PatchMergingConfig]

    @model_validator(mode="after")
    def _stages_and_merges_chain(self) -> PyramidVisionConfig:
        if len(self.inter_stage) != len(self.stages) - 1:
            raise ValueError(
                f"need {len(self.stages) - 1} inter-stage merges, got {len(self.inter_stage)}"
            )
        for i, merge in enumerate(self.inter_stage):
            if merge.in_dim != self.stages[i].attn.dim:
                raise ValueError(f"merge {i} in_dim {merge.in_dim} != stage {i} dim")
            if merge.out_dim != self.stages[i + 1].attn.dim:
                raise ValueError(f"merge {i} out_dim {merge.out_dim} != stage {i + 1} dim")
        return self


def pyramid_vision_config(
    *,
    dims: tuple[int, ...] = (64, 128, 320, 512),
    depths: tuple[int, ...] = (3, 4, 6, 3),
    heads: tuple[int, ...] = (1, 2, 5, 8),
    sr_ratios: tuple[int, ...] = (8, 4, 2, 1),
    window: int = 8,
    window_stages: int = 2,
    ff_mult: float = 4.0,
    dropout: float = 0.0,
) -> PyramidVisionConfig:
    """PVTv2-style hybrid backbone: 2-D window attention in the first `window_stages`
    high-resolution stages, spatially-reduced global attention in the rest.

    Defaults are tuned for 1024² images patch-embedded with an 8×8 stem (stage-1 grid
    128×128); `window=8` and the sr_ratios divide that grid cleanly. RoPE 2-D positions
    every stage; norm defaults to RMSNorm.

    Papers: PVT/PVTv2 (Wang et al., 2021/2022, https://arxiv.org/abs/2102.12122,
    https://arxiv.org/abs/2106.13797) for the SRA pyramid; Swin (Liu et al., 2021,
    https://arxiv.org/abs/2103.14030) for the window-attention stages.
    """
    n = len(dims)
    stages: list[PyramidStageConfig] = []
    for i in range(n):
        dim, dim_head = dims[i], dims[i] // heads[i]
        attn: SpatialAttnConfig = (
            WindowAttention2DConfig(
                dim=dim, heads=heads[i], dim_head=dim_head, window=window, dropout=dropout
            )
            if i < window_stages
            else SpatialReductionAttentionConfig(
                dim=dim, heads=heads[i], dim_head=dim_head, reduction=sr_ratios[i], dropout=dropout
            )
        )
        stages.append(
            PyramidStageConfig(
                attn=attn,
                ff=SwiGLUConfig(dim=dim, mult=ff_mult, dropout=dropout),
                norm=RMSNormConfig(dim=dim),
                pos_encoding=RoPE2DConfig(dim_head=dim_head),
                depth=depths[i],
            )
        )
    merges = [PatchMergingConfig(in_dim=dims[i], out_dim=dims[i + 1]) for i in range(n - 1)]
    return PyramidVisionConfig(stages=stages, inter_stage=merges)


class _PyramidStage(nn.Module):
    """A stack of SpatialTransformerLayers at a fixed grid; SpatialInput → SpatialInput."""

    def __init__(self, config: PyramidStageConfig) -> None:
        super().__init__()
        pos = build_pos_encoding(config.pos_encoding)
        self.layers = nn.ModuleList(
            SpatialTransformerLayer(
                attn=build_spatial_attn(config.attn, pos),
                ff=build_ff(config.ff),
                norm_attn=build_norm(config.norm),
                norm_ff=build_norm(config.norm),
            )
            for _ in range(config.depth)
        )

    def forward(self, input: SpatialInput) -> SpatialInput:
        for layer in self.layers:
            input = layer(input)
        return input


def _to_feature_map(input: SpatialInput) -> Float[Tensor, "b d h w"]:
    h, w = input.grid
    return rearrange(input.x, "b (h w) d -> b d h w", h=h, w=w)


class PyramidVisionBackbone(nn.Module):
    """Multi-scale vision backbone (PVTv2-style hybrid).

    Consumes a SpatialInput from your own patch-embed stem and returns one feature map per
    stage as (b, d, H, W) — the multi-scale pyramid a detection/segmentation head expects.

    Paper: "PVTv2: Improved Baselines with Pyramid Vision Transformer", Wang et al., 2022 —
    https://arxiv.org/abs/2106.13797
    """

    def __init__(self, config: PyramidVisionConfig) -> None:
        super().__init__()
        self.stages = nn.ModuleList(_PyramidStage(stage) for stage in config.stages)
        self.merges = nn.ModuleList(build_patch_merging(m) for m in config.inter_stage)

    def forward(self, input: SpatialInput) -> list[Float[Tensor, "b d h w"]]:
        """Shape flow (4-stage example, stem grid H₀×W₀, dims d₀..d₃):

        stem in   (b, H₀·W₀, d₀)
        stage i   (b, Hᵢ·Wᵢ, dᵢ)        emit feature map (b, dᵢ, Hᵢ, Wᵢ)
        merge i   (b, Hᵢ₊₁·Wᵢ₊₁, dᵢ₊₁)  grid halved, width widened
        returns   [(b, dᵢ, Hᵢ, Wᵢ) for each stage]  — the feature pyramid
        """
        features: list[Tensor] = []
        x = input
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(_to_feature_map(x))
            if i < len(self.merges):
                x = self.merges[i](x)
        return features
