"""Variable-width Transformer encoder preset and configuration helpers."""

from __future__ import annotations

import torch.nn as nn
from pydantic import BaseModel, Field, model_validator
from torch import Tensor

from stackformers.attention.config import NoAttnBiasConfig, SelfAttentionConfig
from stackformers.attention.factory import build_attn_bias
from stackformers.attention.self_attn import SelfAttention
from stackformers.feedforward.config import SwiGLUConfig
from stackformers.feedforward.factory import build_ff
from stackformers.layers import (
    PostNormTransformerLayer,
    ReorderedNormTransformerLayer,
    SandwichNormTransformerLayer,
    TransformerLayer,
    TransformerLayerBase,
)
from stackformers.norm.config import NormPlacement, RMSNormConfig
from stackformers.norm.factory import build_norm
from stackformers.positional.config import RoPE1DConfig
from stackformers.positional.factory import build_pos_encoding
from stackformers.sequence import SequenceInput


class VariableWidthTransformerEncoderConfig(BaseModel):
    """Configure model-width and head-width arrays for a Transformer block stack."""

    d_models: list[int]
    dim_heads: list[int]
    causal: bool = False
    ff_mult: float = Field(default=4.0, gt=0.0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    norm_placement: NormPlacement = "pre"

    @model_validator(mode="after")
    def _check_width_schedule(self) -> VariableWidthTransformerEncoderConfig:
        """Reject schedules that cannot define valid attention geometry."""
        _validate_width_schedule(self.d_models, self.dim_heads)
        return self


def variable_width_encoder_config(
    d_models: list[int],
    dim_heads: list[int],
    *,
    causal: bool = False,
    ff_mult: float = 4.0,
    dropout: float = 0.0,
    norm_placement: NormPlacement = "pre",
) -> VariableWidthTransformerEncoderConfig:
    """Build a global-attention RoPE/RMSNorm/SwiGLU variable-width preset.

    Each pair ``(d_models[i], dim_heads[i])`` specifies one Transformer block.
    The number of query heads is derived exactly as ``d_model // dim_head``.
    """
    return VariableWidthTransformerEncoderConfig(
        d_models=d_models,
        dim_heads=dim_heads,
        causal=causal,
        ff_mult=ff_mult,
        dropout=dropout,
        norm_placement=norm_placement,
    )


def _validate_width_schedule(d_models: list[int], dim_heads: list[int]) -> None:
    """Ensure the two per-block schedules define valid attention geometries."""
    if not d_models:
        raise ValueError("d_models must contain at least one block width")
    if len(d_models) != len(dim_heads):
        raise ValueError(
            f"d_models and dim_heads must have equal lengths; got {len(d_models)} and"
            f" {len(dim_heads)}"
        )
    for index, (d_model, dim_head) in enumerate(zip(d_models, dim_heads, strict=True)):
        if d_model <= 0 or dim_head <= 0:
            raise ValueError(
                f"d_models[{index}] and dim_heads[{index}] must be positive;"
                f" got {d_model} and {dim_head}"
            )
        if d_model % dim_head != 0:
            raise ValueError(
                f"d_models[{index}] ({d_model}) must be divisible by dim_heads[{index}]"
                f" ({dim_head})"
            )


class _VariableWidthLayer(nn.Module):
    """Adapt the residual width before applying one ordinary Transformer block."""

    def __init__(
        self,
        projection: nn.Module,
        layer: TransformerLayerBase,
    ) -> None:
        """Store the stage-boundary projection and width-preserving block."""
        super().__init__()
        self.projection = projection
        self.layer = layer

    def forward(self, input: SequenceInput) -> SequenceInput:
        """Project only the feature tensor while preserving all sequence metadata."""
        projected = input._replace(x=self.projection(input.x))
        return self.layer(projected)


class VariableWidthTransformerEncoder(nn.Module):
    """Run explicit Transformer blocks with learned projections at width changes.

    The input feature width must equal ``config.d_models[0]``. The returned tensor uses
    the last configured width. Adjacent equal-width blocks use an identity; differing
    widths use a bias-free learned projection before the incoming block.

    Reference: Wu et al., "Variable-Width Transformers" (2026),
    https://arxiv.org/abs/2606.18246. That work uses parameter-free residual resizing;
    this preset's learned projections are a related experimental transition choice,
    not an exact reproduction.
    """

    def __init__(self, config: VariableWidthTransformerEncoderConfig) -> None:
        """Build all layer-local collaborators and stage-boundary projections."""
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(self._build_layers(config))
        self.final_norm = build_norm(RMSNormConfig(dim=config.d_models[-1]))

    def _build_layers(
        self,
        config: VariableWidthTransformerEncoderConfig,
    ) -> list[_VariableWidthLayer]:
        """Build one projected wrapper per configured Transformer block."""
        previous_dim = config.d_models[0]
        layers: list[_VariableWidthLayer] = []
        for d_model, dim_head in zip(config.d_models, config.dim_heads, strict=True):
            current_dim = d_model
            projection = self._build_projection(previous_dim, current_dim)
            layer = self._build_transformer_layer(d_model, dim_head, config)
            layers.append(_VariableWidthLayer(projection, layer))
            previous_dim = current_dim
        return layers

    def _build_projection(self, input_dim: int, output_dim: int) -> nn.Module:
        """Return an identity within a stage or a learned projection across stages."""
        if input_dim == output_dim:
            return nn.Identity()
        return nn.Linear(input_dim, output_dim, bias=False)

    def _build_transformer_layer(
        self,
        d_model: int,
        dim_head: int,
        config: VariableWidthTransformerEncoderConfig,
    ) -> TransformerLayerBase:
        """Construct one block from its indexed dimensions and shared preset settings."""
        attn_config = SelfAttentionConfig(
            dim=d_model,
            heads=d_model // dim_head,
            dim_head=dim_head,
            causal=config.causal,
            dropout=config.dropout,
        )
        self_attn = SelfAttention(
            config=attn_config,
            pos_encoding=build_pos_encoding(RoPE1DConfig(dim_head=dim_head)),
            attn_bias=build_attn_bias(NoAttnBiasConfig()),
        )
        feed_forward = build_ff(
            SwiGLUConfig(dim=d_model, mult=config.ff_mult, dropout=config.dropout)
        )
        norm_config = RMSNormConfig(dim=d_model)
        match config.norm_placement:
            case "pre":
                return TransformerLayer(
                    self_attn,
                    feed_forward,
                    build_norm(norm_config),
                    build_norm(norm_config),
                )
            case "post":
                return PostNormTransformerLayer(
                    self_attn,
                    feed_forward,
                    build_norm(norm_config),
                    build_norm(norm_config),
                )
            case "sandwich":
                return SandwichNormTransformerLayer(
                    self_attn,
                    feed_forward,
                    build_norm(norm_config),
                    build_norm(norm_config),
                    build_norm(norm_config),
                    build_norm(norm_config),
                )
            case "reordered":
                return ReorderedNormTransformerLayer(
                    self_attn,
                    feed_forward,
                    build_norm(norm_config),
                    build_norm(norm_config),
                )

    def forward(self, input: SequenceInput) -> Tensor:
        """Encode a padded or packed sequence into the configured final width."""
        for layer in self.layers:
            input = layer(input)
        return self.final_norm(input.x)
