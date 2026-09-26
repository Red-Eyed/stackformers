"""Variable-width Transformer encoder preset and configuration helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import torch.nn as nn
from pydantic import BaseModel, Field, model_validator
from returns.result import Failure, Result, Success
from torch import Tensor
from typing_extensions import override

from stackformers._result import unwrap_or_raise
from stackformers.attention.config import (
    AttnBiasConfig,
    DistanceBiasConfig,
    NoAttnBiasConfig,
    SelfAttentionConfig,
)
from stackformers.attention.factory import build_attn_bias
from stackformers.attention.self_attn import SelfAttention
from stackformers.feedforward.config import FeedForwardConfig, SwiGLUConfig
from stackformers.feedforward.factory import build_ff
from stackformers.layers import (
    PostNormTransformerLayer,
    ReorderedNormTransformerLayer,
    SandwichNormTransformerLayer,
    TransformerLayer,
    TransformerLayerBase,
)
from stackformers.norm.config import NormConfig, NormPlacement, RMSNormConfig
from stackformers.norm.factory import build_norm
from stackformers.positional.config import NoPosEncodingConfig, PosEncodingConfig, RoPE1DConfig
from stackformers.positional.factory import build_pos_encoding

if TYPE_CHECKING:
    from stackformers.sequence import SequenceInput

_T = TypeVar("_T")


def _schedule_length_error(name: str, expected: int, actual: int) -> ValueError:
    """Preserve legacy head-dimension diagnostics while naming other schedule failures."""
    match name:
        case "dim_heads":
            return ValueError(
                f"d_models and dim_heads must have equal lengths; got {expected} and {actual}"
            )
        case _:
            return ValueError(
                f"d_models/layers and {name} must have equal lengths;"
                f" expected {expected}, got {actual}"
            )


def _layer_values(value: _T | list[_T], count: int, name: str) -> Result[list[_T], ValueError]:
    """Broadcast a shared setting or reject a schedule with the wrong layer count."""
    if count == 0:
        return Failure(ValueError("d_models must contain at least one block width"))
    match value:
        case list():
            if len(value) != count:
                return Failure(_schedule_length_error(name, count, len(value)))
            return Success(value)
        case _:
            return Success([value] * count)


class VariableWidthEncoderLayerConfig(BaseModel):
    """Describe every configurable component used by one variable-width block."""

    attn: SelfAttentionConfig
    ff: FeedForwardConfig
    norm: NormConfig
    pos_encoding: PosEncodingConfig
    attn_bias: AttnBiasConfig = NoAttnBiasConfig()

    @model_validator(mode="after")
    def _check_dimensions(self) -> VariableWidthEncoderLayerConfig:
        """Reject collaborators whose residual-stream dimensions do not agree."""
        return unwrap_or_raise(self._dimensions_result())

    def _dimensions_result(self) -> Result[VariableWidthEncoderLayerConfig, ValueError]:
        """Return inconsistent collaborator dimensions as data for boundary validation."""
        if self.ff.dim != self.attn.dim:
            return Failure(
                ValueError(f"ff.dim ({self.ff.dim}) must equal attn.dim ({self.attn.dim})")
            )
        if self.norm.dim != self.attn.dim:
            return Failure(
                ValueError(f"norm.dim ({self.norm.dim}) must equal attn.dim ({self.attn.dim})")
            )
        if (
            not isinstance(self.pos_encoding, NoPosEncodingConfig)
            and self.pos_encoding.dim_head != self.attn.dim_head
        ):
            return Failure(
                ValueError(
                    f"pos_encoding.dim_head ({self.pos_encoding.dim_head}) must equal"
                    f" attn.dim_head ({self.attn.dim_head})"
                )
            )
        if (
            isinstance(self.attn_bias, DistanceBiasConfig)
            and self.attn_bias.heads != self.attn.heads
        ):
            return Failure(
                ValueError(
                    f"attn_bias.heads ({self.attn_bias.heads}) must equal"
                    f" attn.heads ({self.attn.heads})"
                )
            )
        return Success(self)


class VariableWidthTransformerEncoderConfig(BaseModel):
    """Configure an ordered list of complete, independently validated blocks."""

    layers: list[VariableWidthEncoderLayerConfig] = Field(min_length=1)
    norm_placement: NormPlacement | list[NormPlacement] = "pre"

    @model_validator(mode="after")
    def _check_placements(self) -> VariableWidthTransformerEncoderConfig:
        """Validate placement schedules before any modules are constructed."""
        unwrap_or_raise(_layer_values(self.norm_placement, len(self.layers), "norm_placement"))
        return self


def variable_width_encoder_config(
    *,
    d_models: list[int],
    dim_heads: int | list[int],
    heads: int | list[int] | None = None,
    causal: bool | list[bool] = False,
    ff_mult: float | list[float] = 4.0,
    dropout: float | list[float] = 0.0,
    norm_placement: NormPlacement | list[NormPlacement] = "pre",
) -> VariableWidthTransformerEncoderConfig:
    """Build the default global-attention RoPE/RMSNorm/SwiGLU preset.

    ``d_models`` defines the layer count. Every other setting accepts a shared
    scalar or a list of exactly that length; mismatched lists raise ``ValueError``.
    Omit ``heads`` to derive query heads exactly as ``d_model // dim_head``;
    this requires each model width divisible by its head dimension. Explicit
    ``heads`` makes the internal attention width independent of the model width.
    Construct ``VariableWidthTransformerEncoderConfig`` directly to configure every
    attention, feed-forward, norm, positional, and attention-bias component per block.
    """
    count = len(d_models)
    head_dims = unwrap_or_raise(_layer_values(dim_heads, count, "dim_heads"))
    query_heads = unwrap_or_raise(_query_head_counts(d_models, head_dims, heads))
    causal_values = unwrap_or_raise(_layer_values(causal, count, "causal"))
    ff_mults = unwrap_or_raise(_layer_values(ff_mult, count, "ff_mult"))
    dropouts = unwrap_or_raise(_layer_values(dropout, count, "dropout"))
    layers = [
        _plain_layer_config(
            d_model,
            layer_heads,
            dim_head,
            layer_causal,
            layer_ff_mult,
            layer_dropout,
        )
        for d_model, layer_heads, dim_head, layer_causal, layer_ff_mult, layer_dropout in zip(
            d_models, query_heads, head_dims, causal_values, ff_mults, dropouts, strict=True
        )
    ]
    return VariableWidthTransformerEncoderConfig(
        layers=layers,
        norm_placement=norm_placement,
    )


def _query_head_counts(
    d_models: list[int], dim_heads: list[int], heads: int | list[int] | None
) -> Result[list[int], ValueError]:
    """Resolve explicit head schedules or retain the legacy exact-inference contract."""
    match heads:
        case None:
            return _validate_width_schedule(d_models, dim_heads).map(
                lambda _: [
                    d_model // dim_head
                    for d_model, dim_head in zip(d_models, dim_heads, strict=True)
                ]
            )
        case _:
            return _layer_values(heads, len(d_models), "heads")


def _validate_width_schedule(
    d_models: list[int], dim_heads: list[int]
) -> Result[tuple[()], ValueError]:
    """Ensure the two per-block schedules define valid attention geometries."""
    if not d_models:
        return Failure(ValueError("d_models must contain at least one block width"))
    if len(d_models) != len(dim_heads):
        return Failure(
            ValueError(
                "d_models and dim_heads must have equal lengths;"
                f" got {len(d_models)} and {len(dim_heads)}"
            )
        )
    for index, (d_model, dim_head) in enumerate(zip(d_models, dim_heads, strict=True)):
        if d_model <= 0 or dim_head <= 0:
            return Failure(
                ValueError(
                    f"d_models[{index}] and dim_heads[{index}] must be positive;"
                    f" got {d_model} and {dim_head}"
                )
            )
        if d_model % dim_head != 0:
            return Failure(
                ValueError(
                    f"d_models[{index}] ({d_model}) must be divisible by"
                    f" dim_heads[{index}] ({dim_head})"
                )
            )
    return Success(())


def _plain_layer_config(
    d_model: int,
    heads: int,
    dim_head: int,
    causal: bool,
    ff_mult: float,
    dropout: float,
) -> VariableWidthEncoderLayerConfig:
    """Build one default preset layer from explicit attention geometry and settings."""
    return VariableWidthEncoderLayerConfig(
        attn=SelfAttentionConfig(
            dim=d_model,
            heads=heads,
            dim_head=dim_head,
            causal=causal,
            dropout=dropout,
        ),
        ff=SwiGLUConfig(dim=d_model, mult=ff_mult, dropout=dropout),
        norm=RMSNormConfig(dim=d_model),
        pos_encoding=RoPE1DConfig(dim_head=dim_head),
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

    @override
    def forward(self, input: SequenceInput) -> SequenceInput:
        """Project only the feature tensor while preserving all sequence metadata."""
        projected = input._replace(x=self.projection(input.x))
        return self.layer(projected)

    if TYPE_CHECKING:
        __call__ = forward


class VariableWidthTransformerEncoder(nn.Module):
    """Run explicit Transformer blocks with learned projections at width changes.

    The input feature width must equal the first layer's ``attn.dim``. The returned tensor
    uses the last layer's width. Adjacent equal-width blocks use an identity; differing
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
        self.final_norm = build_norm(config=config.layers[-1].norm)

    def _build_layers(
        self,
        config: VariableWidthTransformerEncoderConfig,
    ) -> list[_VariableWidthLayer]:
        """Build one projected wrapper per configured Transformer block."""
        previous_dim = config.layers[0].attn.dim
        layers: list[_VariableWidthLayer] = []
        placements = unwrap_or_raise(
            _layer_values(config.norm_placement, len(config.layers), "norm_placement")
        )
        for layer_config, placement in zip(config.layers, placements, strict=True):
            current_dim = layer_config.attn.dim
            projection = self._build_projection(previous_dim, current_dim)
            layer = self._build_transformer_layer(layer_config, placement)
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
        config: VariableWidthEncoderLayerConfig,
        norm_placement: NormPlacement,
    ) -> TransformerLayerBase:
        """Construct one block entirely from its explicit component configurations."""
        self_attn = SelfAttention(
            config=config.attn,
            pos_encoding=build_pos_encoding(config=config.pos_encoding),
            attn_bias=build_attn_bias(config=config.attn_bias),
        )
        feed_forward = build_ff(config=config.ff)
        match norm_placement:
            case "pre":
                return TransformerLayer(
                    self_attn=self_attn,
                    ff=feed_forward,
                    norm_attn=build_norm(config=config.norm),
                    norm_ff=build_norm(config=config.norm),
                )
            case "post":
                return PostNormTransformerLayer(
                    self_attn=self_attn,
                    ff=feed_forward,
                    norm_attn=build_norm(config=config.norm),
                    norm_ff=build_norm(config=config.norm),
                )
            case "sandwich":
                return SandwichNormTransformerLayer(
                    self_attn=self_attn,
                    ff=feed_forward,
                    norm_attn_pre=build_norm(config=config.norm),
                    norm_attn_post=build_norm(config=config.norm),
                    norm_ff_pre=build_norm(config=config.norm),
                    norm_ff_post=build_norm(config=config.norm),
                )
            case "reordered":
                return ReorderedNormTransformerLayer(
                    self_attn=self_attn,
                    ff=feed_forward,
                    norm_attn=build_norm(config=config.norm),
                    norm_ff=build_norm(config=config.norm),
                )

    @override
    def forward(self, input: SequenceInput) -> Tensor:
        """Encode a padded or packed sequence into the configured final width."""
        for layer in self.layers:
            input = layer(input)
        return self.final_norm(input.x)

    if TYPE_CHECKING:
        __call__ = forward
