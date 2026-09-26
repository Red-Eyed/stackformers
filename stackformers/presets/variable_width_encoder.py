"""Variable-width Transformer encoder preset and configuration helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    norm_placement: NormPlacement = "pre"


def variable_width_encoder_config(
    *,
    d_models: list[int],
    dim_heads: list[int],
    causal: bool = False,
    ff_mult: float = 4.0,
    dropout: float = 0.0,
    norm_placement: NormPlacement = "pre",
) -> VariableWidthTransformerEncoderConfig:
    """Build the default global-attention RoPE/RMSNorm/SwiGLU preset.

    Each pair ``(d_models[i], dim_heads[i])`` specifies one Transformer block.
    The number of query heads is derived exactly as ``d_model // dim_head``.
    Construct ``VariableWidthTransformerEncoderConfig`` directly to configure every
    attention, feed-forward, norm, positional, and attention-bias component per block.
    """
    unwrap_or_raise(_validate_width_schedule(d_models, dim_heads))
    layers = [
        _plain_layer_config(
            d_model,
            dim_head,
            causal,
            ff_mult,
            dropout,
        )
        for d_model, dim_head in zip(d_models, dim_heads, strict=True)
    ]
    return VariableWidthTransformerEncoderConfig(
        layers=layers,
        norm_placement=norm_placement,
    )


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
    dim_head: int,
    causal: bool,
    ff_mult: float,
    dropout: float,
) -> VariableWidthEncoderLayerConfig:
    """Expand one width pair into the default preset's explicit component configs."""
    return VariableWidthEncoderLayerConfig(
        attn=SelfAttentionConfig(
            dim=d_model,
            heads=d_model // dim_head,
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
        for layer_config in config.layers:
            current_dim = layer_config.attn.dim
            projection = self._build_projection(previous_dim, current_dim)
            layer = self._build_transformer_layer(layer_config, config.norm_placement)
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
