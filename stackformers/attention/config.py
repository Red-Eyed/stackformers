from __future__ import annotations

import warnings

from pydantic import BaseModel, Field, model_validator

_ALIGN = 64  # tensor-core alignment for FP16/BF16


def _validate_attn_dims(dim: int, heads: int, dim_head: int, kv_heads: int | None) -> None:
    if dim % heads != 0:
        raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
    kv_h = kv_heads if kv_heads is not None else heads
    if heads % kv_h != 0:
        raise ValueError(f"heads ({heads}) must be divisible by kv_heads ({kv_h})")
    if dim_head % _ALIGN != 0:
        warnings.warn(
            f"dim_head={dim_head} is not a multiple of {_ALIGN}. "
            "Unaligned head dimension reduces GPU throughput on tensor-core hardware. "
            f"Nearest aligned values: {(dim_head // _ALIGN) * _ALIGN} or "
            f"{(dim_head // _ALIGN + 1) * _ALIGN}.",
            UserWarning,
            stacklevel=3,
        )
    projected = heads * dim_head
    if projected != dim:
        warnings.warn(
            f"heads * dim_head = {heads} * {dim_head} = {projected}, "
            f"which does not equal dim={dim}. "
            "The Q/K/V projections are non-square — this is valid but may not be intentional. "
            f"Standard transformers use dim_head = dim // heads = {dim // heads}.",
            UserWarning,
            stacklevel=3,
        )


class SelfAttentionConfig(BaseModel):
    dim: int = Field(
        gt=0,
        description="Model (embedding) dimension — input and output width of the attention sublayer.",  # noqa: E501
    )
    heads: int = Field(default=8, gt=0, description="Number of query heads.")
    dim_head: int = Field(
        default=64,
        gt=0,
        description=(
            "Dimension per head. Output projection width is heads * dim_head,"
            " which need not equal dim (non-square projections are valid)."
        ),
    )
    kv_heads: int | None = Field(
        default=None,
        description=(
            "Number of key/value heads. None = equal to heads (standard MHA)."
            " Set to 1 for MQA or any divisor of heads for GQA."
        ),
    )
    causal: bool = Field(
        default=False,
        description=(
            "Apply a causal mask so each position attends only to itself and earlier positions."
        ),
    )
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Attention weight dropout probability, applied during training only.",
    )
    window_size: int | None = Field(
        default=None,
        description=(
            "Sliding-window width. None = global attention."
            " A positive integer restricts each token to a local window of that width."
        ),
    )
    qk_norm: bool = Field(
        default=False,
        description=(
            "Apply RMSNorm to queries and keys (in head space) before the dot product."
            " Stabilises logit scale at large depth or large dim_head."
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> "SelfAttentionConfig":
        _validate_attn_dims(self.dim, self.heads, self.dim_head, self.kv_heads)
        return self

    @property
    def effective_kv_heads(self) -> int:
        return self.kv_heads if self.kv_heads is not None else self.heads

    @property
    def groups(self) -> int:
        return self.heads // self.effective_kv_heads


class CrossAttentionConfig(BaseModel):
    dim: int = Field(
        gt=0,
        description="Model (embedding) dimension — input and output width of the attention sublayer.",  # noqa: E501
    )
    heads: int = Field(default=8, gt=0, description="Number of query heads.")
    dim_head: int = Field(
        default=64,
        gt=0,
        description=(
            "Dimension per head. Output projection width is heads * dim_head,"
            " which need not equal dim (non-square projections are valid)."
        ),
    )
    kv_heads: int | None = Field(
        default=None,
        description=(
            "Number of key/value heads. None = equal to heads (standard MHA)."
            " Set to 1 for MQA or any divisor of heads for GQA."
        ),
    )
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Attention weight dropout probability, applied during training only.",
    )
    qk_norm: bool = Field(
        default=False,
        description=(
            "Apply RMSNorm to queries and keys (in head space) before the dot product."
            " Stabilises logit scale at large depth or large dim_head."
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> "CrossAttentionConfig":
        _validate_attn_dims(self.dim, self.heads, self.dim_head, self.kv_heads)
        return self

    @property
    def effective_kv_heads(self) -> int:
        return self.kv_heads if self.kv_heads is not None else self.heads

    @property
    def groups(self) -> int:
        return self.heads // self.effective_kv_heads
