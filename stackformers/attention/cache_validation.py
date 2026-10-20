"""Pure cache admission checks with typed failures and no exception policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from stackformers.attention.bias import NoAttnBias
from stackformers.result import Err, Ok, Result

if TYPE_CHECKING:
    from stackformers.attention.packed_self_attn import PackedSelfAttention
    from stackformers.attention.self_attn import SelfAttention


def supported_self_attention(
    attention: SelfAttention | PackedSelfAttention,
) -> Result[SelfAttention | PackedSelfAttention, NotImplementedError]:
    """Return the supported cached equation or explain its unsupported features."""
    if not attention.config.causal or attention.config.window_size is not None:
        return Err(
            NotImplementedError(
                "cached self-attention currently supports global causal attention only"
            )
        )
    match attention.attn_bias:
        case NoAttnBias():
            return Ok(attention)
        case _:
            return Err(
                NotImplementedError("cached self-attention currently supports NoAttnBias only")
            )


def single_token(token_count: int) -> Result[tuple[()], ValueError]:
    """Admit exactly one target token without raising on unsupported counts."""
    if token_count != 1:
        return Err(ValueError("cached self-attention requires exactly one target token"))
    return Ok(())


def matching_layer_counts(
    cross_layers: int, self_layers: int, expected_layers: int
) -> Result[tuple[()], ValueError]:
    """Return mismatched dense-cache layer counts as data, in public validation order."""
    if cross_layers != expected_layers:
        return Err(
            ValueError(f"cross cache has {cross_layers} layers, decoder has {expected_layers}")
        )
    if self_layers != expected_layers:
        return Err(
            ValueError(f"self cache has {self_layers} layers, decoder has {expected_layers}")
        )
    return Ok(())
