"""Pure cache admission checks with typed failures and no exception policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from returns.result import Failure, Result, Success

from stackformers.attention.bias import NoAttnBias

if TYPE_CHECKING:
    from stackformers.attention.self_attn import SelfAttention


def supported_self_attention(
    attention: SelfAttention,
) -> Result[SelfAttention, NotImplementedError]:
    """Return the supported cached equation or explain its unsupported features."""
    if not attention.config.causal or attention.config.window_size is not None:
        return Failure(
            NotImplementedError(
                "cached self-attention currently supports global causal attention only"
            )
        )
    match attention.attn_bias:
        case NoAttnBias():
            return Success(attention)
        case _:
            return Failure(
                NotImplementedError("cached self-attention currently supports NoAttnBias only")
            )


def single_token(token_count: int) -> Result[tuple[()], ValueError]:
    """Admit exactly one target token without raising on unsupported counts."""
    if token_count != 1:
        return Failure(ValueError("cached self-attention requires exactly one target token"))
    return Success(())


def matching_layer_counts(
    cross_layers: int, self_layers: int, expected_layers: int
) -> Result[tuple[()], ValueError]:
    """Return mismatched dense-cache layer counts as data, in public validation order."""
    if cross_layers != expected_layers:
        return Failure(
            ValueError(f"cross cache has {cross_layers} layers, decoder has {expected_layers}")
        )
    if self_layers != expected_layers:
        return Failure(
            ValueError(f"self cache has {self_layers} layers, decoder has {expected_layers}")
        )
    return Success(())
