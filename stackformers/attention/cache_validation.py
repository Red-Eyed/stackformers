"""Pure cache admission checks with typed failures and no exception policy."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from stackformers.attention.bias import NoAttnBias
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.packed_cross_attn import PackedCrossAttention
from stackformers.attention.packed_self_attn import PackedSelfAttention
from stackformers.attention.self_attn import SelfAttention
from stackformers.result import Err, Ok, Result

if TYPE_CHECKING:
    from stackformers.attention.protocols import CrossAttn, SelfAttn


def cacheable_self_attention(
    attention: SelfAttn,
) -> Result[SelfAttention | PackedSelfAttention, TypeError]:
    """Admit only built-in equations; subclass forward overrides cannot be inferred safely."""
    match attention:
        case SelfAttention() if type(attention) is SelfAttention:
            return Ok(attention)
        case PackedSelfAttention() if type(attention) is PackedSelfAttention:
            return Ok(attention)
        case _:
            return Err(
                TypeError(
                    f"cached decoding does not support {type(attention).__name__} self-attention"
                )
            )


def cacheable_cross_attention(
    attention: CrossAttn,
) -> Result[CrossAttention | PackedCrossAttention, TypeError]:
    """Reject unknown cross-attention equations before reusing projection attributes."""
    match attention:
        case CrossAttention() if type(attention) is CrossAttention:
            return Ok(attention)
        case PackedCrossAttention() if type(attention) is PackedCrossAttention:
            return Ok(attention)
        case _:
            return Err(
                TypeError(
                    f"cached decoding does not support {type(attention).__name__} cross-attention"
                )
            )


def supported_self_attention(
    attention: SelfAttn,
) -> Result[SelfAttention | PackedSelfAttention, TypeError | NotImplementedError]:
    """Return the supported cached equation or explain its unsupported features."""
    admission = cacheable_self_attention(attention)
    match admission:
        case Err() as failure:
            return failure
        case Ok(value=attention):
            pass
        case _:
            assert_never(admission)
    if not attention.config.causal or attention.config.window_size is not None:
        return Err(
            NotImplementedError(
                "cached self-attention currently supports global causal attention only"
            )
        )
    match attention.attn_bias:
        case NoAttnBias() if type(attention.attn_bias) is NoAttnBias:
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
