"""Per-call attention visibility, independent of padding and sequence boundaries."""

from __future__ import annotations

from typing import Literal, TypeAlias, assert_never

import torch
from jaxtyping import Bool
from torch import Tensor

from stackformers.result import Err, Ok, Result

SelfAttentionMask: TypeAlias = Literal["default", "causal", "bidirectional"] | Bool[Tensor, "n n"]
CrossAttentionMask: TypeAlias = Bool[Tensor, "n s"] | None


def valid_pair_mask(
    mask: Tensor, n: int, s: int, device: torch.device
) -> Result[Tensor, TypeError | ValueError]:
    """Admit a shared boolean query/key mask without inspecting tensor values."""
    if mask.dtype != torch.bool:
        return Err(TypeError("attention mask must have boolean dtype (True means allowed)"))
    if mask.ndim != 2 or mask.shape != (n, s):
        return Err(
            ValueError(f"attention mask must have shape ({n}, {s}), got {tuple(mask.shape)}")
        )
    if mask.device != device:
        return Err(ValueError("attention mask must be on the same device as the input"))
    return Ok(mask)


def resolve_self_mask(
    mask: SelfAttentionMask,
    causal: bool,
    window_size: int | None,
    n: int,
    device: torch.device,
) -> Result[tuple[bool, int | None, Tensor | None], TypeError | ValueError]:
    """Replace configured visibility only when the caller supplies an explicit override."""
    match mask:
        case Tensor():
            match valid_pair_mask(mask, n, n, device):
                case Ok(value=allowed):
                    return Ok((False, None, allowed))
                case Err() as failure:
                    return failure
                case _ as unreachable:
                    assert_never(unreachable)
        case "default":
            return Ok((causal, window_size, None))
        case "causal":
            return Ok((True, None, None))
        case "bidirectional":
            return Ok((False, None, None))
        case _:
            return Err(ValueError(f"unknown self-attention mask: {mask!r}"))


def validate_cross_mask(
    mask: CrossAttentionMask, n: int, s: int, device: torch.device
) -> Result[Tensor | None, TypeError | ValueError]:
    """Admit optional cross visibility without inspecting tensor values."""
    match mask:
        case None:
            return Ok(None)
        case _:
            return valid_pair_mask(mask, n, s, device)


def require_context(has_context: bool, mask: CrossAttentionMask) -> None:
    """Reject cross visibility that would otherwise be silently ignored without context."""
    if not has_context and mask is not None:
        raise ValueError("cross_attn_mask requires context")
