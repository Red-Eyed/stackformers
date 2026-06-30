"""The experimental ``varlen_attn`` kernel behind one interface function.

``torch.nn.attention.varlen`` is an experimental API: the symbol is absent in some torch
builds, and its call signature is not yet stable. Everything that touches it lives here —
the guarded import, the eligibility/bias checks, and the call itself — so the rest of the
package depends only on :func:`try_varlen_attn` and never on the volatile import path.
"""

from __future__ import annotations

import warnings

import torch
from torch import Tensor

from stackformers.sequence import PackedSequence

_FALLBACK_NOTE = " Falling back to padded SDPA; performance may be lower."


def _load():
    try:
        from torch.nn.attention.varlen import varlen_attn
    except ImportError as exc:
        return None, f"varlen_attn could not be imported ({exc})"
    return varlen_attn, None


_varlen_attn, _import_error = _load()


def _eligible(q: Tensor) -> bool:
    """Whether this query's device/dtype is one ``varlen_attn`` would handle at all."""
    return q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)


def _window(causal: bool, window_size: int | None) -> tuple[int, int]:
    if window_size is None:
        return (-1, 0) if causal else (-1, -1)
    return (window_size, 0) if causal else (window_size // 2, window_size // 2)


def _skip(reason: str) -> None:
    warnings.warn(reason + _FALLBACK_NOTE, stacklevel=3)
    return None


def try_varlen_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_seq: PackedSequence,
    k_seq: PackedSequence,
    causal: bool,
    window_size: int | None,
    bias: Tensor | None,
) -> Tensor | None:
    """Attempt varlen attention for a packed batch; return ``None`` to use the SDPA fallback.

    q/k/v are ``(nt, h, d)``; the result is the packed ``(nt_q, h, d)`` tensor. ``None`` means
    the caller should run the padded SDPA path instead. When varlen *would* apply (CUDA +
    float16/bfloat16) but cannot, a ``UserWarning`` explains why: an absent kernel, an
    attention bias (varlen_attn has no bias slot), or a runtime/signature error from the call
    (an unsupported GPU or a changed experimental API). The expected CPU/float32/export route
    returns ``None`` silently — that is normal, not a failure.
    """
    if not _eligible(q):
        return None
    if _varlen_attn is None:
        return _skip(f"varlen_attn is unavailable ({_import_error}).")
    if bias is not None:
        return _skip("varlen_attn does not support attention bias.")
    try:
        result = _varlen_attn(
            query=q,
            key=k,
            value=v,
            cu_seq_q=q_seq.cu_seqlens.to(torch.int32),
            cu_seq_k=k_seq.cu_seqlens.to(torch.int32),
            max_q=q_seq.max_seqlen,
            max_k=k_seq.max_seqlen,
            window_size=_window(causal, window_size),
        )
    except (RuntimeError, TypeError) as exc:
        return _skip(f"varlen_attn call failed ({type(exc).__name__}: {exc}).")
    if not isinstance(result, Tensor):
        return _skip(f"varlen_attn returned {type(result).__name__}, expected Tensor.")
    return result
