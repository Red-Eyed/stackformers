"""The experimental ``varlen_attn`` kernel behind one interface function.

``torch.nn.attention.varlen`` is an experimental API: the symbol is absent in some torch
builds, and its call signature is not yet stable. Everything that touches it lives here —
the guarded import, the eligibility/bias checks, and the call itself — so the rest of the
package depends only on :func:`try_varlen_attn` and never on the volatile import path.
"""

from __future__ import annotations

import warnings
from enum import Enum, auto
from typing import TYPE_CHECKING, NamedTuple, ParamSpec, Protocol, assert_never

import torch
from returns.result import Failure, Result, Success
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Callable

    from stackformers.sequence import PackedSequence

_FALLBACK_NOTE = " Falling back to padded SDPA; performance may be lower."
_P = ParamSpec("_P")


class FallbackKind(Enum):
    """Distinguish normal backend ineligibility from degraded eligible execution."""

    INELIGIBLE = auto()
    UNAVAILABLE = auto()
    UNSUPPORTED_BIAS = auto()
    CALL_FAILED = auto()
    INVALID_RETURN = auto()


class VarlenFallback(NamedTuple):
    """Preserve why the caller should choose padded attention, including kernel causes."""

    kind: FallbackKind
    detail: str | Exception


class _VarlenKernel(Protocol):
    """Normalized kernel capability, independent of the vendor's response schema."""

    def __call__(
        self,
        *,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        cu_seq_q: Tensor,
        cu_seq_k: Tensor,
        max_q: int,
        max_k: int,
        window_size: tuple[int, int],
    ) -> Result[Tensor, VarlenFallback]:
        """Return output or expected fallback detail; unexpected errors may propagate."""
        ...


def _checked_kernel(call: Callable[_P, object]) -> Callable[_P, Result[Tensor, VarlenFallback]]:
    """Normalize vendor responses and failures while retaining its parameter signature."""

    def invoke(*args: _P.args, **kwargs: _P.kwargs) -> Result[Tensor, VarlenFallback]:
        """Translate expected vendor failures and validate its untrusted response."""
        try:
            response: object = call(*args, **kwargs)
        except (RuntimeError, TypeError) as exc:
            return Failure(VarlenFallback(FallbackKind.CALL_FAILED, exc))
        match response:
            case Tensor():
                return Success(response)
            case _:
                return Failure(VarlenFallback(FallbackKind.INVALID_RETURN, type(response).__name__))

    return invoke


def _load() -> Result[_VarlenKernel, ImportError]:
    """Load the optional kernel and preserve its import failure for fallback diagnostics."""
    try:
        from torch.nn.attention.varlen import varlen_attn
    except ImportError as exc:
        return Failure(exc)
    return Success(_checked_kernel(varlen_attn))


_kernel = _load()


def _eligible(q: Tensor) -> bool:
    """Whether this query's device/dtype is one ``varlen_attn`` would handle at all."""
    return q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)


def _window(causal: bool, window_size: int | None) -> tuple[int, int]:
    """Translate the existing attention window convention to the kernel's two bounds."""
    if window_size is None:
        return (-1, 0) if causal else (-1, -1)
    return (window_size, 0) if causal else (window_size // 2, window_size // 2)


def attempt_varlen_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_seq: PackedSequence,
    k_seq: PackedSequence,
    causal: bool,
    window_size: int | None,
    bias: Tensor | None,
    *,
    kernel: Result[_VarlenKernel, ImportError],
) -> Result[Tensor, VarlenFallback]:
    """Attempt the injected kernel; return typed fallback detail without emitting warnings."""
    if not _eligible(q):
        return Failure(VarlenFallback(FallbackKind.INELIGIBLE, "device or dtype is ineligible"))
    match kernel:
        case Failure(error):
            return Failure(VarlenFallback(FallbackKind.UNAVAILABLE, error))
        case _:
            call = kernel.unwrap()
    if bias is not None:
        return Failure(VarlenFallback(FallbackKind.UNSUPPORTED_BIAS, "attention bias"))
    return call(
        query=q,
        key=k,
        value=v,
        cu_seq_q=q_seq.cu_seqlens.to(torch.int32),
        cu_seq_k=k_seq.cu_seqlens.to(torch.int32),
        max_q=q_seq.max_seqlen,
        max_k=k_seq.max_seqlen,
        window_size=_window(causal, window_size),
    )


def _fallback_reason(fallback: VarlenFallback) -> str:
    """Render a typed backend failure for the established warning contract."""
    match fallback.kind:
        case FallbackKind.INELIGIBLE:
            return str(fallback.detail)
        case FallbackKind.UNAVAILABLE:
            return (
                "varlen_attn is unavailable"
                f" (varlen_attn could not be imported ({fallback.detail}))."
            )
        case FallbackKind.UNSUPPORTED_BIAS:
            return "varlen_attn does not support attention bias."
        case FallbackKind.CALL_FAILED:
            return f"varlen_attn call failed ({type(fallback.detail).__name__}: {fallback.detail})."
        case FallbackKind.INVALID_RETURN:
            return f"varlen_attn returned {fallback.detail}, expected Tensor."
        case _:
            assert_never(fallback.kind)


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
    """Preserve tensor/None results and eligible-backend warnings at the API boundary.

    CPU and unsupported dtypes select padded attention silently. Eligible kernel failures
    retain their original diagnostic and warn before the caller takes the padded fallback.
    """
    outcome = attempt_varlen_attn(q, k, v, q_seq, k_seq, causal, window_size, bias, kernel=_kernel)
    match outcome:
        case Failure(fallback):
            if fallback.kind is not FallbackKind.INELIGIBLE:
                warnings.warn(_fallback_reason(fallback) + _FALLBACK_NOTE, stacklevel=2)
            return None
        case _:
            return outcome.unwrap()
