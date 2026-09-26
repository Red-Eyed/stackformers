"""Tests for stackformers/attention/varlen_backend.py.

Covers the fallback contract of ``try_varlen_attn``: it returns the kernel's tensor on the
happy path and ``None`` (with a reason-bearing ``UserWarning`` when the kernel was eligible
but unusable) on every degraded path — absent symbol, attention bias, a runtime/signature
error from the call, or a non-Tensor return. The experimental kernel is faked via monkeypatch
so the logic is exercised without CUDA or the real (possibly-absent) ``varlen_attn``.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from stackformers.attention import varlen_backend as vb
from stackformers.result import Err, Ok, Result
from stackformers.sequence import PackedSequence

NT, H, DH = 5, 2, 8

Packed = tuple[torch.Tensor, torch.Tensor, torch.Tensor, PackedSequence]


@pytest.fixture
def packed() -> Packed:
    """Two packed sequences (lengths 3 and 2) of (nt, h, d) q/k/v plus their PackedSequence."""
    seq = PackedSequence(cu_seqlens=torch.tensor([0, 3, 5], dtype=torch.long), max_seqlen=3)
    q = torch.randn(NT, H, DH)
    k = torch.randn(NT, H, DH)
    v = torch.randn(NT, H, DH)
    return q, k, v, seq


@pytest.fixture
def force_eligible(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the query is CUDA + float16 so the eligibility guard does not short-circuit."""

    def eligible(q: torch.Tensor) -> bool:
        """Admit CPU tensors to exercise the backend invocation boundary."""
        return True

    monkeypatch.setattr(vb, "_eligible", eligible)


def _attempt(packed: Packed, bias: torch.Tensor | None = None) -> torch.Tensor | None:
    """Exercise the existing tensor-or-None warning adapter."""
    q, k, v, seq = packed
    return vb.try_varlen_attn(q, k, v, seq, seq, causal=False, window_size=None, bias=bias)


def test_ineligible_falls_back_silently(packed: Packed, recwarn: pytest.WarningsRecorder) -> None:
    """A plain CPU/float32 query is the expected fallback route — None, and no warning."""
    out = _attempt(packed)
    assert out is None
    assert len(recwarn) == 0


def test_unavailable_kernel_warns(
    packed: Packed, force_eligible: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An eligible query with no imported kernel falls back and reports the import reason."""
    monkeypatch.setattr(vb, "_kernel", Err(ImportError("no module")))
    with pytest.warns(UserWarning, match="unavailable"):
        out = _attempt(packed)
    assert out is None


def test_attention_bias_warns(
    packed: Packed, force_eligible: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """varlen_attn has no bias slot: an eligible query with a bias falls back with a warning."""

    def kernel(**kw: object) -> torch.Tensor:
        """Provide a callable backend while bias eligibility is checked."""
        return torch.zeros(NT, H, DH)

    monkeypatch.setattr(vb, "_kernel", Ok(vb._checked_kernel(kernel)))
    bias = torch.zeros(1, 1, NT, NT)
    with pytest.warns(UserWarning, match="attention bias"):
        out = _attempt(packed, bias=bias)
    assert out is None


def test_signature_error_warns(
    packed: Packed, force_eligible: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed experimental signature raises TypeError, which is caught and falls back."""

    def renamed_kwarg(*, query: torch.Tensor) -> torch.Tensor:
        """Expose an incompatible vendor signature so Python raises the TypeError."""
        return query

    monkeypatch.setattr(vb, "_kernel", Ok(vb._checked_kernel(renamed_kwarg)))
    with pytest.warns(UserWarning, match="call failed.*unexpected keyword argument 'key'"):
        out = _attempt(packed)
    assert out is None


def test_runtime_error_warns(
    packed: Packed, force_eligible: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unsupported-GPU RuntimeError from the call is caught and falls back."""

    def unsupported(**kw: object) -> torch.Tensor:
        """Simulate an unsupported device at the vendor call boundary."""
        raise RuntimeError("varlen_attn requires compute capability >= 8.0")

    monkeypatch.setattr(vb, "_kernel", Ok(vb._checked_kernel(unsupported)))
    with pytest.warns(UserWarning, match="call failed"):
        out = _attempt(packed)
    assert out is None


def test_non_tensor_return_warns(
    packed: Packed, force_eligible: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed return contract (non-Tensor) falls back rather than propagating bad data."""

    def kernel(**kw: object) -> tuple[str, ...]:
        """Return an invalid payload to exercise response validation."""
        return ("not", "a", "tensor")

    monkeypatch.setattr(vb, "_kernel", Ok(vb._checked_kernel(kernel)))
    with pytest.warns(UserWarning, match="expected Tensor"):
        out = _attempt(packed)
    assert out is None


def test_success_returns_kernel_tensor(
    packed: Packed, force_eligible: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The happy path returns the kernel's tensor unchanged and emits no warning."""
    expected = torch.randn(NT, H, DH)

    def kernel(**kw: object) -> torch.Tensor:
        """Return the known tensor to check that the adapter preserves identity."""
        return expected

    monkeypatch.setattr(vb, "_kernel", Ok(vb._checked_kernel(kernel)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _attempt(packed)
    assert out is expected


def test_core_unavailable_preserves_import_error(
    packed: Packed,
    force_eligible: None,
    recwarn: pytest.WarningsRecorder,
) -> None:
    """A missing injected kernel returns its exact cause and leaves warning policy outside."""
    q, k, v, seq = packed
    error = ImportError("missing experimental backend")
    outcome = vb.attempt_varlen_attn(q, k, v, seq, seq, False, None, None, kernel=Err(error))
    assert isinstance(outcome, Err)
    fallback = outcome.error
    assert fallback.kind is vb.FallbackKind.UNAVAILABLE
    assert fallback.detail is error
    assert len(recwarn) == 0


def test_core_rejects_auxiliary_return_without_auxiliary_request(
    packed: Packed,
    force_eligible: None,
    recwarn: pytest.WarningsRecorder,
) -> None:
    """A vendor-compatible tuple remains unexpected for this adapter's tensor-only calls."""
    q, k, v, seq = packed

    def kernel(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the vendor's auxiliary schema despite no auxiliary request."""
        return torch.zeros_like(q), torch.zeros(NT, H)

    outcome = vb.attempt_varlen_attn(
        q, k, v, seq, seq, False, None, None, kernel=Ok(vb._checked_kernel(kernel))
    )
    assert isinstance(outcome, Err)
    assert outcome.error.kind is vb.FallbackKind.INVALID_RETURN
    assert outcome.error.detail == "tuple"
    assert len(recwarn) == 0


@pytest.mark.parametrize("error_type", [RuntimeError, TypeError])
def test_core_call_failure_preserves_kernel_exception(
    packed: Packed,
    force_eligible: None,
    error_type: type[Exception],
    recwarn: pytest.WarningsRecorder,
) -> None:
    """Expected third-party errors retain identity without warnings from core execution."""
    q, k, v, seq = packed
    error = error_type("kernel failure")

    def kernel(**kwargs: object) -> torch.Tensor:
        """Inject an expected experimental failure without needing a CUDA device."""
        raise error

    outcome = vb.attempt_varlen_attn(
        q, k, v, seq, seq, False, None, None, kernel=Ok(vb._checked_kernel(kernel))
    )
    assert isinstance(outcome, Err)
    fallback = outcome.error
    assert fallback.kind is vb.FallbackKind.CALL_FAILED
    assert fallback.detail is error
    assert len(recwarn) == 0


def test_core_does_not_hide_unexpected_kernel_errors(packed: Packed, force_eligible: None) -> None:
    """Unexpected failures still propagate instead of being silently labeled a fallback."""
    q, k, v, seq = packed
    error = ValueError("unexpected kernel bug")

    def kernel(**kwargs: object) -> torch.Tensor:
        """Inject a failure outside the existing backend catch policy."""
        raise error

    with pytest.raises(ValueError, match="unexpected kernel bug") as caught:
        vb.attempt_varlen_attn(
            q, k, v, seq, seq, False, None, None, kernel=Ok(vb._checked_kernel(kernel))
        )
    assert caught.value is error


@pytest.mark.parametrize("succeeds", [True, False])
def test_core_preserves_normalized_kernel_outcome(
    packed: Packed,
    force_eligible: None,
    succeeds: bool,
    recwarn: pytest.WarningsRecorder,
) -> None:
    """Injected normalized providers need no vendor validation or outcome reconstruction."""
    q, k, v, seq = packed
    expected: Result[torch.Tensor, vb.VarlenFallback]
    if succeeds:
        expected = Ok(q)
    else:
        expected = Err(vb.VarlenFallback(vb.FallbackKind.CALL_FAILED, RuntimeError("failed")))

    def kernel(**kwargs: object) -> Result[torch.Tensor, vb.VarlenFallback]:
        """Supply an already normalized outcome directly to the core."""
        return expected

    outcome = vb.attempt_varlen_attn(q, k, v, seq, seq, False, None, None, kernel=Ok(kernel))
    assert outcome is expected
    assert len(recwarn) == 0


@pytest.mark.parametrize("valid_response", [True, False])
def test_loader_normalizes_vendor_response(
    packed: Packed,
    force_eligible: None,
    monkeypatch: pytest.MonkeyPatch,
    valid_response: bool,
) -> None:
    """The actual loader binds the vendor to the Result protocol before core execution."""
    vendor = pytest.importorskip("torch.nn.attention.varlen")
    q, k, v, seq = packed

    def kernel(**kwargs: object) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Simulate both tensor and auxiliary payloads from the imported vendor symbol."""
        return q if valid_response else (q, torch.zeros(NT, H))

    monkeypatch.setattr(vendor, "varlen_attn", kernel)
    outcome = vb.attempt_varlen_attn(q, k, v, seq, seq, False, None, None, kernel=vb._load())
    if valid_response:
        assert isinstance(outcome, Ok)
        assert outcome.value is q
    else:
        assert isinstance(outcome, Err)
        assert outcome.error == vb.VarlenFallback(vb.FallbackKind.INVALID_RETURN, "tuple")
