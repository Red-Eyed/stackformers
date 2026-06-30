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
    monkeypatch.setattr(vb, "_eligible", lambda q: True)


def _attempt(packed: Packed, bias: torch.Tensor | None = None) -> torch.Tensor | None:
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
    monkeypatch.setattr(vb, "_varlen_attn", None)
    monkeypatch.setattr(vb, "_import_error", "varlen_attn could not be imported (no module)")
    with pytest.warns(UserWarning, match="unavailable"):
        out = _attempt(packed)
    assert out is None


def test_attention_bias_warns(
    packed: Packed, force_eligible: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """varlen_attn has no bias slot: an eligible query with a bias falls back with a warning."""
    monkeypatch.setattr(vb, "_varlen_attn", lambda **kw: torch.zeros(NT, H, DH))
    bias = torch.zeros(1, 1, NT, NT)
    with pytest.warns(UserWarning, match="attention bias"):
        out = _attempt(packed, bias=bias)
    assert out is None


def test_signature_error_warns(
    packed: Packed, force_eligible: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed experimental signature raises TypeError, which is caught and falls back."""

    def renamed_kwarg(**kw: object) -> torch.Tensor:
        raise TypeError("unexpected keyword argument 'cu_seq_q'")

    monkeypatch.setattr(vb, "_varlen_attn", renamed_kwarg)
    with pytest.warns(UserWarning, match="call failed"):
        out = _attempt(packed)
    assert out is None


def test_runtime_error_warns(
    packed: Packed, force_eligible: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unsupported-GPU RuntimeError from the call is caught and falls back."""

    def unsupported(**kw: object) -> torch.Tensor:
        raise RuntimeError("varlen_attn requires compute capability >= 8.0")

    monkeypatch.setattr(vb, "_varlen_attn", unsupported)
    with pytest.warns(UserWarning, match="call failed"):
        out = _attempt(packed)
    assert out is None


def test_non_tensor_return_warns(
    packed: Packed, force_eligible: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed return contract (non-Tensor) falls back rather than propagating bad data."""
    monkeypatch.setattr(vb, "_varlen_attn", lambda **kw: ("not", "a", "tensor"))
    with pytest.warns(UserWarning, match="expected Tensor"):
        out = _attempt(packed)
    assert out is None


def test_success_returns_kernel_tensor(
    packed: Packed, force_eligible: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The happy path returns the kernel's tensor unchanged and emits no warning."""
    expected = torch.randn(NT, H, DH)
    monkeypatch.setattr(vb, "_varlen_attn", lambda **kw: expected)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _attempt(packed)
    assert out is expected
