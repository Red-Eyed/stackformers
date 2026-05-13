from __future__ import annotations

import pytest
import torch
import torch.export

from stackformers.positional.config import YaRNConfig
from stackformers.positional.none import NoPosEncoding
from stackformers.positional.rope1d import RotaryEmbedding1D
from stackformers.positional.rope2d import RotaryEmbedding2D
from tests.conftest import atol

B, H, N, DH = 2, 4, 8, 32


@pytest.fixture
def qk(device_dtype: tuple[torch.device, torch.dtype]) -> tuple[torch.Tensor, torch.Tensor]:
    device, dtype = device_dtype
    return (
        torch.randn(B, H, N, DH, device=device, dtype=dtype),
        torch.randn(B, H, N, DH, device=device, dtype=dtype),
    )


@pytest.fixture
def rope1d(device_dtype: tuple[torch.device, torch.dtype]) -> RotaryEmbedding1D:
    device, dtype = device_dtype
    return RotaryEmbedding1D(dim_head=DH).to(device=device, dtype=dtype)


@pytest.fixture
def rope2d(device_dtype: tuple[torch.device, torch.dtype]) -> RotaryEmbedding2D:
    device, dtype = device_dtype
    return RotaryEmbedding2D(dim_head=DH).to(device=device, dtype=dtype)


def test_no_pos_encoding_passthrough(
    qk: tuple[torch.Tensor, torch.Tensor],
) -> None:
    q, k = qk
    q_out, k_out = NoPosEncoding()(q, k)
    assert torch.equal(q_out, q)
    assert torch.equal(k_out, k)


def test_rope1d_output_shape(
    rope1d: RotaryEmbedding1D,
    qk: tuple[torch.Tensor, torch.Tensor],
) -> None:
    q, k = qk
    q_out, k_out = rope1d(q, k)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_rope1d_cross_attn_different_lengths(
    rope1d: RotaryEmbedding1D,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q = torch.randn(B, H, 6, DH, device=device, dtype=dtype)
    k = torch.randn(B, H, 12, DH, device=device, dtype=dtype)
    q_out, k_out = rope1d(q, k)
    assert q_out.shape == (B, H, 6, DH)
    assert k_out.shape == (B, H, 12, DH)


def test_rope1d_different_positions_produce_different_output(
    rope1d: RotaryEmbedding1D,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q = torch.ones(1, 1, 4, DH, device=device, dtype=dtype)
    k = torch.ones(1, 1, 4, DH, device=device, dtype=dtype)
    q_out, _ = rope1d(q, k)
    assert not torch.allclose(q_out[:, :, 0], q_out[:, :, 1])


def test_rope1d_preserves_norms(
    rope1d: RotaryEmbedding1D,
    qk: tuple[torch.Tensor, torch.Tensor],
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    _, dtype = device_dtype
    q, k = qk
    q_out, k_out = rope1d(q, k)
    tol = atol(dtype)
    assert torch.allclose(q.norm(dim=-1), q_out.norm(dim=-1), atol=tol)
    assert torch.allclose(k.norm(dim=-1), k_out.norm(dim=-1), atol=tol)


def test_rope2d_output_shape(
    rope2d: RotaryEmbedding2D,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q = torch.randn(B, H, N, DH, device=device, dtype=dtype)
    k = torch.randn(B, H, N, DH, device=device, dtype=dtype)
    row_ids = torch.arange(N, dtype=dtype, device=device)
    col_ids = torch.arange(N, dtype=dtype, device=device)
    q_out, k_out = rope2d(q, k, row_ids, col_ids)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


# --- YaRN scaling ---

_YARN = YaRNConfig(scale=4.0, original_max_seq_len=512)


@pytest.fixture
def yarn_rope(device_dtype: tuple[torch.device, torch.dtype]) -> RotaryEmbedding1D:
    device, dtype = device_dtype
    return RotaryEmbedding1D(dim_head=DH, yarn=_YARN).to(device=device, dtype=dtype)


def test_yarn_output_shape(
    yarn_rope: RotaryEmbedding1D,
    qk: tuple[torch.Tensor, torch.Tensor],
) -> None:
    q, k = qk
    q_out, k_out = yarn_rope(q, k)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_yarn_preserves_norms(
    yarn_rope: RotaryEmbedding1D,
    qk: tuple[torch.Tensor, torch.Tensor],
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    _, dtype = device_dtype
    q, k = qk
    q_out, k_out = yarn_rope(q, k)
    tol = atol(dtype)
    assert torch.allclose(q.norm(dim=-1), q_out.norm(dim=-1), atol=tol)
    assert torch.allclose(k.norm(dim=-1), k_out.norm(dim=-1), atol=tol)


def test_yarn_differs_from_base_rope(
    qk: tuple[torch.Tensor, torch.Tensor],
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    base = RotaryEmbedding1D(dim_head=DH).to(device=device, dtype=dtype)
    yarn = RotaryEmbedding1D(dim_head=DH, yarn=_YARN).to(device=device, dtype=dtype)
    q, k = qk
    q_base, _ = base(q, k)
    q_yarn, _ = yarn(q, k)
    assert not torch.allclose(q_base, q_yarn)


# --- torch.export ---


def _export_rope(
    rope: RotaryEmbedding1D, n: int, s: int, max_len: int = 512
) -> torch.export.ExportedProgram:
    rope.eval()
    q = torch.randn(1, H, n, DH)
    k = torch.randn(1, H, s, DH)
    n_dim = torch.export.Dim("n", min=1, max=max_len)
    s_dim = torch.export.Dim("s", min=1, max=max_len)
    return torch.export.export(rope, (q, k), dynamic_shapes=({2: n_dim}, {2: s_dim}))


def test_rope1d_export_succeeds() -> None:
    assert _export_rope(RotaryEmbedding1D(dim_head=DH), n=8, s=8) is not None


def test_rope1d_export_runs_at_new_length() -> None:
    mod = _export_rope(RotaryEmbedding1D(dim_head=DH), n=8, s=8).module()
    q = torch.randn(1, H, 16, DH)
    q_out, k_out = mod(q, q)
    assert q_out.shape == (1, H, 16, DH)
    assert k_out.shape == (1, H, 16, DH)


def test_rope1d_export_cross_attn_different_lengths() -> None:
    # n != s was the removed branch — must export and run correctly
    mod = _export_rope(RotaryEmbedding1D(dim_head=DH), n=6, s=12).module()
    q = torch.randn(1, H, 4, DH)
    k = torch.randn(1, H, 20, DH)
    q_out, k_out = mod(q, k)
    assert q_out.shape == (1, H, 4, DH)
    assert k_out.shape == (1, H, 20, DH)


def test_yarn_rope_export_succeeds() -> None:
    yarn = YaRNConfig(scale=4.0, original_max_seq_len=512)
    assert _export_rope(RotaryEmbedding1D(dim_head=DH, yarn=yarn), n=8, s=8) is not None


def test_yarn_rope_export_runs_at_extended_length() -> None:
    yarn = YaRNConfig(scale=4.0, original_max_seq_len=512)
    mod = _export_rope(RotaryEmbedding1D(dim_head=DH, yarn=yarn), n=8, s=8, max_len=4096).module()
    q = torch.randn(1, H, 2048, DH)
    q_out, k_out = mod(q, q)
    assert q_out.shape == (1, H, 2048, DH)
    assert k_out.shape == (1, H, 2048, DH)
