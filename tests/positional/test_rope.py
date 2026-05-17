from __future__ import annotations

import pytest
import torch
import torch.export
import torch.nn as nn

from stackformers.positional.config import RoPE1DConfig, RoPE2DConfig, YaRNConfig
from stackformers.positional.none import NoPosEncoding
from stackformers.positional.rope1d import RotaryEmbedding1D
from stackformers.positional.rope2d import RotaryEmbedding2D
from tests.conftest import atol

B, H, N, DH = 2, 4, 8, 32


def _positions(b: int, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Sequential 1-D positions as (b, n, 1)."""
    pos = torch.arange(n, device=device, dtype=dtype).view(1, n, 1).expand(b, -1, -1)
    return pos.clone()


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
    return RotaryEmbedding1D(RoPE1DConfig(dim_head=DH)).to(device=device, dtype=dtype)


@pytest.fixture
def rope2d(device_dtype: tuple[torch.device, torch.dtype]) -> RotaryEmbedding2D:
    device, dtype = device_dtype
    return RotaryEmbedding2D(RoPE2DConfig(dim_head=DH)).to(device=device, dtype=dtype)


def test_no_pos_encoding_passthrough(
    qk: tuple[torch.Tensor, torch.Tensor],
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q, k = qk
    pos = _positions(B, N, device, dtype)
    q_out, k_out = NoPosEncoding().forward_padded(q, k, pos, pos)
    assert torch.equal(q_out, q)
    assert torch.equal(k_out, k)


def test_rope1d_output_shape(
    rope1d: RotaryEmbedding1D,
    qk: tuple[torch.Tensor, torch.Tensor],
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q, k = qk
    pos = _positions(B, N, device, dtype)
    q_out, k_out = rope1d.forward_padded(q, k, pos, pos)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_rope1d_cross_attn_different_lengths(
    rope1d: RotaryEmbedding1D,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q = torch.randn(B, H, 6, DH, device=device, dtype=dtype)
    k = torch.randn(B, H, 12, DH, device=device, dtype=dtype)
    q_pos = _positions(B, 6, device, dtype)
    k_pos = _positions(B, 12, device, dtype)
    q_out, k_out = rope1d.forward_padded(q, k, q_pos, k_pos)
    assert q_out.shape == (B, H, 6, DH)
    assert k_out.shape == (B, H, 12, DH)


def test_rope1d_different_positions_produce_different_output(
    rope1d: RotaryEmbedding1D,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q = torch.ones(1, 1, 4, DH, device=device, dtype=dtype)
    k = torch.ones(1, 1, 4, DH, device=device, dtype=dtype)
    pos = _positions(1, 4, device, dtype)
    q_out, _ = rope1d.forward_padded(q, k, pos, pos)
    assert not torch.allclose(q_out[:, :, 0], q_out[:, :, 1])


def test_rope1d_relative_distance_invariant(
    rope1d: RotaryEmbedding1D,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q = torch.randn(1, H, 1, DH, device=device, dtype=dtype)
    k = torch.randn(1, H, 1, DH, device=device, dtype=dtype)

    def _pos(p: int) -> torch.Tensor:
        return torch.full((1, 1, 1), float(p), device=device, dtype=dtype)

    # Same relative distance (delta=3) at two different absolute offsets.
    q1, k1 = rope1d.forward_padded(q, k, _pos(1), _pos(4))
    q2, k2 = rope1d.forward_padded(q, k, _pos(8), _pos(11))
    score1 = (q1 @ k1.transpose(-1, -2))[0, :, 0, 0]
    score2 = (q2 @ k2.transpose(-1, -2))[0, :, 0, 0]
    assert torch.allclose(score1, score2, atol=atol(dtype))


def test_rope1d_preserves_norms(
    rope1d: RotaryEmbedding1D,
    qk: tuple[torch.Tensor, torch.Tensor],
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q, k = qk
    pos = _positions(B, N, device, dtype)
    q_out, k_out = rope1d.forward_padded(q, k, pos, pos)
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
    pos = torch.stack([row_ids, col_ids], dim=-1).unsqueeze(0).expand(B, -1, -1)  # b n 2
    q_out, k_out = rope2d.forward_padded(q, k, pos, pos)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


# --- YaRN scaling ---

_YARN = YaRNConfig(scale=4.0, original_max_seq_len=512)


@pytest.fixture
def yarn_rope(device_dtype: tuple[torch.device, torch.dtype]) -> RotaryEmbedding1D:
    device, dtype = device_dtype
    return RotaryEmbedding1D(RoPE1DConfig(dim_head=DH, yarn=_YARN)).to(device=device, dtype=dtype)


def test_yarn_output_shape(
    yarn_rope: RotaryEmbedding1D,
    qk: tuple[torch.Tensor, torch.Tensor],
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q, k = qk
    pos = _positions(B, N, device, dtype)
    q_out, k_out = yarn_rope.forward_padded(q, k, pos, pos)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_yarn_preserves_norms(
    yarn_rope: RotaryEmbedding1D,
    qk: tuple[torch.Tensor, torch.Tensor],
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q, k = qk
    pos = _positions(B, N, device, dtype)
    q_out, k_out = yarn_rope.forward_padded(q, k, pos, pos)
    tol = atol(dtype)
    assert torch.allclose(q.norm(dim=-1), q_out.norm(dim=-1), atol=tol)
    assert torch.allclose(k.norm(dim=-1), k_out.norm(dim=-1), atol=tol)


def test_yarn_differs_from_base_rope(
    qk: tuple[torch.Tensor, torch.Tensor],
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    base = RotaryEmbedding1D(RoPE1DConfig(dim_head=DH)).to(device=device, dtype=dtype)
    yarn = RotaryEmbedding1D(RoPE1DConfig(dim_head=DH, yarn=_YARN)).to(device=device, dtype=dtype)
    q, k = qk
    pos = _positions(B, N, device, dtype)
    q_base, _ = base.forward_padded(q, k, pos, pos)
    q_yarn, _ = yarn.forward_padded(q, k, pos, pos)
    assert not torch.allclose(q_base, q_yarn)


# --- torch.export ---


class _PaddedRopeWrapper(nn.Module):
    """Wraps forward_padded into a plain forward for torch.export."""

    def __init__(self, enc: RotaryEmbedding1D) -> None:
        super().__init__()
        self.enc = enc

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_pos: torch.Tensor,
        k_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.enc.forward_padded(q, k, q_pos, k_pos)


def _export_rope(
    rope: RotaryEmbedding1D, n: int, s: int, max_len: int = 512
) -> torch.export.ExportedProgram:
    wrapper = _PaddedRopeWrapper(rope).eval()
    b = 1
    q = torch.randn(b, H, n, DH)
    k = torch.randn(b, H, s, DH)
    q_pos = torch.arange(n, dtype=torch.float32).view(1, n, 1).expand(b, -1, -1).clone()
    k_pos = torch.arange(s, dtype=torch.float32).view(1, s, 1).expand(b, -1, -1).clone()
    n_dim = torch.export.Dim("n", min=1, max=max_len)
    s_dim = torch.export.Dim("s", min=1, max=max_len)
    return torch.export.export(
        wrapper,
        (q, k, q_pos, k_pos),
        dynamic_shapes=({2: n_dim}, {2: s_dim}, {1: n_dim}, {1: s_dim}),
    )


def test_rope1d_export_succeeds() -> None:
    assert _export_rope(RotaryEmbedding1D(RoPE1DConfig(dim_head=DH)), n=8, s=8) is not None


def test_rope1d_export_runs_at_new_length() -> None:
    mod = _export_rope(RotaryEmbedding1D(RoPE1DConfig(dim_head=DH)), n=8, s=8).module()
    b = 1
    q = torch.randn(b, H, 16, DH)
    q_pos = torch.arange(16, dtype=torch.float32).view(1, 16, 1).expand(b, -1, -1).clone()
    q_out, k_out = mod(q, q, q_pos, q_pos)
    assert q_out.shape == (b, H, 16, DH)
    assert k_out.shape == (b, H, 16, DH)


def test_rope1d_export_cross_attn_different_lengths() -> None:
    mod = _export_rope(RotaryEmbedding1D(RoPE1DConfig(dim_head=DH)), n=6, s=12).module()
    b = 1
    q = torch.randn(b, H, 4, DH)
    k = torch.randn(b, H, 20, DH)
    q_pos = torch.arange(4, dtype=torch.float32).view(1, 4, 1).expand(b, -1, -1).clone()
    k_pos = torch.arange(20, dtype=torch.float32).view(1, 20, 1).expand(b, -1, -1).clone()
    q_out, k_out = mod(q, k, q_pos, k_pos)
    assert q_out.shape == (b, H, 4, DH)
    assert k_out.shape == (b, H, 20, DH)


def test_yarn_rope_export_succeeds() -> None:
    yarn = YaRNConfig(scale=4.0, original_max_seq_len=512)
    assert (
        _export_rope(RotaryEmbedding1D(RoPE1DConfig(dim_head=DH, yarn=yarn)), n=8, s=8) is not None
    )


def test_yarn_rope_export_runs_at_extended_length() -> None:
    yarn = YaRNConfig(scale=4.0, original_max_seq_len=512)
    mod = _export_rope(
        RotaryEmbedding1D(RoPE1DConfig(dim_head=DH, yarn=yarn)), n=8, s=8, max_len=4096
    ).module()
    b = 1
    q = torch.randn(b, H, 2048, DH)
    q_pos = torch.arange(2048, dtype=torch.float32).view(1, 2048, 1).expand(b, -1, -1).clone()
    q_out, k_out = mod(q, q, q_pos, q_pos)
    assert q_out.shape == (b, H, 2048, DH)
    assert k_out.shape == (b, H, 2048, DH)
