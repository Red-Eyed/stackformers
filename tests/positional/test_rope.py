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


# --- RoPE-2D ---
#
# A row-major GH x GW grid, never row == col. On the diagonal (row == col) the row and column
# angles coincide, which makes a broken frequency layout indistinguishable from a correct one.

GH, GW = 3, 5
GN = GH * GW


def _grid_positions(b: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Row-major (row, col) positions over a GH x GW grid, as (b, GN, 2)."""
    rows = torch.arange(GH, device=device, dtype=dtype).repeat_interleave(GW)
    cols = torch.arange(GW, device=device, dtype=dtype).repeat(GH)
    return torch.stack([rows, cols], dim=-1).unsqueeze(0).expand(b, -1, -1).clone()


def _reference_rope2d(t: torch.Tensor, positions: torch.Tensor, base: int = 10_000) -> torch.Tensor:
    """Axial 2-D RoPE written directly from the definition, independent of the implementation.

    Rotates channel pair (i, i + dh/2) by row * w_i for the first dh/4 pairs and by col * w_i
    for the rest. t: (b, h, n, dh), positions: (b, n, 2).
    """
    dh = t.shape[-1]
    half, quarter = dh // 2, dh // 4
    inv = 1.0 / (base ** (torch.arange(0, half, 2, dtype=torch.float32) / half))
    inv = inv.to(device=t.device)

    out = torch.zeros_like(t, dtype=torch.float32)
    src = t.float()
    for i in range(half):
        axis = 0 if i < quarter else 1
        freq = inv[i if i < quarter else i - quarter]
        angle = positions[..., axis].float() * freq  # (b, n)
        cos, sin = angle.cos().unsqueeze(1), angle.sin().unsqueeze(1)  # (b, 1, n)
        x, y = src[..., i], src[..., i + half]
        out[..., i] = x * cos - y * sin
        out[..., i + half] = x * sin + y * cos
    return out.to(t.dtype)


@pytest.fixture
def grid_qk(device_dtype: tuple[torch.device, torch.dtype]) -> tuple[torch.Tensor, torch.Tensor]:
    device, dtype = device_dtype
    return (
        torch.randn(B, H, GN, DH, device=device, dtype=dtype),
        torch.randn(B, H, GN, DH, device=device, dtype=dtype),
    )


@pytest.fixture
def grid_pos(device_dtype: tuple[torch.device, torch.dtype]) -> torch.Tensor:
    device, dtype = device_dtype
    return _grid_positions(B, device, dtype)


def test_rope2d_output_shape(
    rope2d: RotaryEmbedding2D,
    grid_qk: tuple[torch.Tensor, torch.Tensor],
    grid_pos: torch.Tensor,
) -> None:
    q, k = grid_qk
    q_out, k_out = rope2d.forward_padded(q, k, grid_pos, grid_pos)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_rope2d_matches_reference_rotation(
    rope2d: RotaryEmbedding2D,
    grid_qk: tuple[torch.Tensor, torch.Tensor],
    grid_pos: torch.Tensor,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """Pins the exact rotation against a definition-first oracle, not just its properties."""
    _, dtype = device_dtype
    q, k = grid_qk
    q_out, k_out = rope2d.forward_padded(q, k, grid_pos, grid_pos)
    assert torch.allclose(q_out, _reference_rope2d(q, grid_pos), atol=atol(dtype))
    assert torch.allclose(k_out, _reference_rope2d(k, grid_pos), atol=atol(dtype))


def test_rope2d_is_orthogonal(
    rope2d: RotaryEmbedding2D,
    grid_qk: tuple[torch.Tensor, torch.Tensor],
    grid_pos: torch.Tensor,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """A rotation preserves norms and inner products; a squeeze or reflection does not."""
    _, dtype = device_dtype
    q, k = grid_qk
    q_out, k_out = rope2d.forward_padded(q, k, grid_pos, grid_pos)
    tol = atol(dtype)
    assert torch.allclose(q.norm(dim=-1), q_out.norm(dim=-1), atol=tol)
    assert torch.allclose(k.norm(dim=-1), k_out.norm(dim=-1), atol=tol)
    # Same position for q and k, so the shared rotation must cancel in the inner product.
    before = (q.float() * k.float()).sum(dim=-1)
    after = (q_out.float() * k_out.float()).sum(dim=-1)
    assert torch.allclose(before, after, atol=tol * DH)


@pytest.mark.parametrize("offset", [(0, 1), (1, 0), (2, 3), (-1, 2)], ids=str)
def test_rope2d_score_depends_only_on_relative_offset(
    device: torch.device,
    offset: tuple[int, int],
) -> None:
    """The defining property: sliding a query/key pair across the grid must not move the score."""
    rope = RotaryEmbedding2D(RoPE2DConfig(dim_head=DH)).to(device=device)
    q = torch.randn(1, H, 1, DH, device=device)
    k = torch.randn(1, H, 1, DH, device=device)
    d_row, d_col = offset

    def score(row: int, col: int) -> torch.Tensor:
        q_pos = torch.tensor([[[float(row), float(col)]]], device=device)
        k_pos = torch.tensor([[[float(row + d_row), float(col + d_col)]]], device=device)
        q_out, k_out = rope.forward_padded(q, k, q_pos, k_pos)
        return (q_out @ k_out.transpose(-1, -2))[0, :, 0, 0]

    anchor = score(0, 0)
    for row, col in [(2, 3), (5, 9), (11, 4), (7, 7)]:
        assert torch.allclose(anchor, score(row, col), atol=atol(torch.float32) * 10)


def test_rope2d_distinct_offsets_give_distinct_scores(device: torch.device) -> None:
    """Guards the degenerate fix where every offset collapses to the same score."""
    rope = RotaryEmbedding2D(RoPE2DConfig(dim_head=DH)).to(device=device)
    q = torch.randn(1, H, 1, DH, device=device)
    k = torch.randn(1, H, 1, DH, device=device)

    def score(k_row: float, k_col: float) -> torch.Tensor:
        q_pos = torch.zeros(1, 1, 2, device=device)
        k_pos = torch.tensor([[[k_row, k_col]]], device=device)
        q_out, k_out = rope.forward_padded(q, k, q_pos, k_pos)
        return (q_out @ k_out.transpose(-1, -2))[0, :, 0, 0]

    # A row-offset and the matching col-offset must not be conflated: the axes are separate.
    assert not torch.allclose(score(2.0, 0.0), score(0.0, 2.0))
    assert not torch.allclose(score(0.0, 0.0), score(1.0, 1.0))


@pytest.mark.parametrize("axis", [0, 1], ids=["row", "col"])
def test_rope2d_each_axis_changes_the_encoding(
    rope2d: RotaryEmbedding2D,
    device_dtype: tuple[torch.device, torch.dtype],
    axis: int,
) -> None:
    """Moving along either axis alone must change the output — neither axis may be ignored."""
    device, dtype = device_dtype
    q = torch.randn(1, H, 1, DH, device=device, dtype=dtype)
    base_pos = torch.full((1, 1, 2), 2.0, device=device, dtype=dtype)
    moved_pos = base_pos.clone()
    moved_pos[..., axis] += 3.0
    base_out, _ = rope2d.forward_padded(q, q, base_pos, base_pos)
    moved_out, _ = rope2d.forward_padded(q, q, moved_pos, moved_pos)
    assert not torch.allclose(base_out, moved_out, atol=atol(dtype))


def test_rope2d_packed_matches_padded(
    rope2d: RotaryEmbedding2D,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """Both layouts are the same per-token operation and must agree token for token."""
    device, dtype = device_dtype
    q = torch.randn(1, H, GN, DH, device=device, dtype=dtype)
    pos = _grid_positions(1, device, dtype)
    padded, _ = rope2d.forward_padded(q, q, pos, pos)

    packed_q = q[0].transpose(0, 1).contiguous()  # (nt, h, dh)
    packed_out, _ = rope2d.forward_packed(packed_q, packed_q, pos[0], pos[0])
    assert torch.allclose(packed_out, padded[0].transpose(0, 1), atol=atol(dtype))


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
