from __future__ import annotations

import pytest
import torch
import torch.export

from stackformers.positional.config import RoPE1DConfig, YaRNConfig
from stackformers.positional.none import NoPosEncoding
from stackformers.positional.rope1d import RotaryEmbedding1D
from stackformers.positional.rope2d import RotaryEmbedding2D
from stackformers.sequence import make_padded_input
from tests.conftest import atol

B, H, N, DH = 2, 4, 8, 32


def _make_inputs(q: torch.Tensor, k: torch.Tensor) -> tuple[object, object]:
    """Build PaddedInput wrappers for q and k tensors (b h n dh → b n d projection stripped)."""
    b, _, n, _ = q.shape
    _, _, s, _ = k.shape
    device = q.device
    dummy_x_q = torch.zeros(b, n, 1, device=device, dtype=q.dtype)
    dummy_x_k = torch.zeros(b, s, 1, device=device, dtype=k.dtype)
    mask_q = torch.ones(b, n, dtype=torch.bool, device=device)
    mask_k = torch.ones(b, s, dtype=torch.bool, device=device)
    return make_padded_input(dummy_x_q, mask_q), make_padded_input(dummy_x_k, mask_k)


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
    return RotaryEmbedding2D(dim_head=DH).to(device=device, dtype=dtype)


def test_no_pos_encoding_passthrough(
    qk: tuple[torch.Tensor, torch.Tensor],
) -> None:
    q, k = qk
    q_inp, k_inp = _make_inputs(q, k)
    q_out, k_out = NoPosEncoding()(q, k, q_inp, k_inp)  # type: ignore[arg-type]
    assert torch.equal(q_out, q)
    assert torch.equal(k_out, k)


def test_rope1d_output_shape(
    rope1d: RotaryEmbedding1D,
    qk: tuple[torch.Tensor, torch.Tensor],
) -> None:
    q, k = qk
    q_inp, k_inp = _make_inputs(q, k)
    q_out, k_out = rope1d(q, k, q_inp, k_inp)  # type: ignore[arg-type]
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_rope1d_cross_attn_different_lengths(
    rope1d: RotaryEmbedding1D,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q = torch.randn(B, H, 6, DH, device=device, dtype=dtype)
    k = torch.randn(B, H, 12, DH, device=device, dtype=dtype)
    q_inp, k_inp = _make_inputs(q, k)
    q_out, k_out = rope1d(q, k, q_inp, k_inp)  # type: ignore[arg-type]
    assert q_out.shape == (B, H, 6, DH)
    assert k_out.shape == (B, H, 12, DH)


def test_rope1d_different_positions_produce_different_output(
    rope1d: RotaryEmbedding1D,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q = torch.ones(1, 1, 4, DH, device=device, dtype=dtype)
    k = torch.ones(1, 1, 4, DH, device=device, dtype=dtype)
    q_inp, k_inp = _make_inputs(q, k)
    q_out, _ = rope1d(q, k, q_inp, k_inp)  # type: ignore[arg-type]
    assert not torch.allclose(q_out[:, :, 0], q_out[:, :, 1])


def test_rope1d_preserves_norms(
    rope1d: RotaryEmbedding1D,
    qk: tuple[torch.Tensor, torch.Tensor],
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    _, dtype = device_dtype
    q, k = qk
    q_inp, k_inp = _make_inputs(q, k)
    q_out, k_out = rope1d(q, k, q_inp, k_inp)  # type: ignore[arg-type]
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
    return RotaryEmbedding1D(RoPE1DConfig(dim_head=DH, yarn=_YARN)).to(device=device, dtype=dtype)


def test_yarn_output_shape(
    yarn_rope: RotaryEmbedding1D,
    qk: tuple[torch.Tensor, torch.Tensor],
) -> None:
    q, k = qk
    q_inp, k_inp = _make_inputs(q, k)
    q_out, k_out = yarn_rope(q, k, q_inp, k_inp)  # type: ignore[arg-type]
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_yarn_preserves_norms(
    yarn_rope: RotaryEmbedding1D,
    qk: tuple[torch.Tensor, torch.Tensor],
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    _, dtype = device_dtype
    q, k = qk
    q_inp, k_inp = _make_inputs(q, k)
    q_out, k_out = yarn_rope(q, k, q_inp, k_inp)  # type: ignore[arg-type]
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
    q_inp, k_inp = _make_inputs(q, k)
    q_base, _ = base(q, k, q_inp, k_inp)  # type: ignore[arg-type]
    q_yarn, _ = yarn(q, k, q_inp, k_inp)  # type: ignore[arg-type]
    assert not torch.allclose(q_base, q_yarn)


# --- torch.export ---


def _export_rope(
    rope: RotaryEmbedding1D, n: int, s: int, max_len: int = 512
) -> torch.export.ExportedProgram:
    rope.eval()
    b = 1
    q = torch.randn(b, H, n, DH)
    k = torch.randn(b, H, s, DH)
    q_inp = make_padded_input(torch.zeros(b, n, 1), torch.ones(b, n, dtype=torch.bool))
    k_inp = make_padded_input(torch.zeros(b, s, 1), torch.ones(b, s, dtype=torch.bool))
    n_dim = torch.export.Dim("n", min=1, max=max_len)
    s_dim = torch.export.Dim("s", min=1, max=max_len)
    from stackformers.sequence import PaddedInput as PI

    # PaddedInput fields in order: x (b n d), mask (b n), abs_positions (b n)
    return torch.export.export(
        rope,
        (q, k, q_inp, k_inp),
        dynamic_shapes=(
            {2: n_dim},
            {2: s_dim},
            PI(x={1: n_dim}, mask={1: n_dim}, abs_positions={1: n_dim}),  # type: ignore[arg-type]
            PI(x={1: s_dim}, mask={1: s_dim}, abs_positions={1: s_dim}),  # type: ignore[arg-type]
        ),
    )


def test_rope1d_export_succeeds() -> None:
    assert _export_rope(RotaryEmbedding1D(RoPE1DConfig(dim_head=DH)), n=8, s=8) is not None


def test_rope1d_export_runs_at_new_length() -> None:
    mod = _export_rope(RotaryEmbedding1D(RoPE1DConfig(dim_head=DH)), n=8, s=8).module()
    b = 1
    q = torch.randn(b, H, 16, DH)
    q_inp = make_padded_input(torch.zeros(b, 16, 1), torch.ones(b, 16, dtype=torch.bool))
    q_out, k_out = mod(q, q, q_inp, q_inp)
    assert q_out.shape == (b, H, 16, DH)
    assert k_out.shape == (b, H, 16, DH)


def test_rope1d_export_cross_attn_different_lengths() -> None:
    mod = _export_rope(RotaryEmbedding1D(RoPE1DConfig(dim_head=DH)), n=6, s=12).module()
    b = 1
    q = torch.randn(b, H, 4, DH)
    k = torch.randn(b, H, 20, DH)
    q_inp = make_padded_input(torch.zeros(b, 4, 1), torch.ones(b, 4, dtype=torch.bool))
    k_inp = make_padded_input(torch.zeros(b, 20, 1), torch.ones(b, 20, dtype=torch.bool))
    q_out, k_out = mod(q, k, q_inp, k_inp)
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
    q_inp = make_padded_input(torch.zeros(b, 2048, 1), torch.ones(b, 2048, dtype=torch.bool))
    q_out, k_out = mod(q, q, q_inp, q_inp)
    assert q_out.shape == (b, H, 2048, DH)
    assert k_out.shape == (b, H, 2048, DH)
