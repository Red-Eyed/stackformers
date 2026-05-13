from __future__ import annotations

import torch
import torch.export

from stackformers.v1.positional.config import YaRNConfig
from stackformers.v1.positional.rope1d import RotaryEmbedding1D

H, DH = 2, 32


def _export(rope: RotaryEmbedding1D, n: int, s: int) -> torch.export.ExportedProgram:
    rope.eval()
    q = torch.randn(1, H, n, DH)
    k = torch.randn(1, H, s, DH)
    n_dim = torch.export.Dim("n", min=1, max=512)
    s_dim = torch.export.Dim("s", min=1, max=512)
    return torch.export.export(
        rope,
        (q, k),
        dynamic_shapes=({2: n_dim}, {2: s_dim}),
    )


def test_rope1d_export_self_attn_shape() -> None:
    ep = _export(RotaryEmbedding1D(dim_head=DH), n=8, s=8)
    assert ep is not None


def test_rope1d_export_runs_at_new_length() -> None:
    ep = _export(RotaryEmbedding1D(dim_head=DH), n=8, s=8)
    mod = ep.module()
    q = torch.randn(1, H, 16, DH)
    k = torch.randn(1, H, 16, DH)
    q_out, k_out = mod(q, k)
    assert q_out.shape == (1, H, 16, DH)
    assert k_out.shape == (1, H, 16, DH)


def test_rope1d_export_cross_attn_different_lengths() -> None:
    # n != s was the removed branch — must export and run correctly
    ep = _export(RotaryEmbedding1D(dim_head=DH), n=6, s=12)
    mod = ep.module()
    q = torch.randn(1, H, 4, DH)
    k = torch.randn(1, H, 20, DH)
    q_out, k_out = mod(q, k)
    assert q_out.shape == (1, H, 4, DH)
    assert k_out.shape == (1, H, 20, DH)


def test_yarn_rope_export_succeeds() -> None:
    yarn = YaRNConfig(scale=4.0, original_max_seq_len=512)
    ep = _export(RotaryEmbedding1D(dim_head=DH, yarn=yarn), n=8, s=8)
    assert ep is not None


def test_yarn_rope_export_runs_at_extended_length() -> None:
    # Export with a Dim range that covers the extended context (scale=4 → 2048).
    yarn = YaRNConfig(scale=4.0, original_max_seq_len=512)
    rope = RotaryEmbedding1D(dim_head=DH, yarn=yarn)
    rope.eval()
    q0 = torch.randn(1, H, 8, DH)
    k0 = torch.randn(1, H, 8, DH)
    n_dim = torch.export.Dim("n", min=1, max=4096)
    ep = torch.export.export(rope, (q0, k0), dynamic_shapes=({2: n_dim}, {2: n_dim}))

    mod = ep.module()
    q = torch.randn(1, H, 2048, DH)
    k = torch.randn(1, H, 2048, DH)
    q_out, k_out = mod(q, k)
    assert q_out.shape == (1, H, 2048, DH)
    assert k_out.shape == (1, H, 2048, DH)
