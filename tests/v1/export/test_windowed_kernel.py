from __future__ import annotations

import torch
import torch.export

from stackformers.v1.attention.kernels.windowed import WindowedSDPAKernel

H, DH = 2, 16


def _export(kernel: WindowedSDPAKernel, n: int) -> torch.export.ExportedProgram:
    kernel.eval()
    q = torch.randn(1, H, n, DH)
    k = torch.randn(1, H, n, DH)
    v = torch.randn(1, H, n, DH)
    n_dim = torch.export.Dim("n", min=1, max=512)
    seq = {2: n_dim}
    return torch.export.export(
        kernel,
        (q, k, v, None, None, False),
        dynamic_shapes=(seq, seq, seq, None, None, None),
    )


def test_windowed_kernel_export_succeeds() -> None:
    ep = _export(WindowedSDPAKernel(window_size=4), n=8)
    assert ep is not None


def test_windowed_kernel_export_runs_at_new_length() -> None:
    ep = _export(WindowedSDPAKernel(window_size=4), n=8)
    mod = ep.module()
    # window_size=4, n=3: sequence shorter than window — previously hit the removed branch
    q = torch.randn(1, H, 3, DH)
    k = torch.randn(1, H, 3, DH)
    v = torch.randn(1, H, 3, DH)
    out = mod(q, k, v, None, None, False)
    assert out.shape == (1, H, 3, DH)


def test_windowed_kernel_export_runs_at_longer_length() -> None:
    ep = _export(WindowedSDPAKernel(window_size=4), n=8)
    mod = ep.module()
    q = torch.randn(1, H, 32, DH)
    k = torch.randn(1, H, 32, DH)
    v = torch.randn(1, H, 32, DH)
    out = mod(q, k, v, None, None, False)
    assert out.shape == (1, H, 32, DH)


def test_windowed_causal_kernel_export_succeeds() -> None:
    ep = _export(WindowedSDPAKernel(window_size=4, causal=True), n=8)
    assert ep is not None
