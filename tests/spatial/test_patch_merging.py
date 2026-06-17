from __future__ import annotations

import torch

from stackformers.spatial.config import PatchMergingConfig
from stackformers.spatial.factory import build_patch_merging
from stackformers.spatial.input import SpatialInput, make_spatial_input

B, H, W, IN, OUT = 2, 8, 8, 32, 64
N = H * W


def test_patch_merging_halves_grid_widens_dim(
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    merge = build_patch_merging(PatchMergingConfig(in_dim=IN, out_dim=OUT)).to(
        device=device, dtype=dtype
    )
    x = torch.randn(B, N, IN, device=device, dtype=dtype)
    out = merge(make_spatial_input(x, (H, W)))
    assert isinstance(out, SpatialInput)
    assert out.grid == (H // 2, W // 2)
    assert out.x.shape == (B, (H // 2) * (W // 2), OUT)
    assert out.abs_positions.shape == (B, (H // 2) * (W // 2), 2)


def test_patch_merging_gradients(device: torch.device) -> None:
    merge = build_patch_merging(PatchMergingConfig(in_dim=IN, out_dim=OUT)).to(device=device)
    x = torch.randn(B, N, IN, device=device, requires_grad=True)
    merge(make_spatial_input(x, (H, W))).x.sum().backward()
    assert x.grad is not None


def test_patch_merging_norm_dim_synced() -> None:
    config = PatchMergingConfig(in_dim=IN, out_dim=OUT)
    assert config.norm.dim == OUT
