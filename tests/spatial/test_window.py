from __future__ import annotations

import pytest
import torch

from stackformers.positional.config import RoPE2DConfig
from stackformers.positional.none import NoPosEncoding
from stackformers.positional.rope2d import RotaryEmbedding2D
from stackformers.spatial.config import WindowAttention2DConfig
from stackformers.spatial.factory import build_spatial_attn
from stackformers.spatial.input import make_spatial_input
from stackformers.spatial.protocols import SpatialAttn
from stackformers.spatial.window import WindowAttention2D, partition_windows

B, H, W, D, HEADS, DH, WIN = 2, 8, 8, 64, 4, 16, 4
N = H * W


def _attn(window: int = WIN, **kw: object) -> WindowAttention2D:
    config = WindowAttention2DConfig(dim=D, heads=HEADS, dim_head=DH, window=window, **kw)  # type: ignore[arg-type]
    return WindowAttention2D(config, NoPosEncoding())


def test_partition_merge_roundtrip(device: torch.device) -> None:
    x = torch.randn(B, N, D, device=device)
    windows, merge = partition_windows(x, H, W, WIN)
    assert windows.shape == (B * (H // WIN) * (W // WIN), WIN * WIN, D)
    assert torch.equal(merge(windows), x)


def test_window_attn_output_shape(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    attn = _attn().to(device=device, dtype=dtype)
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    assert attn(make_spatial_input(x, (H, W))).shape == (B, N, D)


def test_window_attn_satisfies_protocol() -> None:
    assert isinstance(_attn(), SpatialAttn)


def test_window_attn_with_rope2d(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    config = WindowAttention2DConfig(dim=D, heads=HEADS, dim_head=DH, window=WIN)
    attn = WindowAttention2D(config, RotaryEmbedding2D(RoPE2DConfig(dim_head=DH))).to(
        device=device, dtype=dtype
    )
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    assert attn(make_spatial_input(x, (H, W))).shape == (B, N, D)


def test_window_attn_gqa_shape(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    attn = _attn(kv_heads=2).to(device=device, dtype=dtype)
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    assert attn(make_spatial_input(x, (H, W))).shape == (B, N, D)


def test_window_attn_rejects_indivisible_grid(device: torch.device) -> None:
    attn = _attn(window=3).to(device=device)
    x = torch.randn(B, N, D, device=device)
    with pytest.raises(ValueError, match="not divisible"):
        attn(make_spatial_input(x, (H, W)))


def test_window_attn_gradients(device: torch.device) -> None:
    attn = _attn().to(device=device)
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    attn(make_spatial_input(x, (H, W))).sum().backward()
    assert x.grad is not None


def test_build_spatial_attn_dispatch_window() -> None:
    config = WindowAttention2DConfig(dim=D, heads=HEADS, dim_head=DH, window=WIN)
    assert isinstance(build_spatial_attn(config, NoPosEncoding()), WindowAttention2D)
