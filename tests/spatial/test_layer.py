from __future__ import annotations

import torch

from stackformers.feedforward.config import SwiGLUConfig
from stackformers.feedforward.factory import build_ff
from stackformers.norm.config import RMSNormConfig
from stackformers.norm.factory import build_norm
from stackformers.positional.none import NoPosEncoding
from stackformers.spatial.config import WindowAttention2DConfig
from stackformers.spatial.factory import build_spatial_attn
from stackformers.spatial.input import SpatialInput, make_spatial_input
from stackformers.spatial.layer import SpatialTransformerLayer

B, H, W, D, HEADS, DH, WIN = 2, 8, 8, 64, 4, 16, 4
N = H * W


def _layer() -> SpatialTransformerLayer:
    attn = build_spatial_attn(
        WindowAttention2DConfig(dim=D, heads=HEADS, dim_head=DH, window=WIN), NoPosEncoding()
    )
    return SpatialTransformerLayer(
        attn=attn,
        ff=build_ff(SwiGLUConfig(dim=D)),
        norm_attn=build_norm(RMSNormConfig(dim=D)),
        norm_ff=build_norm(RMSNormConfig(dim=D)),
    )


def test_layer_preserves_grid_and_shape(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    layer = _layer().to(device=device, dtype=dtype)
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    out = layer(make_spatial_input(x, (H, W)))
    assert isinstance(out, SpatialInput)
    assert out.x.shape == (B, N, D)
    assert out.grid == (H, W)


def test_layer_gradients(device: torch.device) -> None:
    layer = _layer().to(device=device)
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    layer(make_spatial_input(x, (H, W))).x.sum().backward()
    assert x.grad is not None
