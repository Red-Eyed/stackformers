from __future__ import annotations

import torch

from stackformers.positional.config import RoPE2DConfig
from stackformers.positional.none import NoPosEncoding
from stackformers.positional.protocols import PosEncoding
from stackformers.positional.rope2d import RotaryEmbedding2D
from stackformers.spatial.config import SpatialReductionAttentionConfig
from stackformers.spatial.factory import build_spatial_attn
from stackformers.spatial.input import make_spatial_input
from stackformers.spatial.kv_reduction import ConvKVReduction, NoKVReduction
from stackformers.spatial.protocols import SpatialAttn
from stackformers.spatial.spatial_reduction import SpatialReductionAttention

B, H, W, D, HEADS, DH = 2, 8, 8, 64, 4, 16
N = H * W


def _attn(
    reduction: int, pos: PosEncoding | None = None, **kw: object
) -> SpatialReductionAttention:
    config = SpatialReductionAttentionConfig(
        dim=D,
        heads=HEADS,
        dim_head=DH,
        reduction=reduction,
        **kw,  # type: ignore[arg-type]
    )
    built = build_spatial_attn(config, pos or NoPosEncoding())
    assert isinstance(built, SpatialReductionAttention)
    return built


def test_sra_output_shape(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    attn = _attn(reduction=2).to(device=device, dtype=dtype)
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    assert attn(make_spatial_input(x, (H, W))).shape == (B, N, D)


def test_sra_satisfies_protocol() -> None:
    assert isinstance(_attn(reduction=2), SpatialAttn)


def test_sra_reduction_one_uses_null_reduction() -> None:
    assert isinstance(_attn(reduction=1).kv_reduction, NoKVReduction)


def test_sra_reduction_gt_one_uses_conv() -> None:
    assert isinstance(_attn(reduction=2).kv_reduction, ConvKVReduction)


def test_sra_full_attention_shape(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    attn = _attn(reduction=1).to(device=device, dtype=dtype)  # r=1 → global full attention
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    assert attn(make_spatial_input(x, (H, W))).shape == (B, N, D)


def test_sra_with_rope2d(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    attn = _attn(reduction=2, pos=RotaryEmbedding2D(RoPE2DConfig(dim_head=DH))).to(
        device=device, dtype=dtype
    )
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    assert attn(make_spatial_input(x, (H, W))).shape == (B, N, D)


def test_sra_gqa_shape(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    attn = _attn(reduction=2, kv_heads=2).to(device=device, dtype=dtype)
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    assert attn(make_spatial_input(x, (H, W))).shape == (B, N, D)


def test_sra_gradients(device: torch.device) -> None:
    attn = _attn(reduction=2).to(device=device)
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    attn(make_spatial_input(x, (H, W))).sum().backward()
    assert x.grad is not None
