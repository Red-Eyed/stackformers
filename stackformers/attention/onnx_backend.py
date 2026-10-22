"""ONNX cache-update and valid-length attention lowerings with eager PyTorch equivalents."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

if TYPE_CHECKING:
    from jaxtyping import Float, Int


def update_buffer(
    cache: Float[Tensor, "b h capacity dh"],
    update: Float[Tensor, "b h n dh"],
    index: Int[Tensor, "b"],
) -> Tensor:
    """Expose TensorScatter to ONNX; eager execution writes into the owned buffer."""
    if torch.onnx.is_in_onnx_export():
        return torch.onnx.ops.symbolic(
            "::TensorScatter",
            (cache, update, index),
            {"axis": 2},
            dtype=cache.dtype,
            shape=cache.shape,
            version=24,
        )
    return cache.index_copy_(2, index, update)


def valid_length_attention(
    q: Float[Tensor, "b h n dh"],
    k: Float[Tensor, "b kv_h capacity dh"],
    v: Float[Tensor, "b kv_h capacity dh"],
    valid_length: Int[Tensor, "b"],
) -> Tensor:
    """Exclude unused capacity using Attention-24's valid-length input during export."""
    if torch.onnx.is_in_onnx_export():
        return torch.onnx.ops.symbolic(
            "::Attention",
            (q, k, v, None, None, None, valid_length),
            {"is_causal": 0},
            dtype=q.dtype,
            shape=q.shape,
            version=24,
        )
    positions = torch.arange(k.shape[2], device=k.device)
    valid = rearrange(positions < valid_length, "s -> 1 s")
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=valid, is_causal=False, enable_gqa=q.shape[1] != k.shape[1]
    )
