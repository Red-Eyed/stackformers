"""Eager equivalents of the batch-one ONNX cache kernels preserve prefix semantics."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from stackformers.attention.onnx_backend import update_buffer, valid_length_attention
from tests.conftest import atol


@pytest.fixture
def qkv(device_dtype: tuple[torch.device, torch.dtype]) -> tuple[torch.Tensor, ...]:
    """Use GQA queries and finite nonzero unused cache slots to expose length mistakes."""
    device, dtype = device_dtype
    q = torch.linspace(-1, 1, 8, device=device, dtype=dtype).reshape(1, 2, 1, 4)
    k = torch.linspace(-0.5, 0.5, 20, device=device, dtype=dtype).reshape(1, 1, 5, 4)
    v = torch.linspace(1, 2, 20, device=device, dtype=dtype).reshape(1, 1, 5, 4)
    return q, k, v


def test_valid_length_ignores_unused_capacity(qkv: tuple[torch.Tensor, ...]) -> None:
    """Finite unused slots contribute neither attention probabilities nor output values."""
    q, k, v = qkv
    length = torch.tensor([3], device=q.device, dtype=torch.int64)
    expected = F.scaled_dot_product_attention(q, k[:, :, :3], v[:, :, :3], enable_gqa=True)
    actual = valid_length_attention(q, k, v, length)
    torch.testing.assert_close(actual, expected, atol=atol(q.dtype), rtol=atol(q.dtype))


def test_update_retains_owned_storage(qkv: tuple[torch.Tensor, ...]) -> None:
    """Only the new token slot changes while the caller's allocated buffer remains owned."""
    _, k, _ = qkv
    before = k.clone()
    index = torch.tensor([2], device=k.device, dtype=torch.int64)
    update = torch.full_like(k[:, :, :1], 7)
    result = update_buffer(k, update, index)
    assert result.data_ptr() == k.data_ptr()
    torch.testing.assert_close(k[:, :, :2], before[:, :, :2], rtol=0, atol=0)
    torch.testing.assert_close(k[:, :, 2:3], update, rtol=0, atol=0)
    torch.testing.assert_close(k[:, :, 3:], before[:, :, 3:], rtol=0, atol=0)
