"""Projected attention widths are independent of the input and output model width."""

import pytest
import torch

from stackformers.attention.config import CrossAttentionConfig, SelfAttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.self_attn import SelfAttention
from stackformers.positional.none import NoPosEncoding
from stackformers.sequence import PaddedInput, make_padded_input, padded_to_packed


@pytest.fixture
def queries(device: torch.device) -> PaddedInput:
    """Provide differentiable queries whose width is not divisible by twelve heads."""
    x = torch.linspace(-1.0, 1.0, 2 * 512, device=device).reshape(1, 2, 512)
    x.requires_grad_()
    return make_padded_input(x, torch.ones(1, 2, device=device, dtype=torch.bool))


@pytest.fixture
def context(device: torch.device) -> PaddedInput:
    """Provide independent differentiable context with a different sequence length."""
    x = torch.linspace(1.0, -1.0, 3 * 512, device=device).reshape(1, 3, 512)
    x.requires_grad_()
    return make_padded_input(x, torch.ones(1, 3, device=device, dtype=torch.bool))


def _assert_finite_nonzero_gradient(tensor: torch.Tensor) -> None:
    """Require the projected attention path to preserve a finite training signal."""
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()
    assert tensor.grad.abs().sum() > 0


@pytest.mark.parametrize("dim_head", [16, 64], ids=["compressed", "expanded"])
@pytest.mark.parametrize("kv_heads", [None, 3, 1], ids=["mha", "gqa", "mqa"])
def test_self_attention_independent_widths(
    queries: PaddedInput, dim_head: int, kv_heads: int | None
) -> None:
    """Nondivisible model widths preserve values, layout parity, and input gradients."""
    config = SelfAttentionConfig(dim=512, heads=12, dim_head=dim_head, kv_heads=kv_heads)
    attention = SelfAttention(config, NoPosEncoding()).to(device=queries.x.device)
    output = attention(queries)
    packed_output = attention(padded_to_packed(queries))

    assert output.shape == queries.x.shape
    assert torch.isfinite(output).all()
    torch.testing.assert_close(packed_output, output.reshape(-1, 512))
    output.square().mean().backward()
    _assert_finite_nonzero_gradient(queries.x)
    _assert_finite_nonzero_gradient(attention.to_q.weight)


@pytest.mark.parametrize("dim_head", [16, 64], ids=["compressed", "expanded"])
@pytest.mark.parametrize("kv_heads", [None, 3, 1], ids=["mha", "gqa", "mqa"])
def test_cross_attention_independent_widths(
    queries: PaddedInput, context: PaddedInput, dim_head: int, kv_heads: int | None
) -> None:
    """Independent widths preserve cross-attention layout parity and both input gradients."""
    config = CrossAttentionConfig(dim=512, heads=12, dim_head=dim_head, kv_heads=kv_heads)
    attention = CrossAttention(config, NoPosEncoding()).to(device=queries.x.device)
    output = attention(queries, context)
    packed_output = attention(padded_to_packed(queries), padded_to_packed(context))

    assert output.shape == queries.x.shape
    assert torch.isfinite(output).all()
    torch.testing.assert_close(packed_output, output.reshape(-1, 512))
    output.square().mean().backward()
    _assert_finite_nonzero_gradient(queries.x)
    _assert_finite_nonzero_gradient(context.x)
    _assert_finite_nonzero_gradient(attention.to_q.weight)
