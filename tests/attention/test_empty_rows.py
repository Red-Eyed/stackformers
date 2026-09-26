"""Mask exclusion is exact even when padding, windows, or bias remove every key."""

import pytest
import torch

from stackformers.attention.config import CrossAttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.ops import padded_sdpa
from stackformers.positional.none import NoPosEncoding
from stackformers.sequence import make_padded_input, padded_to_packed


@pytest.mark.parametrize(
    ("causal", "window", "expected"),
    [
        (False, None, (11.0, 11.0, 11.0)),
        (True, None, (0.0, 8.0, 11.0)),
        (False, 1, (0.0, 8.0, 14.0)),
        (True, 1, (0.0, 8.0, 11.0)),
    ],
)
def test_excluded_keys_have_zero_output_and_gradient(
    causal: bool,
    window: int | None,
    expected: tuple[float, float, float],
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """Empty rows contribute nothing; remaining rows average only permitted values."""
    device, dtype = device_dtype
    q = torch.zeros(2, 1, 3, 1, device=device, dtype=dtype, requires_grad=True)
    k = torch.zeros_like(q, requires_grad=True)
    v = torch.tensor([2.0, 8.0, 14.0], device=device, dtype=dtype)
    v = v.reshape(1, 1, 3, 1).repeat(2, 1, 1, 1).requires_grad_()
    mask = torch.tensor([[False, False, False], [False, True, True]], device=device)

    out = padded_sdpa(q, k, v, mask, causal, window, None)

    assert out[0].eq(0).all()
    torch.testing.assert_close(out[1, 0, :, 0], out.new_tensor(expected))
    out.sum().backward()
    for tensor in (q, k, v):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad[0].eq(0).all()
    assert v.grad is not None
    assert v.grad[1, :, 0].eq(0).all()


def test_bias_can_exclude_a_single_head() -> None:
    """Empty-row handling respects per-head exclusions introduced by additive bias."""
    q = torch.zeros(1, 2, 2, 1)
    bias = torch.zeros(1, 2, 2, 2)
    bias[:, 0] = float("-inf")
    out = padded_sdpa(
        q, q, torch.ones_like(q), torch.ones(1, 2, dtype=torch.bool), False, None, bias
    )
    assert out[:, 0].eq(0).all()
    assert out[:, 1].eq(1).all()


@pytest.mark.parametrize("packed", [False, True], ids=["padded", "packed"])
def test_cross_attention_empty_context(packed: bool) -> None:
    """A document with no valid context yields zero even beside a nonempty document."""
    attention = CrossAttention(CrossAttentionConfig(dim=64, heads=1), NoPosEncoding())
    queries = make_padded_input(torch.randn(2, 1, 64), torch.ones(2, 1, dtype=torch.bool))
    context = make_padded_input(torch.randn(2, 2, 64), torch.tensor([[False, False], [True, True]]))
    if packed:
        out = attention(padded_to_packed(queries), padded_to_packed(context))
    else:
        out = attention(queries, context)
    assert out[0].eq(0).all()
    assert torch.isfinite(out).all()
