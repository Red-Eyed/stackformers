from __future__ import annotations

import pytest
import torch

from stackformers.attention.config import CrossAttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.positional.none import NoPosEncoding
from stackformers.sequence import PaddedInput, make_padded_input, materialize_padding_mask

B, N, S, D, H, DH = 2, 8, 12, 64, 4, 16


@pytest.fixture
def cross_attn(device_dtype: tuple[torch.device, torch.dtype]) -> CrossAttention:
    device, dtype = device_dtype
    config = CrossAttentionConfig(dim=D, heads=H, dim_head=DH)
    return CrossAttention(config=config, pos_encoding=NoPosEncoding()).to(
        device=device, dtype=dtype
    )


@pytest.fixture
def x_ctx_input(
    device_dtype: tuple[torch.device, torch.dtype],
) -> tuple[torch.Tensor, torch.Tensor]:
    device, dtype = device_dtype
    return (
        torch.randn(B, N, D, device=device, dtype=dtype),
        torch.randn(B, S, D, device=device, dtype=dtype),
    )


def test_cross_attn_output_shape(
    cross_attn: CrossAttention,
    x_ctx_input: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Output keeps x's sequence length and model dim regardless of context length."""
    x, ctx = x_ctx_input
    device = x.device
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool, device=device))
    ctx_inp = make_padded_input(ctx, torch.ones(B, S, dtype=torch.bool, device=device))
    assert cross_attn(x_inp, ctx_inp).shape == (B, N, D)


def test_cross_attn_with_ctx_mask(
    cross_attn: CrossAttention,
    x_ctx_input: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Padding on the context (key) side does not crash and preserves output shape."""
    x, ctx = x_ctx_input
    device = x.device
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool, device=device))
    mask = torch.ones(B, S, dtype=torch.bool, device=device)
    mask[0, 10:] = False
    ctx_inp = make_padded_input(ctx, mask)
    assert cross_attn(x_inp, ctx_inp).shape == (B, N, D)


def test_cross_attn_with_x_mask(
    cross_attn: CrossAttention,
    x_ctx_input: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Padding on x zeros out output positions beyond the query mask boundary."""
    x, ctx = x_ctx_input
    device = x.device
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    mask[0, 5:] = False
    x_inp = make_padded_input(x, mask)
    ctx_inp = make_padded_input(ctx, torch.ones(B, S, dtype=torch.bool, device=device))
    out = cross_attn(x_inp, ctx_inp)
    assert out.shape == (B, N, D)
    assert out[0, 5:].eq(0).all()


def test_cross_attn_output_driven_by_context(
    cross_attn: CrossAttention,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """K/V come from context: with x=0, changing context changes output."""
    device, dtype = device_dtype
    x = torch.zeros(B, N, D, device=device, dtype=dtype)
    ctx1 = torch.randn(B, S, D, device=device, dtype=dtype)
    ctx2 = torch.randn(B, S, D, device=device, dtype=dtype)
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool, device=device))
    ctx1_inp = make_padded_input(ctx1, torch.ones(B, S, dtype=torch.bool, device=device))
    ctx2_inp = make_padded_input(ctx2, torch.ones(B, S, dtype=torch.bool, device=device))
    with torch.no_grad():
        out1 = cross_attn(x_inp, ctx1_inp)
        out2 = cross_attn(x_inp, ctx2_inp)
    assert not torch.allclose(out1, out2)


def test_cross_attn_different_seq_lengths(
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """x and context may have different sequence lengths; output shape follows x."""
    device, dtype = device_dtype
    config = CrossAttentionConfig(dim=D, heads=H, dim_head=DH)
    attn = CrossAttention(config=config, pos_encoding=NoPosEncoding()).to(
        device=device, dtype=dtype
    )
    x = torch.randn(B, 5, D, device=device, dtype=dtype)
    ctx = torch.randn(B, 20, D, device=device, dtype=dtype)
    x_inp = make_padded_input(x, torch.ones(B, 5, dtype=torch.bool, device=device))
    ctx_inp = make_padded_input(ctx, torch.ones(B, 20, dtype=torch.bool, device=device))
    assert attn(x_inp, ctx_inp).shape == (B, 5, D)


def test_cross_attn_gqa(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    config = CrossAttentionConfig(dim=D, heads=H, dim_head=DH, kv_heads=2)
    attn = CrossAttention(config=config, pos_encoding=NoPosEncoding()).to(
        device=device, dtype=dtype
    )
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    ctx = torch.randn(B, S, D, device=device, dtype=dtype)
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool, device=device))
    ctx_inp = make_padded_input(ctx, torch.ones(B, S, dtype=torch.bool, device=device))
    assert attn(x_inp, ctx_inp).shape == (B, N, D)


@pytest.fixture
def mean_cross_attention(device_dtype: tuple[torch.device, torch.dtype]) -> CrossAttention:
    """Use uniform scores and identity values to expose exact cross visibility."""
    device, dtype = device_dtype
    attention = CrossAttention(
        CrossAttentionConfig(dim=4, heads=1, dim_head=4), NoPosEncoding()
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        attention.to_q.weight.zero_()
        attention.to_k.weight.zero_()
        attention.to_v.weight.copy_(torch.eye(4, device=device, dtype=dtype))
        attention.to_out.weight.copy_(torch.eye(4, device=device, dtype=dtype))
    return attention


@pytest.fixture
def visibility_cross_inputs(
    device_dtype: tuple[torch.device, torch.dtype],
) -> tuple[PaddedInput, PaddedInput]:
    """Construct two documents with different lengths on each side of cross-attention."""
    device, dtype = device_dtype
    queries = make_padded_input(
        torch.ones(2, 3, 4, device=device, dtype=dtype),
        torch.tensor([[True, True, True], [True, True, False]], device=device),
    )
    values = torch.tensor([[1, 3, 9, 999], [10, 20, 30, 40]], device=device, dtype=dtype)
    context = make_padded_input(
        values[..., None].expand(-1, -1, 4),
        torch.tensor([[True, True, True, False], [True] * 4], device=device),
    )
    return queries, context


@pytest.mark.parametrize("packed", [False, True])
def test_runtime_cross_visibility(
    mean_cross_attention: CrossAttention,
    visibility_cross_inputs: tuple[PaddedInput, PaddedInput],
    packed: bool,
) -> None:
    """A shared rectangular mask composes with context/query validity and empty rows."""
    from stackformers.sequence import padded_to_packed
    from tests.conftest import atol

    queries, context = visibility_cross_inputs
    allowed = torch.tensor(
        [[True, False, False, True], [False, False, False, True], [False] * 4],
        device=queries.x.device,
    )
    expected = queries.x.new_tensor([[1, 0, 0], [25, 40, 0]])[..., None].expand(-1, -1, 4)
    if packed:
        expected = expected[materialize_padding_mask(queries)]
        actual = mean_cross_attention(
            padded_to_packed(queries), padded_to_packed(context), cross_attn_mask=allowed
        )
    else:
        actual = mean_cross_attention(queries, context, cross_attn_mask=allowed)
    torch.testing.assert_close(actual, expected, atol=atol(actual.dtype), rtol=0)
