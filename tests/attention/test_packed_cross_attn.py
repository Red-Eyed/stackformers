"""Packed context KV parity for independent lengths and checkpoint conversion."""

from __future__ import annotations

import pytest
import torch
from torch._dynamo.exc import UserError

from stackformers.attention.config import CrossAttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.layout import call_cross_attention
from stackformers.positional.config import RoPE1DConfig
from stackformers.positional.rope1d import RotaryEmbedding1D
from stackformers.sequence import (
    SequenceInput,
    make_packed_input,
    make_padded_input,
)
from tests.conftest import atol


@pytest.fixture(params=[1, 2, 4])
def source(
    request: pytest.FixtureRequest, device_dtype: tuple[torch.device, torch.dtype]
) -> CrossAttention:
    """Cover all head-sharing geometries with normalized, rotary Q/K."""
    device, dtype = device_dtype
    return (
        CrossAttention(
            CrossAttentionConfig(dim=32, heads=4, dim_head=8, kv_heads=request.param, qk_norm=True),
            RotaryEmbedding1D(RoPE1DConfig(dim_head=8)),
        )
        .to(device=device, dtype=dtype)
        .eval()
    )


@pytest.fixture(params=[False, True], ids=["padded", "packed"])
def inputs(
    request: pytest.FixtureRequest, source: CrossAttention
) -> tuple[SequenceInput, SequenceInput]:
    """Use unequal query and context lengths with document padding."""
    weight = source.to_q.weight
    query = torch.randn(2, 3, 32, device=weight.device, dtype=weight.dtype)
    context = torch.randn(2, 5, 32, device=weight.device, dtype=weight.dtype)
    if request.param:
        cu_q = torch.tensor([0, 3, 5], device=weight.device, dtype=torch.int32)
        cu_kv = torch.tensor([0, 5, 8], device=weight.device, dtype=torch.int32)
        return (
            make_packed_input(query.reshape(6, 32)[:5], cu_q, max_seqlen=3),
            make_packed_input(context.reshape(10, 32)[:8], cu_kv, max_seqlen=5),
        )
    query_mask = torch.ones(2, 3, dtype=torch.bool, device=weight.device)
    query_mask[1, -1] = False
    context_mask = torch.ones(2, 5, dtype=torch.bool, device=weight.device)
    context_mask[1, 3:] = False
    return make_padded_input(query, query_mask), make_padded_input(context, context_mask)


@pytest.mark.parametrize("explicit", [False, True])
def test_forward_parity(
    source: CrossAttention, inputs: tuple[SequenceInput, SequenceInput], explicit: bool
) -> None:
    """Packing preserves masked cross-context attention and the original checkpoint."""
    query, context = inputs
    packed = source.to_packed()
    mask = torch.ones(3, 5, device=query.x.device, dtype=torch.bool)
    mask[1] = False
    if explicit:
        expected = call_cross_attention(source, query, context, cross_attn_mask=mask)
        actual = call_cross_attention(packed, query, context, cross_attn_mask=mask)
    else:
        expected = call_cross_attention(source, query, context)
        actual = call_cross_attention(packed, query, context)
    torch.testing.assert_close(actual, expected, atol=atol(actual.dtype), rtol=atol(actual.dtype))
    restored = packed.unpack()
    for name, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


def test_gradients(device: torch.device) -> None:
    """Packing preserves gradients through independent query/context inputs."""
    source = CrossAttention(
        CrossAttentionConfig(dim=32, heads=4, dim_head=8, kv_heads=2, qk_norm=True),
        RotaryEmbedding1D(RoPE1DConfig(dim_head=8)),
    ).to(device)
    packed = source.to_packed()
    query = torch.randn(2, 3, 32, device=device, requires_grad=True)
    context = torch.randn(2, 7, 32, device=device, requires_grad=True)
    other_query = query.detach().clone().requires_grad_()
    other_context = context.detach().clone().requires_grad_()
    q_mask = torch.ones(2, 3, device=device, dtype=torch.bool)
    kv_mask = torch.ones(2, 7, device=device, dtype=torch.bool)
    source(
        make_padded_input(query, q_mask), make_padded_input(context, kv_mask)
    ).square().sum().backward()
    packed(
        make_padded_input(other_query, q_mask), make_padded_input(other_context, kv_mask)
    ).square().sum().backward()
    torch.testing.assert_close(query.grad, other_query.grad)
    torch.testing.assert_close(context.grad, other_context.grad)
    assert packed.kv.weight.grad is not None
    key, value = packed.kv.weight.grad.chunk(2)
    torch.testing.assert_close(key, source.to_k.weight.grad)
    torch.testing.assert_close(value, source.to_v.weight.grad)
    for name in ("to_q.weight", "to_out.weight", "q_norm.weight", "k_norm.weight"):
        torch.testing.assert_close(packed.get_parameter(name).grad, source.get_parameter(name).grad)


def test_dynamic_export(source: CrossAttention) -> None:
    """One graph retains independent query and context sequence dimensions."""
    packed = source.to_packed()
    weight = source.to_q.weight
    query = make_padded_input(
        torch.randn(2, 3, 32, device=weight.device, dtype=weight.dtype),
        torch.ones(2, 3, device=weight.device, dtype=torch.bool),
    )
    context = make_padded_input(
        torch.randn(2, 5, 32, device=weight.device, dtype=weight.dtype),
        torch.ones(2, 5, device=weight.device, dtype=torch.bool),
    )
    shapes = torch.export.ShapesCollection()
    query_length = torch.export.Dim("queries", min=1, max=32)
    context_length = torch.export.Dim("context", min=1, max=32)
    for tensor in query:
        shapes[tensor] = {1: query_length}
    for tensor in context:
        shapes[tensor] = {1: context_length}
    try:
        program = torch.export.export(
            packed, (query, context), dynamic_shapes=shapes.dynamic_shapes(packed, (query, context))
        ).module()
    except UserError as error:
        if not torch.__version__.startswith("2.11.") or "min(8*" not in str(error):
            raise
        # This exporter guard also rejects the ordinary implementation on PyTorch 2.11.
        with pytest.raises(UserError, match=r"min\(8\*"):
            torch.export.export(
                source,
                (query, context),
                dynamic_shapes=shapes.dynamic_shapes(source, (query, context)),
            )
        pytest.xfail("PyTorch 2.11 cannot prove the existing GQA stride guards")
    for queries, context_tokens in ((1, 1), (2, 7), (7, 2)):
        resized_query = make_padded_input(
            query.x[:, :1].expand(-1, queries, -1).contiguous(),
            torch.ones(2, queries, device=weight.device, dtype=torch.bool),
        )
        resized_context = make_padded_input(
            context.x[:, :1].expand(-1, context_tokens, -1).contiguous(),
            torch.ones(2, context_tokens, device=weight.device, dtype=torch.bool),
        )
        torch.testing.assert_close(
            program(resized_query, resized_context), packed(resized_query, resized_context)
        )
