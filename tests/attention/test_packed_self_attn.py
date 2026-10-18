"""Packed QKV parity across head sharing, layouts, visibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from torch._dynamo.exc import UserError

from stackformers.attention.config import SelfAttentionConfig
from stackformers.attention.self_attn import SelfAttention
from stackformers.positional.config import RoPE1DConfig
from stackformers.positional.rope1d import RotaryEmbedding1D
from stackformers.sequence import SequenceInput, make_packed_input, make_padded_input
from tests.conftest import atol

if TYPE_CHECKING:
    from stackformers.attention.masking import SelfAttentionMask


@pytest.fixture(params=[1, 2, 4])
def source(
    request: pytest.FixtureRequest, device_dtype: tuple[torch.device, torch.dtype]
) -> SelfAttention:
    """Exercise MQA, GQA, and MHA with rotary positions and QK normalization."""
    device, dtype = device_dtype
    config = SelfAttentionConfig(
        dim=32, heads=4, dim_head=8, kv_heads=request.param, causal=True, qk_norm=True
    )
    return (
        SelfAttention(config, RotaryEmbedding1D(RoPE1DConfig(dim_head=8)))
        .to(device=device, dtype=dtype)
        .eval()
    )


@pytest.fixture(params=[False, True], ids=["padded", "packed"])
def sequence(request: pytest.FixtureRequest, source: SelfAttention) -> SequenceInput:
    """Represent two unequal documents in either supported layout."""
    weight = source.to_q.weight
    if request.param:
        x = torch.randn(8, 32, device=weight.device, dtype=weight.dtype)
        cu = torch.tensor([0, 5, 8], device=weight.device, dtype=torch.int32)
        return make_packed_input(x, cu, max_seqlen=5)
    x = torch.randn(2, 5, 32, device=weight.device, dtype=weight.dtype)
    mask = (
        torch.arange(5, device=weight.device)[None]
        < torch.tensor([5, 3], device=weight.device)[:, None]
    )
    return make_padded_input(x, mask)


@pytest.mark.parametrize("visibility", ["default", "causal", "bidirectional", "explicit"])
def test_forward_parity(source: SelfAttention, sequence: SequenceInput, visibility: str) -> None:
    """Packing preserves position, normalization, masks, and output order."""
    packed = source.to_packed()
    mask: SelfAttentionMask
    match visibility:
        case "default" | "causal" | "bidirectional":
            mask = visibility
        case _:
            mask = torch.eye(5, dtype=torch.bool, device=sequence.x.device)
    expected = source(sequence, self_attn_mask=mask)
    actual = packed(sequence, self_attn_mask=mask)
    torch.testing.assert_close(actual, expected, atol=atol(actual.dtype), rtol=atol(actual.dtype))
    for name, value in source.state_dict().items():
        torch.testing.assert_close(packed.unpack().state_dict()[name], value, rtol=0, atol=0)


def test_gradients(device: torch.device) -> None:
    """Unequal Q/KV row groups receive the original projection gradients."""
    source = SelfAttention(
        SelfAttentionConfig(dim=32, heads=4, dim_head=8, kv_heads=2, qk_norm=True),
        RotaryEmbedding1D(RoPE1DConfig(dim_head=8)),
    ).to(device)
    packed = source.to_packed()
    x = torch.randn(2, 5, 32, device=device, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    mask = torch.ones(2, 5, device=device, dtype=torch.bool)
    source(make_padded_input(x, mask)).square().sum().backward()
    packed(make_padded_input(other, mask)).square().sum().backward()
    torch.testing.assert_close(x.grad, other.grad)
    assert packed.qkv.weight.grad is not None
    for actual, layer in zip(
        packed.qkv.weight.grad.split((32, 16, 16)),
        (source.to_q, source.to_k, source.to_v),
        strict=True,
    ):
        torch.testing.assert_close(actual, layer.weight.grad)
    for name in ("to_out.weight", "q_norm.weight", "k_norm.weight"):
        torch.testing.assert_close(packed.get_parameter(name).grad, source.get_parameter(name).grad)


def test_dynamic_export(source: SelfAttention) -> None:
    """One exported graph handles singleton and longer sequence axes."""
    packed = source.to_packed()
    weight = source.to_q.weight
    example = make_padded_input(
        torch.randn(2, 3, 32, device=weight.device, dtype=weight.dtype),
        torch.ones(2, 3, device=weight.device, dtype=torch.bool),
    )
    shapes = torch.export.ShapesCollection()
    length = torch.export.Dim("tokens", min=1, max=32)
    for tensor in example:
        shapes[tensor] = {1: length}
    try:
        program = torch.export.export(
            packed, (example,), dynamic_shapes=shapes.dynamic_shapes(packed, (example,))
        ).module()
    except UserError as error:
        if not torch.__version__.startswith("2.11.") or "min(8*" not in str(error):
            raise
        # This exporter guard also rejects the ordinary implementation on PyTorch 2.11.
        with pytest.raises(UserError, match=r"min\(8\*"):
            torch.export.export(
                source, (example,), dynamic_shapes=shapes.dynamic_shapes(source, (example,))
            )
        pytest.xfail("PyTorch 2.11 cannot prove the existing GQA stride guards")
    for tokens in (1, 2, 7):
        sequence = make_padded_input(
            torch.randn(2, tokens, 32, device=weight.device, dtype=weight.dtype),
            torch.ones(2, tokens, device=weight.device, dtype=torch.bool),
        )
        torch.testing.assert_close(program(sequence), packed(sequence))


def test_window_and_bias(source: SelfAttention, sequence: SequenceInput) -> None:
    """Learned bias and local visibility survive conversion in either sequence layout."""
    from stackformers.attention.config import DistanceBiasConfig
    from stackformers.attention.distance_bias import RelativeDistanceBias

    source.config = source.config.model_copy(update={"window_size": 3})
    source.attn_bias = RelativeDistanceBias(DistanceBiasConfig(heads=4, r_max=10, num_rbf=4)).to(
        device=sequence.x.device, dtype=sequence.x.dtype
    )
    packed = source.to_packed()
    expected = source(sequence)
    actual = packed(sequence)
    torch.testing.assert_close(actual, expected, atol=atol(actual.dtype), rtol=atol(actual.dtype))
