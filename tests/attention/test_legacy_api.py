"""Existing attention API calls retain RC1 checkpoint, output, and gradient behavior."""

import math
from pathlib import Path
from typing import Literal

import pytest
import torch
from pydantic import BaseModel

from stackformers.attention.config import CrossAttentionConfig, SelfAttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.self_attn import SelfAttention
from stackformers.positional.none import NoPosEncoding
from stackformers.sequence import PaddedInput, make_padded_input, padded_to_packed


class LegacyCase(BaseModel, frozen=True):
    """Frozen observations and checkpoint shapes captured from a released model."""

    name: str
    kind: Literal["self", "cross"]
    defaults: bool
    heads: int
    dim_head: int
    kv_heads: int | None
    qk_norm: bool
    causal: bool
    state_shapes: dict[str, list[int]]
    output: list[list[list[float]]]
    query_gradient: list[list[list[float]]]
    context_gradient: list[list[list[float]]]


class LegacyBaseline(BaseModel, frozen=True):
    """Identify the immutable release used to generate these reference observations."""

    version: Literal["5.0.0rc1"]
    commit: Literal["c40fbaffac96e0dac6a5678b8e4452de1df8da3d"]
    cases: list[LegacyCase]


@pytest.fixture(scope="module")
def legacy_baseline() -> LegacyBaseline:
    """Parse the released-wheel observations once at the JSON boundary."""
    path = Path(__file__).with_name("legacy_api_rc1.json")
    return LegacyBaseline.model_validate_json(path.read_text())


@pytest.fixture
def legacy_case(legacy_baseline: LegacyBaseline, case_name: str) -> LegacyCase:
    """Select an independently captured reference for the requested old configuration."""
    return next(case for case in legacy_baseline.cases if case.name == case_name)


@pytest.fixture
def legacy_queries() -> PaddedInput:
    """Recreate the released baseline's differentiable double-precision queries."""
    x = torch.linspace(-0.8, 0.9, 16, dtype=torch.float64).reshape(1, 2, 8)
    x.requires_grad_()
    return make_padded_input(x, torch.ones(1, 2, dtype=torch.bool))


@pytest.fixture
def legacy_context() -> PaddedInput:
    """Recreate the released baseline's independent differentiable context."""
    x = torch.linspace(0.7, -0.6, 24, dtype=torch.float64).reshape(1, 3, 8)
    x.requires_grad_()
    return make_padded_input(x, torch.ones(1, 3, dtype=torch.bool))


def _legacy_self_config(case: LegacyCase) -> SelfAttentionConfig:
    """Exercise the old self-attention config API with explicit or omitted settings."""
    if case.defaults:
        return SelfAttentionConfig(dim=8)
    return SelfAttentionConfig(
        dim=8,
        heads=case.heads,
        dim_head=case.dim_head,
        kv_heads=case.kv_heads,
        qk_norm=case.qk_norm,
        causal=case.causal,
    )


def _legacy_cross_config(case: LegacyCase) -> CrossAttentionConfig:
    """Exercise the old cross-attention config API with explicit or omitted settings."""
    if case.defaults:
        return CrossAttentionConfig(dim=8)
    return CrossAttentionConfig(
        dim=8,
        heads=case.heads,
        dim_head=case.dim_head,
        kv_heads=case.kv_heads,
        qk_norm=case.qk_norm,
    )


def _legacy_attention(case: LegacyCase, keyword_api: bool) -> SelfAttention | CrossAttention:
    """Construct through old positional or keyword module APIs."""
    match case.kind:
        case "self":
            config = _legacy_self_config(case)
            return (
                SelfAttention(config=config, pos_encoding=NoPosEncoding())
                if keyword_api
                else SelfAttention(config, NoPosEncoding())
            ).double()
        case "cross":
            cross_config = _legacy_cross_config(case)
            return (
                CrossAttention(config=cross_config, pos_encoding=NoPosEncoding())
                if keyword_api
                else CrossAttention(cross_config, NoPosEncoding())
            ).double()


def _legacy_checkpoint(case: LegacyCase) -> dict[str, torch.Tensor]:
    """Recreate old parameter values from frozen key/shape metadata independently of the model."""
    return {
        key: torch.linspace(-0.15, 0.15, math.prod(shape), dtype=torch.float64)
        .reshape(shape)
        .add(index * 0.01)
        for index, (key, shape) in enumerate(sorted(case.state_shapes.items()))
    }


def _cross_output(
    attention: CrossAttention,
    queries: PaddedInput,
    context: PaddedInput,
    packed: bool,
    keyword_api: bool,
) -> torch.Tensor:
    """Exercise the old cross-attention calls with jointly narrowed sequence layouts."""
    if packed:
        packed_queries, packed_context = padded_to_packed(queries), padded_to_packed(context)
        return (
            attention(x_input=packed_queries, ctx_input=packed_context)
            if keyword_api
            else attention(packed_queries, packed_context)
        )
    return (
        attention(x_input=queries, ctx_input=context)
        if keyword_api
        else attention(queries, context)
    )


@pytest.mark.parametrize(
    "case_name",
    [
        f"{kind}-{geometry}"
        for kind in ("self", "cross")
        for geometry in ("defaults", "square", "compressed", "expanded", "gqa", "mqa", "normalized")
    ]
    + ["self-causal"],
)
@pytest.mark.parametrize("packed", [False, True], ids=["padded", "packed"])
@pytest.mark.parametrize("keyword_api", [False, True], ids=["positional", "keyword"])
def test_existing_attention_api_matches_rc1(
    legacy_case: LegacyCase,
    legacy_queries: PaddedInput,
    legacy_context: PaddedInput,
    packed: bool,
    keyword_api: bool,
) -> None:
    """Old calls load old checkpoints strictly and retain reference outputs and input gradients."""
    attention = _legacy_attention(legacy_case, keyword_api)
    assert attention.config.heads == legacy_case.heads
    assert attention.config.dim_head == legacy_case.dim_head
    assert attention.config.kv_heads == legacy_case.kv_heads
    assert {key: list(value.shape) for key, value in attention.state_dict().items()} == (
        legacy_case.state_shapes
    )
    attention.load_state_dict(_legacy_checkpoint(legacy_case), strict=True)

    match attention:
        case SelfAttention():
            queries = padded_to_packed(legacy_queries) if packed else legacy_queries
            output = attention(input=queries) if keyword_api else attention(queries)
        case CrossAttention():
            output = _cross_output(attention, legacy_queries, legacy_context, packed, keyword_api)

    expected = torch.tensor(legacy_case.output, dtype=torch.float64)
    torch.testing.assert_close(
        output, expected.reshape(-1, 8) if packed else expected, rtol=1e-10, atol=1e-12
    )
    output.square().mean().backward()
    torch.testing.assert_close(
        legacy_queries.x.grad,
        torch.tensor(legacy_case.query_gradient, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    )
    if legacy_case.kind == "cross":
        torch.testing.assert_close(
            legacy_context.x.grad,
            torch.tensor(legacy_case.context_gradient, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-12,
        )
