"""Shared equations preserve opaque attention state and skip absent context completely."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import pytest
import torch

from stackformers._decoder_equations import (
    NoContext,
    WithContext,
    post_norm_decoder,
    pre_norm_decoder,
    reordered_norm_decoder,
    sandwich_norm_decoder,
)
from stackformers.sequence import PaddedInput, make_padded_input
from tests.conftest import atol
from tests.norm_topology_helpers import ScaleFeedForward

if TYPE_CHECKING:
    from collections.abc import Callable


class EquationCase(NamedTuple):
    """The four equation functions differ only in the number of norm collaborators."""

    equation: Callable[..., tuple[PaddedInput, torch.Tensor]]
    norm_count: int


@pytest.fixture(
    params=[
        EquationCase(pre_norm_decoder, 3),
        EquationCase(post_norm_decoder, 3),
        EquationCase(reordered_norm_decoder, 3),
        EquationCase(sandwich_norm_decoder, 6),
    ],
    ids=["pre", "post", "reordered", "sandwich"],
)
def equation_case(request: pytest.FixtureRequest) -> EquationCase:
    """Exercise each equation through its attention/state callback contract."""
    case = request.param
    assert isinstance(case, EquationCase)
    return case


@pytest.fixture
def equation_input(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    """Build one native sequence record on every supported device and dtype."""
    device, dtype = device_dtype
    x = torch.randn(2, 3, 8, device=device, dtype=dtype)
    return make_padded_input(x, torch.ones(2, 3, dtype=torch.bool, device=device))


@pytest.mark.parametrize("with_context", [False, True])
def test_equation_carries_state_and_context(
    equation_case: EquationCase,
    equation_input: PaddedInput,
    device_dtype: tuple[torch.device, torch.dtype],
    with_context: bool,
) -> None:
    """Opaque state survives by identity; absent context never invokes its branch."""
    device, dtype = device_dtype
    state = torch.randn(5, device=device, dtype=dtype)
    cross_calls = 0

    def self_attention(input: PaddedInput) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an arbitrary state tensor that the layer equation must not interpret."""
        return input.x * 2, state

    def cross_attention(input: PaddedInput, context: PaddedInput) -> torch.Tensor:
        """Require the original context record and expose whether the branch ran."""
        nonlocal cross_calls
        assert context is equation_input
        cross_calls += 1
        return input.x * 3

    context = WithContext(equation_input) if with_context else NoContext()
    output, actual_state = equation_case.equation(
        equation_input,
        self_attention,
        cross_attention,
        context,
        ScaleFeedForward(4.0).to(device=device, dtype=dtype),
        *(torch.nn.Identity() for _ in range(equation_case.norm_count)),
    )

    expected = equation_input.x * (60 if with_context else 15)
    torch.testing.assert_close(output.x, expected, atol=atol(dtype), rtol=atol(dtype))
    assert actual_state is state
    assert output.mask is equation_input.mask
    assert output.abs_positions is equation_input.abs_positions
    assert cross_calls == int(with_context)
