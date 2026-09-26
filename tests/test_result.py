"""Shared Result interoperability and exception-preserving API boundary contracts."""

from typing import TYPE_CHECKING

import pytest
import torch
from returns.result import Failure, Result, Success
from torch import Tensor
from typing_extensions import override

from stackformers._result import unwrap_or_raise

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("error_type", [ValueError, TypeError, NotImplementedError, AssertionError])
def test_boundary_preserves_exception_identity_and_cause(error_type: type[Exception]) -> None:
    """Shared failure values raise their original exception rather than UnwrapFailedError."""
    error = error_type("invalid input")
    cause = RuntimeError("original cause")
    error.__cause__ = cause
    outcome: Result[int, Exception] = Failure(error)
    with pytest.raises(error_type, match="invalid input") as caught:
        unwrap_or_raise(outcome)
    assert caught.value is error
    assert caught.value.__cause__ is cause


def test_boundary_accepts_shared_success_without_copying() -> None:
    """A returns success from another library preserves tensor identity and autograd."""
    value = torch.ones(2, requires_grad=True)
    outcome: Result[Tensor, ValueError] = Success(value)
    output = unwrap_or_raise(outcome)
    assert output is value
    output.sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value))


def test_shared_map_preserves_failure_without_running_callback() -> None:
    """A core outcome composes with returns without a project-specific conversion."""
    error = ValueError("invalid input")
    outcome: Result[int, ValueError] = Failure(error)

    def render(value: int) -> str:
        """Fail if shared composition evaluates a success callback on a failure."""
        pytest.fail("map must skip the failure branch")

    mapped: Result[str, ValueError] = outcome.map(render)
    assert mapped.failure() is error


class _ResultBoundaryModel(torch.nn.Module):
    """Exercise shared Result construction and handling inside compiled inference."""

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Keep admitted tensor outputs native and reject empty sequences at the edge."""
        outcome: Result[Tensor, ValueError] = (
            Failure(ValueError("empty sequence")) if x.shape[1] == 0 else Success(x)
        )
        return unwrap_or_raise(outcome) + 1


@pytest.mark.parametrize("strict", [False, True])
def test_shared_result_preserves_dynamic_torch_export(strict: bool) -> None:
    """Both tracing modes erase internal outcomes and retain resized tensor execution."""
    sample = torch.ones(2, 3)
    shapes = ({0: torch.export.Dim("batch", min=1), 1: torch.export.Dim("tokens", min=1)},)
    exported = torch.export.export(
        _ResultBoundaryModel(), (sample,), dynamic_shapes=shapes, strict=strict
    )
    resized = torch.ones(3, 5)
    output: Tensor = exported.module()(resized)
    torch.testing.assert_close(output, resized + 1)


def test_shared_result_preserves_fullgraph_compile() -> None:
    """The adapter adds no graph break to a compiled tensor-only public call."""
    compiled: Callable[[Tensor], Tensor] = torch.compile(
        _ResultBoundaryModel(), backend="eager", fullgraph=True, dynamic=True
    )
    sample = torch.ones(2, 3)
    torch.testing.assert_close(compiled(sample), sample + 1)
