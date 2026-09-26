"""NamedTuple Result records, PyTorch integration, and public exception contracts."""

from typing import TYPE_CHECKING, assert_never

import pytest
import torch
from torch import Tensor
from typing_extensions import override

from stackformers._result import unwrap_or_raise
from stackformers.result import Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("error_type", [ValueError, TypeError, NotImplementedError, AssertionError])
def test_boundary_preserves_exception_identity_and_cause(error_type: type[Exception]) -> None:
    """The compatibility boundary raises the original exception with its cause intact."""
    error = error_type("invalid input")
    cause = RuntimeError("original cause")
    error.__cause__ = cause
    outcome: Result[int, Exception] = Err(error)
    with pytest.raises(error_type, match="invalid input") as caught:
        unwrap_or_raise(outcome)
    assert caught.value is error
    assert caught.value.__cause__ is cause


def test_boundary_accepts_success_without_copying() -> None:
    """A successful record preserves tensor identity and autograd through the public adapter."""
    value = torch.ones(2, requires_grad=True)
    outcome: Result[Tensor, ValueError] = Ok(value)
    output = unwrap_or_raise(outcome)
    assert output is value
    output.sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value))


@pytest.mark.parametrize("payload", [0, False, None, ""])
def test_alternatives_keep_falsy_payloads_distinct(payload: int | bool | None | str) -> None:
    """Tuple equality and explicit tags preserve the variant even for equal falsy payloads."""
    success = Ok(payload)
    failure = Err(payload)
    assert success.value is payload
    assert failure.error is payload
    assert success != failure


@pytest.mark.parametrize("outcome", [Ok(7), Err("rejected")])
def test_records_are_immutable(outcome: Result[int, str]) -> None:
    """Outcome tags cannot be reassigned after construction."""
    with pytest.raises(AttributeError):
        object.__setattr__(outcome, "kind", "changed")


class _ResultBoundaryModel(torch.nn.Module):
    """Exercise Result construction and public boundary handling inside compiled inference."""

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Keep admitted tensor outputs native and reject empty sequences at the edge."""
        outcome: Result[Tensor, ValueError] = (
            Err(ValueError("empty sequence")) if x.shape[1] == 0 else Ok(x)
        )
        return unwrap_or_raise(outcome) + 1


@pytest.mark.parametrize("strict", [False, True])
def test_result_preserves_dynamic_torch_export(strict: bool) -> None:
    """Both tracing modes erase internal outcomes and retain resized tensor execution."""
    sample = torch.ones(2, 3)
    shapes = ({0: torch.export.Dim("batch", min=1), 1: torch.export.Dim("tokens", min=1)},)
    exported = torch.export.export(
        _ResultBoundaryModel(), (sample,), dynamic_shapes=shapes, strict=strict
    )
    resized = torch.ones(3, 5)
    output: Tensor = exported.module()(resized)
    torch.testing.assert_close(output, resized + 1)


def test_result_preserves_fullgraph_compile() -> None:
    """The adapter adds no graph break to a compiled tensor-only public call."""
    compiled: Callable[[Tensor], Tensor] = torch.compile(
        _ResultBoundaryModel(), backend="eager", fullgraph=True, dynamic=True
    )
    sample = torch.ones(2, 3)
    torch.testing.assert_close(compiled(sample), sample + 1)


class _ResultOutputModel(torch.nn.Module):
    """Expose a NamedTuple success record directly across the PyTorch output boundary."""

    @override
    def forward(self, x: Tensor) -> Ok[Tensor]:
        """Return a tensor-bearing result that PyTorch can flatten and reconstruct."""
        return Ok(x + 1)


@pytest.mark.parametrize("strict", [False, True])
def test_result_output_preserves_export_pytree(strict: bool) -> None:
    """Exported execution reconstructs the exact variant with dynamic native tensor leaves."""
    sample = torch.ones(2, 3)
    shapes = ({0: torch.export.Dim("batch", min=1), 1: torch.export.Dim("tokens", min=1)},)
    exported = torch.export.export(
        _ResultOutputModel(), (sample,), dynamic_shapes=shapes, strict=strict
    )
    resized = torch.ones(3, 5)
    output: Ok[Tensor] = exported.module()(resized)
    assert isinstance(output, Ok)
    assert output.kind == "ok"
    torch.testing.assert_close(output.value, resized + 1)


def test_result_output_preserves_fullgraph_compile() -> None:
    """Full-graph compilation preserves a tensor-bearing NamedTuple result."""
    compiled: Callable[[Tensor], Ok[Tensor]] = torch.compile(
        _ResultOutputModel(), backend="eager", fullgraph=True, dynamic=True
    )
    sample = torch.ones(2, 3)
    output = compiled(sample)
    assert isinstance(output, Ok)
    torch.testing.assert_close(output.value, sample + 1)


class _RecoverableResultModel(torch.nn.Module):
    """Handle an expected empty input inside the graph without raising."""

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Choose a typed failure and recover with a zero tensor for empty sequences."""
        outcome: Result[Tensor, str] = Err("empty sequence") if x.shape[1] == 0 else Ok(x)
        match outcome:
            case Ok(value=value):
                return value + 1
            case Err():
                return torch.zeros_like(x)
            case _:
                assert_never(outcome)


def test_recoverable_failure_stays_data_in_fullgraph_compile() -> None:
    """Both admission and recovery execute under full-graph compilation without exceptions."""
    compiled: Callable[[Tensor], Tensor] = torch.compile(
        _RecoverableResultModel(), backend="eager", fullgraph=True, dynamic=True
    )
    sample = torch.ones(2, 3)
    torch.testing.assert_close(compiled(sample), sample + 1)
    empty = torch.ones(2, 0)
    torch.testing.assert_close(compiled(empty), torch.zeros_like(empty))
