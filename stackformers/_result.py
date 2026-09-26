"""Adapt shared returns outcomes to the established public exception contract."""

from typing import TypeVar

from returns.result import Failure, Result

T = TypeVar("T")


def unwrap_or_raise(result: Result[T, Exception]) -> T:
    """Return success or raise the exact stored exception, preserving type and cause."""
    match result:
        case Failure(error):
            raise error
        case _:
            return result.unwrap()
