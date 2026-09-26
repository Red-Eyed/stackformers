"""Adapt core outcomes to public operations that cannot proceed with rejected inputs."""

from typing import TypeVar, assert_never

from stackformers.result import Err, Ok, Result

T = TypeVar("T")


def unwrap_or_raise(result: Result[T, Exception]) -> T:
    """Return success or raise the exact stored exception, preserving type and cause."""
    match result:
        case Err(error=error):
            raise error
        case Ok(value=value):
            return value
        case _:
            assert_never(result)
