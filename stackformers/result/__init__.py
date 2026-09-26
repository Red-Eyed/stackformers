"""Immutable success and failure records for expected outcomes."""

from typing import Generic, Literal, NamedTuple, TypeAlias, TypeVar, final

_T = TypeVar("_T", covariant=True)
_E = TypeVar("_E", covariant=True)


@final
class Ok(NamedTuple, Generic[_T]):
    """Carry a successful value without copying or wrapping its payload."""

    value: _T
    kind: Literal["ok"] = "ok"


@final
class Err(NamedTuple, Generic[_E]):
    """Carry a failure reason, preserving any original exception and traceback."""

    error: _E
    kind: Literal["err"] = "err"


Result: TypeAlias = Ok[_T] | Err[_E]

__all__ = ["Err", "Ok", "Result"]
