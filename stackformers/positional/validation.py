"""Pure rotary-width admission checks that preserve constructor failure detail."""

from returns.result import Failure, Result, Success


def rotary_head_width(
    dim_head: int, divisor: int, message: str
) -> Result[tuple[()], AssertionError]:
    """Return an invalid pairing width as data for the constructor boundary."""
    if dim_head % divisor != 0:
        return Failure(AssertionError(message))
    return Success(())
