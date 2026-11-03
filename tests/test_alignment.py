"""Positive width rounding preserves exact multiples and reports unusable floors."""

import pytest
from pydantic import ValidationError

from stackformers import AlignmentConfig
from stackformers.result import Err, Ok


@pytest.mark.parametrize(
    ("mode", "dimension", "expected"),
    [("floor", 682, 640), ("ceil", 682, 704), ("floor", 64, 64), ("ceil", 64, 64), ("ceil", 1, 64)],
)
def test_round_dimension(mode: str, dimension: int, expected: int) -> None:
    """Rounding is directional and leaves exact multiples unchanged."""
    config = AlignmentConfig.model_validate({"multiple_of": 64, "rounding_mode": mode})
    assert config.round_dimension(dimension) == Ok(expected)


def test_zero_floor() -> None:
    """A floor below one multiple is a recoverable failure."""
    result = AlignmentConfig(multiple_of=64, rounding_mode="floor").round_dimension(63)
    assert isinstance(result, Err)
    assert str(result.error) == "Dimension rounding must produce a positive width"


@pytest.mark.parametrize(
    "payload",
    [{"multiple_of": 0}, {"multiple_of": -8}, {"multiple_of": 64, "rounding_mode": "nearest"}],
)
def test_invalid_policy(payload: dict[str, int | str]) -> None:
    """Reject invalid multiples and unsupported rounding modes at the boundary."""
    with pytest.raises(ValidationError):
        AlignmentConfig.model_validate(payload)
