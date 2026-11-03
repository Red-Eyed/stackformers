"""Construction-time rounding of internal model widths to positive multiples."""

from typing import Literal, assert_never

from pydantic import BaseModel, Field

from stackformers.result import Err, Ok, Result


class AlignmentConfig(BaseModel, frozen=True):
    """Choose a width multiple and direction; this changes parameter shapes, not padding."""

    multiple_of: int = Field(gt=0, description="Positive multiple required of the rounded width.")
    rounding_mode: Literal["floor", "ceil"] = "ceil"

    def round_dimension(self, dimension: int) -> Result[int, ValueError]:
        """Return a rounded positive width, or a failure when rounding produces zero."""
        match self.rounding_mode:
            case "floor":
                rounded = dimension // self.multiple_of * self.multiple_of
            case "ceil":
                rounded = (dimension + self.multiple_of - 1) // self.multiple_of * self.multiple_of
            case _:
                assert_never(self.rounding_mode)
        if rounded <= 0:
            return Err(ValueError("Dimension rounding must produce a positive width"))
        return Ok(rounded)
