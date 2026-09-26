from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import override

if TYPE_CHECKING:
    from jaxtyping import Float


class RegressionHead(nn.Module):
    """Reconstruct masked tokens via a linear projection, scored with MSE.

    The projection gives the encoder a dedicated place to specialise for
    reconstruction, so the shared representation isn't forced to double as the
    literal token vector at every layer.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    @override
    def forward(
        self,
        encoder_output_at_masked: Float[Tensor, "m d"],
        target_at_masked: Float[Tensor, "m d"],
    ) -> Float[Tensor, ""]:
        """Return mean squared error, or a differentiable zero for an empty selection."""
        prediction: Tensor = self.proj(encoder_output_at_masked)
        if prediction.shape[0] == 0:
            return prediction.sum()
        return F.mse_loss(prediction, target_at_masked)

    if TYPE_CHECKING:
        __call__ = forward
