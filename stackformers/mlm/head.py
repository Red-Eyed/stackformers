from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class RegressionHead(nn.Module):
    """Reconstruct masked tokens via a linear projection, scored with MSE.

    The projection gives the encoder a dedicated place to specialise for
    reconstruction, so the shared representation isn't forced to double as the
    literal token vector at every layer.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        encoder_output_at_masked: Float[Tensor, "m d"],
        target_at_masked: Float[Tensor, "m d"],
    ) -> Float[Tensor, ""]:
        prediction = self.proj(encoder_output_at_masked)
        return F.mse_loss(prediction, target_at_masked)
