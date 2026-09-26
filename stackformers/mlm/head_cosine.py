from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import override

if TYPE_CHECKING:
    from jaxtyping import Float


class CosineHead(nn.Module):
    """Reconstruct masked tokens via a linear projection, scored with cosine similarity.

    Scale-invariant where RegressionHead's MSE is not: it scores direction only, so a
    drifting target magnitude (input.x, once the tokenizer trains purely under the main
    task — see MLMWrapper's docstring) doesn't change what "good reconstruction" means.
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
        """Return mean cosine distance, or a differentiable zero for an empty selection."""
        prediction: Tensor = self.proj(encoder_output_at_masked)
        if prediction.shape[0] == 0:
            return prediction.sum()
        return (1 - F.cosine_similarity(prediction, target_at_masked, dim=-1)).mean()

    if TYPE_CHECKING:
        __call__ = forward
