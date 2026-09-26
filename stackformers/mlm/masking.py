from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import override

from stackformers.sequence import PackedInput, PaddedInput, SequenceInput

if TYPE_CHECKING:
    from jaxtyping import Bool


class RandomMasking(nn.Module):
    """Mask each valid token independently with probability mask_ratio.

    Every token's masking decision is drawn independently, so packed sequences need
    no document-boundary awareness here — that only becomes necessary for a
    contiguous-span (blockwise) strategy, which could otherwise straddle two packed
    documents.
    """

    def __init__(self, mask_ratio: float = 0.15) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio

    @override
    def forward(self, input: SequenceInput) -> Bool[Tensor, "*batch"]:
        match input:
            case PaddedInput(mask=mask):
                candidate = torch.rand(mask.shape, device=mask.device) < self.mask_ratio
                return candidate & mask
            case PackedInput(x=x):
                return torch.rand(x.shape[0], device=x.device) < self.mask_ratio

    if TYPE_CHECKING:
        __call__ = forward
