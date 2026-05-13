from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from stackformers.positional.config import NoPosEncodingConfig
from stackformers.sequence import SequenceInfo


class NoPosEncoding(nn.Module):
    """Null object for PosEncoding — passes q, k unchanged regardless of layout."""

    def __init__(self, _config: NoPosEncodingConfig = NoPosEncodingConfig()) -> None:
        super().__init__()

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        _q_seq_info: SequenceInfo | None = None,
        _k_seq_info: SequenceInfo | None = None,
    ) -> tuple[Tensor, Tensor]:
        return q, k
