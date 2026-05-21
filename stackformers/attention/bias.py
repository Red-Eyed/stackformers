from __future__ import annotations

import torch.nn as nn

from stackformers.sequence import PaddedInput


class NoAttnBias(nn.Module):
    """Null object for AttnBias — signals no bias so varlen_attn remains available."""

    def __call__(self, input: PaddedInput) -> None:
        return None
