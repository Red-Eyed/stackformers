from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from stackformers.mlm.config import MLMWrapperConfig
from stackformers.mlm.head import RegressionHead
from stackformers.mlm.masking import RandomMasking
from stackformers.mlm.protocols import EncoderLike, MaskingStrategy, ReconstructionHead
from stackformers.sequence import SequenceInput


class MLMOutput(NamedTuple):
    out: Float[Tensor, "*batch d"]
    mlm_loss: Float[Tensor, ""]


class MLMWrapper(nn.Module):
    """Masked-token-reconstruction auxiliary loss over an encoder supplied at call time.

    Does not store or own an encoder — none is passed to __init__, and forward() takes
    one as an argument instead of holding a reference across calls. This module's own
    parameter tree (mask_token, masking_strategy, head) is exactly what it owns; the
    encoder stays wherever the caller already keeps it, with no duplicate registration
    under this wrapper.

    `out` is always the encoder's clean, unmasked output — masking is invisible to
    whatever consumes it, so the main pipeline behaves identically whether or not this
    aux loss is being trained alongside it. Only `mlm_loss` ever reflects that masking
    happened. In training mode that costs a second encoder forward pass on a
    separately-masked copy of input; the two passes share weights but never interfere,
    since the corrupted copy built for the loss is never returned as `out`. In eval mode
    only the one clean pass runs and `mlm_loss` reports a constant zero, so callers can
    invoke this unconditionally in both modes without an if-training branch of their
    own — the same role self.training already plays in nn.Dropout or nn.BatchNorm.

    The masked pass detaches input.x once, up front — both the unmasked context fed to
    the encoder and the reconstruction target come from that same detached copy — so
    mlm_loss trains the encoder, mask_token, and head, never whatever produced input.x.
    Detaching only the target is not sufficient: self-attention still mixes the
    unmasked (undetached) positions into the masked positions' predictions, so a live
    path back to input.x would otherwise survive at every unmasked position. Severing
    it entirely removes the representation-collapse shortcut a trainable tokenizer
    would otherwise have available (drive every token toward a constant vector to make
    reconstruction trivial).
    """

    def __init__(
        self,
        config: MLMWrapperConfig,
        masking_strategy: MaskingStrategy | None = None,
        head: ReconstructionHead | None = None,
    ) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.empty(config.dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02, a=-0.04, b=0.04)
        self.masking_strategy = (
            masking_strategy if masking_strategy is not None else RandomMasking(config.mask_ratio)
        )
        self.head = head if head is not None else RegressionHead(config.dim)

    def _masked_loss(self, input: SequenceInput, encoder: EncoderLike) -> Tensor:
        # Detached once, up front, and reused for both corrupted_x and target below —
        # unmasked positions must not carry a live path back to whatever produced
        # input.x either, or self-attention mixes them into the masked positions'
        # predictions and mlm_loss reaches upstream anyway despite target being detached.
        input = input._replace(x=input.x.detach())
        target = input.x
        should_mask = self.masking_strategy(input)
        # should_mask's shape always matches input.x's leading dims, so this select and
        # the boolean indexing below both work for PaddedInput and PackedInput alike.
        corrupted_x = torch.where(should_mask.unsqueeze(-1), self.mask_token, input.x)
        masked_output = encoder(input._replace(x=corrupted_x))
        return self.head(masked_output[should_mask], target[should_mask])

    def forward(self, input: SequenceInput, encoder: EncoderLike) -> MLMOutput:
        clean_output = encoder(input)
        if self.training:
            mlm_loss = self._masked_loss(input, encoder)
            return MLMOutput(out=clean_output, mlm_loss=mlm_loss)
        return MLMOutput(out=clean_output, mlm_loss=clean_output.new_zeros(()))
