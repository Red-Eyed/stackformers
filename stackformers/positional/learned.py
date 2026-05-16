from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from stackformers.positional.config import LearnedPosEncodingConfig
from stackformers.sequence import PackedInput, PaddedInput, SequenceInput


class LearnedPosEncoding(nn.Module):
    """Learned absolute position embeddings added to query and key tensors.

    Each position index is mapped to a learned dh-dimensional vector, which is
    added to q and k after the head projection.  Requires integer position indices
    in abs_positions (coordinate 0); out-of-range indices raise an embedding error.

    Section 3.5 of Vaswani et al., 2017 describes the learned variant alongside
    sinusoidal encodings — https://arxiv.org/abs/1706.03762
    """

    def __init__(self, config: LearnedPosEncodingConfig) -> None:
        super().__init__()
        self.emb = nn.Embedding(config.max_seq_len, config.dim_head)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        q_input: SequenceInput,
        k_input: SequenceInput,
    ) -> tuple[Tensor, Tensor]:
        match q_input:
            case PaddedInput(abs_positions=q_pos):
                k_pos = k_input.abs_positions
                q_idx = q_pos[..., 0].long()  # b n
                k_idx = k_pos[..., 0].long()  # b s
                q = q + self.emb(q_idx).unsqueeze(1)  # b 1 n dh broadcast over h
                k = k + self.emb(k_idx).unsqueeze(1)
                return q, k
            case PackedInput(abs_positions=q_pos):
                k_pos = k_input.abs_positions
                q_idx = q_pos[..., 0].long()  # nt
                k_idx = k_pos[..., 0].long()  # nt
                q = q + self.emb(q_idx).unsqueeze(1)  # nt 1 dh broadcast over h
                k = k + self.emb(k_idx).unsqueeze(1)
                return q, k
