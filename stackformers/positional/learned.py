from __future__ import annotations

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from stackformers.positional.config import LearnedPosEncodingConfig


class LearnedPosEncoding(nn.Module):
    """Learned absolute position embeddings added to query and key tensors.

    Each position index is mapped to a learned dh-dimensional vector, which is
    added to q and k after the head projection.  Requires integer position indices
    in positions[..., 0]; out-of-range indices raise an embedding error.

    Section 3.5 of Vaswani et al., 2017 describes the learned variant alongside
    sinusoidal encodings — https://arxiv.org/abs/1706.03762
    """

    def __init__(self, config: LearnedPosEncodingConfig) -> None:
        super().__init__()
        self.emb = nn.Embedding(config.max_seq_len, config.dim_head)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward_padded(
        self,
        q: Float[Tensor, "b h n dh"],
        k: Float[Tensor, "b h s dh"],
        q_positions: Float[Tensor, "b n c"],
        k_positions: Float[Tensor, "b s c"],
    ) -> tuple[Float[Tensor, "b h n dh"], Float[Tensor, "b h s dh"]]:
        q_idx = q_positions[..., 0].long()  # b n
        k_idx = k_positions[..., 0].long()  # b s
        return q + self.emb(q_idx).unsqueeze(1), k + self.emb(k_idx).unsqueeze(1)

    def forward_packed(
        self,
        q: Float[Tensor, "nt h dh"],
        k: Float[Tensor, "nt h dh"],
        q_positions: Float[Tensor, "nt c"],
        k_positions: Float[Tensor, "nt c"],
    ) -> tuple[Float[Tensor, "nt h dh"], Float[Tensor, "nt h dh"]]:
        q_idx = q_positions[..., 0].long()  # nt
        k_idx = k_positions[..., 0].long()  # nt
        return q + self.emb(q_idx).unsqueeze(1), k + self.emb(k_idx).unsqueeze(1)
