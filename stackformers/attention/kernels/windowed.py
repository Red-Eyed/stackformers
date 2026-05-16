from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from stackformers.attention.kernels._mask import build_window_mask
from stackformers.attention.kernels.sdpa import _padding_mask
from stackformers.sequence import PaddedSequence, SequenceInfo


def _windowed_mask(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    k_seq_info: SequenceInfo | None,
    attn_bias: Tensor | None,
    window_size: int,
    causal: bool,
    dropout_p: float,
) -> Tensor:
    # Build an (n, s) additive mask, then run full SDPA. O(n·s) — simple but suboptimal for large s.
    n, s = q.shape[-2], k.shape[-2]
    combined: Tensor = build_window_mask(n, s, window_size, causal, q.device)

    if attn_bias is not None:
        combined = combined + attn_bias
    if isinstance(k_seq_info, PaddedSequence):
        combined = combined + _padding_mask(k_seq_info.mask, q.dtype)

    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=combined, dropout_p=dropout_p, is_causal=False
    )


def _local_key_windows(
    k: Tensor,
    v: Tensor,
    pad_left: int,
    pad_right: int,
    span: int,
) -> tuple[Tensor, Tensor]:
    """Pad and unfold k/v into per-query local windows: (b, h, n, span, dh).

    After left-padding by pad_left, unfold(size=span, step=1) produces exactly n windows
    so that window i covers the key positions that query i is allowed to attend to.
    """
    k_padded = F.pad(k, (0, 0, pad_left, pad_right))
    v_padded = F.pad(v, (0, 0, pad_left, pad_right))
    # unfold inserts the window dim at the end: (b, h, n, dh, span)
    # transpose to (b, h, n, span, dh)
    k_win = k_padded.unfold(2, span, 1).transpose(-1, -2)
    v_win = v_padded.unfold(2, span, 1).transpose(-1, -2)
    return k_win, v_win


def _windowed_validity_mask(
    n: int,
    k_seq_info: SequenceInfo | None,
    pad_left: int,
    pad_right: int,
    span: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Additive local attention mask (0 = attend, finfo.min = ignore): (b or 1, 1, n, span).

    Marks two kinds of invalid positions inside each local window:
      1. Structural padding zeros from _local_key_windows (outside [pad_left, pad_left+n)).
      2. PaddedSequence padding tokens (positions where the key mask is False).
    Both are handled by building a single validity tensor over the padded key axis and unfolding it
    the same way as k/v, so every window entry maps to the same key position it attends to.
    """
    k_len = n + pad_left + pad_right
    # True for positions that correspond to real (non-structural-padding) keys.
    positions = torch.arange(k_len, device=device)
    valid_base = (positions >= pad_left) & (positions < pad_left + n)

    if isinstance(k_seq_info, PaddedSequence):
        # Pad the per-sample key mask to match the padded key length; padding slots are False.
        k_mask_padded = F.pad(
            k_seq_info.mask.unsqueeze(1).to(torch.bool),
            (pad_left, pad_right),
            value=False,
        )  # (b, 1, k_len)
        valid = valid_base & k_mask_padded  # (b, 1, k_len)
    else:
        valid = valid_base.view(1, 1, k_len)  # (1, 1, k_len)

    valid_win = valid.unfold(2, span, 1)  # (b or 1, 1, n, span)
    mask = torch.zeros(valid_win.shape, dtype=dtype, device=device)
    mask.masked_fill_(~valid_win, torch.finfo(dtype).min)
    return mask


def _windowed_bias_local(
    attn_bias: Tensor,
    pad_left: int,
    pad_right: int,
    span: int,
) -> Tensor:
    """Extract per-query local window from attn_bias (h, n, s) → (h, n, span).

    attn_bias[h, i, j] is the bias for query i → key j.  After padding the key axis, row i of the
    bias should be windowed starting at padded position i (same alignment as k/v).  That's the
    diagonal of the (n_query, n_key_window_start) unfolded tensor, extracted with .diagonal().
    """
    bias_padded = F.pad(attn_bias, (pad_left, pad_right), value=0.0)
    bias_win = bias_padded.unfold(2, span, 1)  # (h, n_q, n_kw, span); n_q == n_kw == n
    # diagonal(dim1=1, dim2=2): entry [h, i, i, :] for each i → (h, span, n)
    # permute to (h, n, span)
    return bias_win.diagonal(dim1=1, dim2=2).permute(0, 2, 1)


def _windowed_unfold(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    k_seq_info: SequenceInfo | None,
    attn_bias: Tensor | None,
    window_size: int,
    causal: bool,
    dropout_p: float,
) -> Tensor:
    """O(n·w) windowed attention via unfold reshape.

    Instead of an n×s attention matrix with most entries masked, we reshape into
    (b·h·n, 1, span) × (b·h·n, span, dh) so SDPA only ever sees the local window.
    """
    b, h, n, dh = q.shape
    w = window_size

    # Causal: query i → keys [i-w, i];  span = w+1, pad only on the left.
    # Bidirectional: query i → keys [i-half, i+half];  span = 2·half+1, symmetric pad.
    if causal:
        pad_left, pad_right = w, 0
        span = w + 1
    else:
        half = w // 2
        pad_left, pad_right = half, half
        span = 2 * half + 1

    k_win, v_win = _local_key_windows(k, v, pad_left, pad_right, span)
    local_mask = _windowed_validity_mask(
        n, k_seq_info, pad_left, pad_right, span, q.dtype, q.device
    )

    if attn_bias is not None:
        local_mask = local_mask + _windowed_bias_local(attn_bias, pad_left, pad_right, span)

    # Flatten (b, h, n) into a single batch dimension so SDPA processes one query token at a time.
    q_flat = q.reshape(b * h * n, 1, dh)
    k_flat = k_win.reshape(b * h * n, span, dh)
    v_flat = v_win.reshape(b * h * n, span, dh)
    mask_flat = local_mask.expand(b, h, n, span).reshape(b * h * n, span).unsqueeze(1)

    out = F.scaled_dot_product_attention(
        q_flat, k_flat, v_flat, attn_mask=mask_flat, dropout_p=dropout_p, is_causal=False
    )
    return out.reshape(b, h, n, dh)


class WindowedSDPAKernel(nn.Module):
    """Padded-sequence sliding-window attention kernel.

    mode="mask"  — additive O(n·s) mask over full SDPA (default, torch.export-safe).
    mode="unfold" — O(n·w) unfold reshape; avoids materialising the full attention matrix.
    """

    def __init__(
        self,
        window_size: int,
        causal: bool = False,
        dropout: float = 0.0,
        mode: Literal["mask", "unfold"] = "mask",
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.causal = causal
        self.dropout = dropout
        self.mode = mode

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_seq_info: SequenceInfo,  # noqa: ARG002
        k_seq_info: SequenceInfo | None,
        attn_bias: Tensor | None,
    ) -> Tensor:
        dropout_p = self.dropout if self.training else 0.0
        if self.mode == "unfold":
            return _windowed_unfold(
                q, k, v, k_seq_info, attn_bias, self.window_size, self.causal, dropout_p
            )
        return _windowed_mask(
            q, k, v, k_seq_info, attn_bias, self.window_size, self.causal, dropout_p
        )
