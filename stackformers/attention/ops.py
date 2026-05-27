from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.varlen import varlen_attn as _varlen_attn

from stackformers.sequence import PackedSequence


def varlen_supported(q: Tensor) -> bool:
    return q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)


def padding_mask(mask: Tensor, dtype: torch.dtype) -> Tensor:
    bias = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    bias.masked_fill_(~mask, torch.finfo(dtype).min)
    return bias.view(mask.shape[0], 1, 1, mask.shape[1])


def window_mask(n: int, s: int, window_size: int, causal: bool, device: torch.device) -> Tensor:
    """Additive sliding-window mask (0 = attend, -inf = ignore): (1, 1, n, s)."""
    q_pos = torch.arange(n, device=device).unsqueeze(1)
    k_pos = torch.arange(s, device=device).unsqueeze(0)
    if causal:
        allowed = (k_pos <= q_pos) & (k_pos >= q_pos - window_size)
    else:
        half = window_size // 2
        allowed = (k_pos >= q_pos - half) & (k_pos <= q_pos + half)
    mask = torch.zeros(1, 1, n, s, dtype=torch.float, device=device)
    mask.masked_fill_(~allowed.unsqueeze(0).unsqueeze(0), float("-inf"))
    return mask


def varlen_window(causal: bool, window_size: int | None) -> tuple[int, int]:
    if window_size is None:
        return (-1, 0) if causal else (-1, -1)
    return (window_size, 0) if causal else (window_size // 2, window_size // 2)


def padded_sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    causal: bool,
    window_size: int | None,
    bias: Tensor | None,
) -> Tensor:
    """SDPA for padded inputs, applying padding mask + optional window mask + attention bias."""
    n, s = q.shape[-2], k.shape[-2]
    attn_mask = padding_mask(mask, q.dtype)
    if bias is not None:
        attn_mask = attn_mask + bias
    if window_size is None:
        if causal:
            attn_mask = attn_mask + window_mask(n, s, s, causal=True, device=q.device)
        # is_causal=False: causal constraint already encoded in attn_mask above
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
    else:
        win_mask = window_mask(n, s, window_size, causal, q.device)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask + win_mask)


def _cu_to_indices(cu: Tensor, b: int) -> tuple[Tensor, Tensor]:
    """Return ``(batch_idx, pos_idx)``, each shape ``(nt,)``, from ``cu_seqlens``.

    Both tensors are built without Python-side iteration over data-dependent lengths, so
    this function is compatible with ``torch.export`` and ``torch.compile``.

    The identity used for ``pos_idx`` is::

        pos_idx[i] = i_global - cu[batch_idx[i]]

    i.e. the within-document offset of the i-th flat token.
    """
    lengths = cu[1:] - cu[:-1]  # (b,) — never converted to a Python list
    batch_idx = torch.repeat_interleave(
        torch.arange(b, device=cu.device, dtype=torch.long), lengths
    )  # (nt,)
    nt = batch_idx.shape[0]
    pos_idx = torch.arange(nt, device=cu.device, dtype=torch.long) - cu[batch_idx]
    return batch_idx, pos_idx


def _packed_heads_to_padded(x: Tensor, cu: Tensor, b: int, n: int) -> tuple[Tensor, Tensor]:
    """Scatter (nt, h, d) → (b, h, n, d). Returns padded tensor and (b, n) bool mask."""
    batch_idx, pos_idx = _cu_to_indices(cu, b)
    h, d = x.shape[1], x.shape[2]
    out = x.new_zeros(b, h, n, d)
    out[batch_idx, :, pos_idx] = x
    mask = torch.zeros(b, n, dtype=torch.bool, device=cu.device)
    mask[batch_idx, pos_idx] = True
    return out, mask


def _padded_heads_to_packed(x: Tensor, mask: Tensor) -> Tensor:
    """Gather (b, h, n, d) → (nt, h, d) using bool mask (b, n)."""
    return x.permute(0, 2, 1, 3)[mask]


def packed_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_seq: PackedSequence,
    k_seq: PackedSequence,
    causal: bool,
    window_size: int | None,
) -> Tensor:
    win = varlen_window(causal, window_size)
    result = _varlen_attn(
        query=q,
        key=k,
        value=v,
        cu_seq_q=q_seq.cu_seqlens.to(torch.int32),
        cu_seq_k=k_seq.cu_seqlens.to(torch.int32),
        max_q=q_seq.max_seqlen,
        max_k=k_seq.max_seqlen,
        window_size=win,
    )
    assert isinstance(result, Tensor)
    return result


def packed_attn_or_fallback(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_seq: PackedSequence,
    k_seq: PackedSequence,
    causal: bool,
    window_size: int | None,
    bias: Tensor | None,
) -> Tensor:
    """Run varlen attention; fall back to padded SDPA when unavailable.

    Inputs and output are all packed: q/k/v are (nt, h, d), result is (nt_q, h, d).
    bias=None → no attention bias; varlen_attn is tried first.
    bias=Tensor → attention bias present; varlen_attn is skipped (it has no bias slot) and a
    warning is emitted. Fallback also activates on CPU, non-float16/bfloat16 dtypes, or GPUs
    that do not support varlen_attn (e.g. compute capability < 8.0).
    """
    if bias is not None and varlen_supported(q):
        warnings.warn(
            "varlen_attn does not support attention bias; falling back to padded SDPA. "
            "Performance may be lower.",
            stacklevel=2,
        )
    elif varlen_supported(q):
        try:
            return packed_attn(q, k, v, q_seq, k_seq, causal, window_size)
        except RuntimeError as exc:
            warnings.warn(
                f"varlen_attn is not supported on this device ({exc}); "
                "falling back to padded SDPA. Performance may be lower.",
                stacklevel=2,
            )

    b = int(q_seq.cu_seqlens.shape[0]) - 1
    q_pad, q_mask = _packed_heads_to_padded(q, q_seq.cu_seqlens, b, q_seq.max_seqlen)
    k_pad, k_mask = _packed_heads_to_padded(k, k_seq.cu_seqlens, b, k_seq.max_seqlen)
    v_pad, _ = _packed_heads_to_padded(v, k_seq.cu_seqlens, b, k_seq.max_seqlen)
    out_pad = padded_sdpa(q_pad, k_pad, v_pad, k_mask, causal, window_size, bias)
    return _padded_heads_to_packed(out_pad, q_mask)
