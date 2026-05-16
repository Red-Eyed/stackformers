from __future__ import annotations

from stackformers.attention.config import AttentionConfig
from stackformers.attention.kernels.config import (
    SDPAKernelConfig,
    VarlenSDPAKernelConfig,
    VarlenWindowedSDPAKernelConfig,
    WindowedSDPAKernelConfig,
)
from stackformers.attention.kernels.sdpa import SDPAKernel
from stackformers.attention.kernels.varlen import VarlenSDPAKernel
from stackformers.attention.kernels.varlen_windowed import VarlenWindowedSDPAKernel
from stackformers.attention.kernels.windowed import WindowedSDPAKernel
from stackformers.attention.protocols import AttnKernel


def build_kernel(attn: "AttentionConfig") -> AttnKernel:
    match attn.kernel:
        case SDPAKernelConfig():
            return SDPAKernel(causal=attn.causal, dropout=attn.dropout)
        case WindowedSDPAKernelConfig():
            return WindowedSDPAKernel(
                window_size=attn.kernel.window_size,
                causal=attn.causal,
                dropout=attn.dropout,
                mode=attn.kernel.mode,
            )
        case VarlenSDPAKernelConfig():
            return VarlenSDPAKernel(causal=attn.causal, dropout=attn.dropout)
        case VarlenWindowedSDPAKernelConfig():
            return VarlenWindowedSDPAKernel(
                window_size=attn.kernel.window_size, causal=attn.causal, dropout=attn.dropout
            )
        case _:
            raise AssertionError(f"Unhandled kernel config: {type(attn.kernel)}")
