from __future__ import annotations

from stackformers.attention.kernels.config import (
    KernelConfig,
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


def build_kernel(config: KernelConfig, causal: bool, dropout: float) -> AttnKernel:
    match config:
        case SDPAKernelConfig():
            return SDPAKernel(causal=causal, dropout=dropout)
        case WindowedSDPAKernelConfig():
            return WindowedSDPAKernel(
                window_size=config.window_size, causal=causal, dropout=dropout
            )
        case VarlenSDPAKernelConfig():
            return VarlenSDPAKernel(causal=causal, dropout=dropout)
        case VarlenWindowedSDPAKernelConfig():
            return VarlenWindowedSDPAKernel(
                window_size=config.window_size, causal=causal, dropout=dropout
            )
        case _:
            raise AssertionError(f"Unhandled kernel config: {type(config)}")
