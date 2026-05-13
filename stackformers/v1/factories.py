from __future__ import annotations

from stackformers.v1.attention.bias import NoBiasBuilder
from stackformers.v1.attention.config import AttentionConfig
from stackformers.v1.attention.cross_attn import CrossAttention
from stackformers.v1.attention.kernels import SDPAKernel
from stackformers.v1.attention.self_attn import SelfAttention
from stackformers.v1.config import DecoderConfig, EncoderConfig
from stackformers.v1.decoder import Decoder, DecoderLayer
from stackformers.v1.encoder import Encoder
from stackformers.v1.feedforward.config import FeedForwardConfig
from stackformers.v1.feedforward.swiglu import SwiGLU
from stackformers.v1.layers import TransformerLayer
from stackformers.v1.norm.rms import RMSNorm
from stackformers.v1.positional.none import NoPosEncoding
from stackformers.v1.positional.rope1d import RotaryEmbedding1D


def build_encoder(config: EncoderConfig, use_rope: bool = True) -> Encoder:
    """Build a standard encoder with RoPE (or no pos-encoding) and SDPA kernel."""
    dim = config.layer.attn.dim
    dh = config.layer.attn.dim_head
    pos = RotaryEmbedding1D(dim_head=dh) if use_rope else NoPosEncoding()

    layers = [
        TransformerLayer(
            self_attn=SelfAttention(
                config=config.layer.attn,
                pos_encoding=pos,
                bias_builder=NoBiasBuilder(),
                kernel=SDPAKernel(dropout=config.dropout),
            ),
            ff=SwiGLU(config.layer.ff),
            norm_attn=RMSNorm(dim),
            norm_ff=RMSNorm(dim),
        )
        for _ in range(config.num_layers)
    ]
    return Encoder(layers=layers, final_norm=RMSNorm(dim))


def build_decoder(config: DecoderConfig, use_rope: bool = True) -> Decoder:
    """Build a standard encoder-decoder decoder with causal self-attn and cross-attn."""
    dim = config.self_attn.dim
    dh = config.self_attn.dim_head
    pos = RotaryEmbedding1D(dim_head=dh) if use_rope else NoPosEncoding()

    causal_cfg = config.self_attn.model_copy(update={"causal": True})

    layers = [
        DecoderLayer(
            self_attn=SelfAttention(
                config=causal_cfg,
                pos_encoding=pos,
                bias_builder=NoBiasBuilder(),
                kernel=SDPAKernel(dropout=config.dropout),
            ),
            cross_attn=CrossAttention(
                config=config.cross_attn,
                pos_encoding=NoPosEncoding(),
                bias_builder=NoBiasBuilder(),
                kernel=SDPAKernel(dropout=config.dropout),
            ),
            ff=SwiGLU(config.ff),
            norm_self=RMSNorm(dim),
            norm_cross=RMSNorm(dim),
            norm_ff=RMSNorm(dim),
        )
        for _ in range(config.num_layers)
    ]
    return Decoder(layers=layers, final_norm=RMSNorm(dim))


def build_gpt(
    dim: int = 768,
    heads: int = 12,
    dim_head: int = 64,
    num_layers: int = 12,
    ff_mult: float = 4.0,
) -> Encoder:
    """GPT-style causal language model backbone (encoder-only with causal mask)."""
    attn_cfg = AttentionConfig(dim=dim, heads=heads, dim_head=dim_head, causal=True)
    ff_cfg = FeedForwardConfig(dim=dim, mult=ff_mult)
    pos = RotaryEmbedding1D(dim_head=dim_head)

    layers = [
        TransformerLayer(
            self_attn=SelfAttention(
                config=attn_cfg,
                pos_encoding=pos,
                bias_builder=NoBiasBuilder(),
                kernel=SDPAKernel(),
            ),
            ff=SwiGLU(ff_cfg),
            norm_attn=RMSNorm(dim),
            norm_ff=RMSNorm(dim),
        )
        for _ in range(num_layers)
    ]
    return Encoder(layers=layers, final_norm=RMSNorm(dim))
