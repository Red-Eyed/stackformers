# attention

Multi-head self-attention and cross-attention with inline SDPA math.

## Design

**Two configs, two modules.** `SelfAttentionConfig` and `CrossAttentionConfig` are separate types because their parameter sets differ: self-attention has `causal` and `window_size`; cross-attention has neither.

**Static vs runtime dispatch.** The attention *type* (global vs sliding-window) is determined at construction time via `window_size: int | None` in `SelfAttentionConfig`. The *backend* (padded vs packed) is determined at runtime: `SelfAttention.forward` and `CrossAttention.forward` dispatch on `SequenceInput` via `match`. This means one model handles both training (packed) and inference (padded) without swapping classes.

**No kernel abstraction.** SDPA math (`F.scaled_dot_product_attention`, `varlen_attn`) is inline in `_forward_padded` and `_forward_packed`. A shared `_packed_attn` module-level function is used by both `SelfAttention` and `CrossAttention`.

**GQA / MQA.** Set `kv_heads < heads` in either config. Key and value heads are repeated via `einops.repeat` before the dot-product — no attention-specific changes needed.

## Extending

To add a new attention variant, create a new `nn.Module` that satisfies the `SelfAttn` or `CrossAttn` protocol. No kernel registration or factory changes required.
