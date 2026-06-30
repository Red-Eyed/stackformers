# attention

Multi-head self-attention and cross-attention.

## Design

**Two configs, two modules.** `SelfAttentionConfig` and `CrossAttentionConfig` are separate types because their parameter sets differ: self-attention has `causal` and `window_size`; cross-attention has neither.

**Static vs runtime dispatch.** The attention *type* (global vs sliding-window) is determined at construction time via `window_size: int | None` in `SelfAttentionConfig`. The *backend* (padded vs packed) is determined at runtime: `SelfAttention.forward` and `CrossAttention.forward` dispatch on `SequenceInput` via `match`. This means one model handles both training (packed) and inference (padded) without swapping classes.

**Kernel ops in `ops.py`.** All low-level attention math lives in `attention/ops.py`: mask builders, `padded_sdpa`, the packed↔padded scatter/gather helpers, and `packed_attn_or_fallback`. The modules stay thin dispatchers.

**Experimental kernel behind one interface function.** `torch.nn.attention.varlen` is an experimental API — the symbol is absent in some torch builds and its signature is not stable. Everything that touches it lives in `varlen_backend.py` behind a single function, `try_varlen_attn(...) -> Tensor | None`: the guarded import, the eligibility/bias checks, and the call itself. The rest of the package depends only on that function and never on the volatile import path, so an absent or renamed symbol degrades to the fallback instead of breaking the whole package at import time.

**Packed attention with automatic fallback.** `PackedInput` normally routes to `varlen_attn`, which requires CUDA and `float16`/`bfloat16`. `packed_attn_or_fallback` calls `try_varlen_attn`; a `None` result means take the padded path — scatter q/k/v to padded layout, run `F.scaled_dot_product_attention`, and gather the result back to packed (same weights, same output shape `(nt, h, d)`). The fallback runs **silently** for the expected CPU/float32/`torch.export` route, and **with a `UserWarning` explaining why** when the kernel was eligible but unusable: it failed to import, an attention bias is present (no bias slot), or the call raised a runtime/signature error (unsupported GPU or a changed experimental API). Dropout works on the fallback path; it is silently skipped only when `varlen_attn` is used (which does not accept a dropout argument).

**`torch.export` and `torch.compile` compatibility.** All scatter/gather helpers in `ops.py` (`_cu_to_indices`, `_packed_heads_to_padded`, `_padded_heads_to_packed`) are fully export-compatible. The key design constraint is that per-document length iteration must never use Python-side loops over data-dependent values (i.e., no `.tolist()` followed by a list comprehension). Instead, `_cu_to_indices` uses `torch.repeat_interleave` to produce `batch_idx` and derives `pos_idx` via the identity `arange(nt) - cu[batch_idx]`. This keeps the computation graph purely symbolic even when `nt` (total tokens) is a data-dependent unbacked symbol.

**GQA / MQA.** Set `kv_heads < heads` in either config. Key and value heads are repeated via `einops.repeat` before the dot-product — no attention-specific changes needed.

## Extending

To add a new attention variant, create a new `nn.Module` that satisfies the `SelfAttn` or `CrossAttn` protocol. No kernel registration or factory changes required.
