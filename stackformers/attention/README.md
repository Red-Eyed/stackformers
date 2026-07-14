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

**Geometry a rotary encoding cannot carry.** `RelativeDistanceBias` adds a learned function of the Euclidean distance between node positions to the logits, making attention invariant to any global translation or rotation of the input. It lives here rather than in `positional/` because it *cannot* live there: RoPE rotates by `ω·p`, which is linear in position by construction — and that linearity is exactly what makes the query and key rotations cancel into a relative offset. Distance is not linear in position, so no rotary encoding, axial or otherwise, can express it; a bias is the only route. Pair it with `NoPosEncoding` — a rotary encoding alongside it would reintroduce the preferred frame it exists to remove. Use it when node coordinates have no meaningful axes; when up and right *do* mean something, RoPE-2D keeps direction and stays varlen-compatible.

**The bias costs the varlen path, by construction.** `varlen_attn` has no bias slot, so any non-`None` `AttnBias` forces the padded SDPA path and materialises a `(b, h, n, s)` tensor. The `(b, n, s, num_rbf)` shell intermediate is retained for backward and is `num_rbf/h` times larger still — it, not the bias, dominates activation memory. `NoAttnBias` returns `None` precisely so the common case keeps the kernel. `TransformerEncoder` shares one bias *module* across layers, but each layer still *calls* it, so the tensor is recomputed and retained per layer; at large node counts the levers are a smaller `num_rbf`, checkpointing the bias, or hoisting the call out of the layer loop.

**Why not `flex_attention`.** It fuses the bias into the kernel and would make it O(n) instead of O(n²) — attempted, and parked on the `flex-attention` branch. It needs `dim_head >= 16`; its `score_mod` must lower as a *pointwise* subgraph, which forbids both a `vector_norm` over coordinates and a sum over shells; a captured tensor requiring grad may be indexed exactly once, which forbids unrolling that sum instead; called eagerly it is unfused and catastrophic (measured 64 s/step at n=2048, OOM at n=4096, against ~40 ms for SDPA), so it is viable only inside `torch.compile`; and under `torch.compile` with *dynamic* shapes it bails back to SDPA anyway. With node counts that vary per batch, the fused kernel would rarely be the one that runs.

## Extending

To add a new attention variant, create a new `nn.Module` that satisfies the `SelfAttn` or `CrossAttn` protocol. No kernel registration or factory changes required.

To add a new attention bias, write an `nn.Module` satisfying `AttnBias` (take `PaddedInput`, return a broadcastable `(b, h, n, s)` tensor or `None`), give it a config with a `kind` discriminator, join it to the `AttnBiasConfig` union, and add one `match` arm to `build_attn_bias`.
