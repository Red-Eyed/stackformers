# attention

Attention modules: projections, kernels, and bias builders.

## Files

| File | Contents |
|------|----------|
| `config.py` | `AttentionConfig` — dim, heads, dim_head, kv_heads, dropout, causal |
| `protocols.py` | `AttnKernel`, `AttnBiasBuilder` (low-level); `SelfAttn`, `CrossAttn` (high-level) |
| `self_attn.py` | `SelfAttention` — Q/K/V projections + kernel dispatch |
| `cross_attn.py` | `CrossAttention` — same as above but keys/values come from a separate context tensor |
| `bias.py` | `NoBiasBuilder` (null object), `ALiBiBuilder` (position bias without positional encodings) |
| `bias_config.py` | `NoBiasConfig`, `ALiBiConfig`; discriminated union `BiasBuilderConfig` |
| `bias_factory.py` | `build_bias_builder(config, heads, causal) -> AttnBiasBuilder` — dispatches on `kind` |
| `kernels/` | One file per `AttnKernel` implementation (see below) |

## Kernels

| Kernel | Sequence format | Notes |
|--------|----------------|-------|
| `SDPAKernel` | padded `(b, h, n, dh)` | `F.scaled_dot_product_attention`; supports causal mask + attention bias |
| `WindowedSDPAKernel` | padded `(b, h, n, dh)` | sliding-window local attention via `F.scaled_dot_product_attention` |
| `VarlenSDPAKernel` | packed `(nt, h, dh)` | fused `varlen_attn` on CUDA; per-sequence loop fallback on CPU |
| `VarlenWindowedSDPAKernel` | packed `(nt, h, dh)` | packed + sliding-window |

## Self-attn vs cross-attn

- `SelfAttention`: x → Q, K, V (same source). Accepts a `SequenceInfo` for masking.
- `CrossAttention`: x → Q, context → K, V. Accepts optional `ctx_seq_info` for context masking.

Both implement their respective high-level protocol (`SelfAttn`, `CrossAttn`) and delegate the actual dot-product computation to an injected `AttnKernel`.

## GQA / MQA

Set `kv_heads < heads` in `AttentionConfig`. Keys and values are repeated via `expand` to match the query head count before kernel dispatch — no kernel changes needed.
