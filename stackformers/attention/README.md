# attention

Q/K/V projections, kernel dispatch, and additive bias.

`SelfAttention` and `CrossAttention` own the linear projections and GQA head-repeat logic. They delegate the actual dot-product computation to an injected `AttnKernel` and additive bias computation to an injected `AttnBiasBuilder`. These are low-level protocols called via `.forward()`, not `()`.

`NoBiasBuilder` is the null object — it returns `None` so the kernel receives no bias without any branching in the attention module. `ALiBiBuilder` computes head-specific distance penalties.

GQA and MQA are enabled by setting `kv_heads < heads` in `AttentionConfig`. Key and value heads are repeated via `expand` before kernel dispatch — no kernel changes needed.
