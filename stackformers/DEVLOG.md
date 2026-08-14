# Transformer model development log

## 2026-08-14 — Configurable normalization placement

### Observation

During training, the feed-forward down-projection activation RMS grew over time, reaching roughly
43 with GELU after an earlier SwiGLU run reached approximately 20,000. The existing pre-norm
layout controls each branch input but does not constrain the branch value added to the residual
stream.

### Decision

Expose `NormPlacement = Literal["pre", "post", "sandwich", "reordered"]` on encoder, decoder, and
cross-attender presets and map each value to a focused layer class. Preserve the existing
`TransformerLayer`, `DecoderLayer`, and `CrossAttenderLayer` as the pre-norm implementations and
defaults; add dedicated post, sandwich, and reordered classes whose constructors accept exactly
the norms they use. Sandwich owns independent pre- and post-branch norm modules instead of tying
their affine parameters. Reordered implements the OLMo 2 branch-output normalization layout
without implicitly enabling QK-Norm or changing any attention default. Decoder/cross-attender
placement affects only the target/query residual stream, not the context sequence.

### Verified

- The omitted/default setting and explicit `"pre"` setting have exactly equal outputs, input
  gradients, and parameter gradients in deterministic regression tests.
- Pre, post, and reordered placements retain the legacy state-dict keys and shapes and accept
  strict checkpoint loading in both directions.
- Config payloads without `norm_placement` restore as `"pre"`; all three existing pre-norm classes
  keep their constructors and state structures for legacy checkpoints.
- Tests distinguish the exact equations of all four layouts for two- and three-branch stacks,
  verify target/query and context gradients, reject invalid config values, and export every layout
  successfully through `torch.export`.

### Unproven

No training ablation or mobile runtime benchmark has yet established which placement gives the
best accuracy, activation range, convergence, or ARM64 latency for the target model. Reordered and
sandwich norms constrain what enters the residual stream, but they do not prevent a large raw FFN
down-projection output before its post-branch norm; that tensor still requires measurement during
quantization calibration.
