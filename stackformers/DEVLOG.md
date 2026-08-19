# Transformer model development log

## 2026-08-19 — Structure-of-arrays variable-width config

### Observation

The variable-width encoder's expanded configuration stored one object per block, even though
the preset varies only model width and head width between blocks. Reading either architecture
schedule therefore required traversing `layers` and extracting nested component fields.

### Decision

Store parallel `d_models` and `dim_heads` arrays directly on
`VariableWidthTransformerEncoderConfig`, while keeping `causal`, `ff_mult`, `dropout`, and
`norm_placement` as shared scalar settings. Validate the two arrays together, remove the
intermediate `VariableWidthEncoderLayerConfig`, and derive the preset's concrete attention,
feed-forward, norm, positional-encoding, and attention-bias configs only while constructing the
encoder. The model topology and tensor operations remain unchanged.

### Verified

- Direct and convenience-factory configs preserve both width arrays and shared settings through
  serialization.
- Config construction rejects empty, unequal, non-positive, and indivisible dimension arrays,
  along with invalid shared feed-forward multipliers and dropout probabilities.
- Encoder construction derives the expected attention dimensions and head counts at each block.
- Existing padded, packed, gradient, normalization-topology, and export behavior remains covered
  by the variable-width encoder tests.

### Unproven

This representation-only change makes no claim about model accuracy, convergence, parameter
count, memory use, or runtime performance.

## 2026-08-19 — HardSwish-gated feed-forward variant

### Observation

SwiGLU couples the gated feed-forward structure to an exact SiLU activation. Mobile-oriented
models may prefer HardSwish's piecewise-linear approximation, but changing `SwiGLU` itself would
silently alter existing configs, checkpoints, and callers that depend on the named operation.

### Decision

Add `HardSwishGLU` as an explicit feed-forward implementation with its own discriminated
`HardSwishGLUConfig`. Retain the same bias-free three-projection structure and parameter-matched
two-thirds hidden-width rule as SwiGLU. Wire it through the common factory without changing any
preset default.

### Verified

- The module preserves token shapes across supported test dtypes, remains bias-free, and
  propagates finite gradients.
- The configuration round-trips through the `FeedForwardConfig` discriminator and the shared
  factory constructs the selected implementation.

### Unproven

No accuracy, convergence, exported-graph size, or ARM latency benefit is claimed. Those properties
remain workload- and runtime-dependent and require comparison against exact SwiGLU.

## 2026-08-18 — Variable-width encoder preset

### Observation

The uniform encoder preset repeats one model dimension and attention-head geometry for every
block. Experiments that allocate residual-stream capacity non-uniformly therefore require manual
stack construction, including dimension-safe projections at every width boundary.

### Decision

Add a separate `VariableWidthTransformerEncoder` preset rather than changing the existing encoder
contract. Its convenience factory accepts parallel `d_models` and `dim_heads` schedules, derives
each block's head count, and expands the schedule into explicit per-layer attention, feed-forward,
norm, positional-encoding, and attention-bias configs. A bias-free learned projection belongs to
the incoming block whenever its width differs from the preceding block; equal-width blocks use an
identity. The preset changes channel capacity only: it does not pool tokens, expose multiscale
features, add skip connections, or claim a spatial hierarchy.

### Verified

- Configuration rejects empty, unequal, non-positive, and indivisible dimension schedules before
  attention construction, and explicit layer configs reject mismatched residual dimensions.
- Padded and packed sequence layouts preserve their token dimensions while producing the final
  configured feature width; gradients cross learned width projections.
- Every normalization topology remains selectable, configs retain their discriminated component
  types through serialization, and padded inference remains compatible with `torch.export`.

### Unproven

No training ablation has established a beneficial width schedule or shown that learned linear
transitions outperform parameter-free residual resizing. The preset's accuracy, convergence,
memory, and latency effects remain workload-dependent.

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
