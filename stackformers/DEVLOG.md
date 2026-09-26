# Transformer model development log

## 2026-09-26 — Empty-input contracts and strict Python checks (4.7.0b8)

### Observation

The rustic-python review reproduced NaN MLM losses when independent masking selected no
tokens, and attention averaging masked values when a document had no valid context. Packed
self-attention also repadded input features for a no-op bias. Cross-attention accepted mixed
layouts in its annotations but failed during reshaping, and mutable attention configs could
silently reinterpret existing projections.

### Decision

Keep independent MLM sampling and define empty reconstruction selections as differentiable
zero losses. Use negative infinity for excluded keys, temporarily unmask fully excluded rows
for backend-safe softmax, and zero those outputs and gradients. Keep padding-only masks compact
until the export-required query expansion. Skip feature repadding for the built-in no-op bias.
Freeze attention configs and express matching cross-attention layouts with overloads and
explicit runtime pair validation.

Adopt the rustic-python strict checker profile for source, tests, and examples at the package's
Python 3.11 minimum. Expose module-call signatures only under `TYPE_CHECKING`, preserving runtime
hooks. Annotate known PyTorch tensor-return boundaries and registered buffers, keep experimental
kernel returns opaque until validation, and remove unqualified or obsolete suppressions.
The remaining two Pyrefly suppressions deliberately mutate immutable values in runtime tests.

### Verified

Focused empty-selection, masked-output/gradient, and static/dynamic ONNX tests passed on CPU,
including required opsets 18–25. Ruff and strict Pyrefly pass for the library, tests, and examples.
The final `just check` passes on Python 3.11: 1118 passed, 21 skipped, 162 expected failures,
and 87 optional-case unexpected passes. The locked environment uses Pyrefly 1.3.1 and Ruff 0.16.9.
A negative type-check probe rejects both mixed-layout orders, a mixed-layout protocol call,
an invalid module argument, an explicit Any leak, and missing function annotations; matching
layout calls retain Tensor return types.

### Unproven

CUDA kernels and GPU/mobile ONNX execution providers were not exercised. The avoided packed
feature allocation is checked at its conversion boundary; end-to-end memory and throughput
effects have not been benchmarked. Static Tensor annotations do not establish numerical shapes.

## 2026-09-25 — ONNX Runtime attention-mask compatibility

### Observation

Static and dynamic opset-23 encoder graphs exported successfully but failed in ONNX Runtime
with `inconsistent q_sequence_length (between attn_mask and Q)`. The padding-only mask had
shape `(batch, 1, 1, keys)`: valid ONNX broadcasting, but rejected by the runtime's Attention
kernel. The test matrix marked opset 23 as an expected failure and excluded newer opsets using
PyTorch's legacy exporter limit.

### Decision

Expand the combined padding/bias mask to the query length in the shared padded SDPA path.
This retains PyTorch view semantics and symbolic lengths without changing attention APIs,
parameters, or masking values. Add a local comment explaining the runtime constraint, require
opsets 23–25, and compare multi-head padded exports against the original broadcast-mask SDPA
with independently resized batch, query, and context axes, including a single query.

### Verified

Static and dynamic plain encoder exports execute with eager parity at opsets 23, 24, and 25
using PyTorch 2.11.0, ONNX 1.22.0, and ONNX Runtime 1.29.0 on CPU.
`just check` passes formatting, lint, type checking, and the full test suite: 1059 passed,
21 skipped, 138 expected failures, and 99 optional-case unexpected passes. Required coverage
includes opsets 18–25 across the existing encoder, decoder, cross-attention, and cache tests,
plus the new padding-mask regression cases.

### Unproven

GPU/mobile execution-provider support and memory/performance effects have not been measured.
The installed exporter does not preserve the requested opset 26 for the plain encoder; this
change does not claim support beyond opset 25.

## 2026-08-21 — Exportable decoder cross/self-attention K/V caches

### Observation

Autoregressive ONNX decoding projected the same fixed encoder context into cross-attention keys
and values at every generation step and recomputed self-attention K/V for the complete target
prefix. On a mobile C++ runtime this repeated work adds latency even though the encoder context and
all earlier target projections are unchanged.

### Decision

Add cache behavior only through composed classes, leaving existing attention, decoder-layer,
decoder, preset, and positional classes unchanged. `CachedCrossAttentionWrapper` projects and
positions context K once; `CachedSelfAttentionWrapper` accepts one target token plus a required growing cache and
returns the cache extended by one position. They share the wrapped modules' projection,
normalization, positional, dropout, and output parameters without copying them. Query-only and
key-only cross positioning use the existing joint positional API with a zero-token counterpart,
so every built-in encoding works without a new positional interface.

`DecoderCrossAttentionCacheBuilder` exposes the once-per-context graph and `CachedDecoderWrapper`
constructs a separate executor for each existing normalization topology. The decoder graph accepts
the immutable dense cross cache, required dense self cache, and an int64 `step_i`; it returns the
output token and next self cache. The context validity mask remains separate because padding cannot
be derived from tensor dimensions. Cache lengths come from tensor dimensions, so there is no
source-length input. Batch, source-token, and past-token axes are dynamic, while
layer/head/feature geometry and the one-token decoder axis remain fixed by the model contract.

### Verified

- Cached and ordinary outputs match for MHA/GQA, QK-Norm, context/target padding, RoPE positional
  encoding, and every decoder normalization topology.
- Repeated calls project only the new self/cross queries and new self K/V; context K/V projection
  runs only during cache construction, and both caches remain at `kv_heads` width.
- Separate encoder, cross-cache builder, and one-token decoder graphs execute in ONNX Runtime with
  growing-prefix parity. Exported runtime-varying batch/source/past axes accept shapes different
  from the export examples, including a zero-length initial self cache.
- Composing cache executors adds no parameters or buffers, leaves ordinary checkpoint keys
  unchanged, and requires no cache-specific methods on the existing model classes.

### Unproven

No mobile-device latency, peak-memory, binary-size, or power measurement has been made. ONNX
Runtime CPU parity does not establish support or performance for a particular mobile execution
provider. Cached self-attention currently supports global causal attention with no attention bias;
sliding-window attention and custom bias collaborators remain unsupported on the cached path.

## 2026-08-19 — Restore complete variable-width layer configs

### Observation

The structure-of-arrays revision stored only model/head widths and reconstructed attention,
SwiGLU, RMSNorm, RoPE, and attention bias inside the model. That shortened the serialized config
by removing the component choices, but violated Stackformers' central contract: model components
must remain explicit, interchangeable configuration values. Adding more parallel arrays would
restore configurability at the cost of length coupling and index-based validation across every
component schedule.

### Decision

Restore `VariableWidthEncoderLayerConfig` as one complete record per block. Each record owns its
attention, feed-forward, norm, positional-encoding, and attention-bias configs and validates their
shared dimensions locally. Keep `variable_width_encoder_config(d_models=..., dim_heads=...)` as the
concise opinionated factory for the default RoPE/RMSNorm/SwiGLU preset; direct construction remains
the public path for replacing any component. Export the layer config and all concrete component
config types from the package root.

### Verified

- Every feed-forward config variant builds inside the variable-width encoder and passes static and
  dynamic PyTorch/ONNX export with ONNX Runtime parity.
- Direct configuration selects non-default attention options, GEGLU, LayerNorm, RoPE-2D, and
  relative-distance bias without model subclassing or internal mutation.
- Layer-local validation rejects mismatched feed-forward, norm, positional, and bias geometry.
- Public API contract tests cover the concrete feed-forward and positional config/model types,
  `VariableWidthEncoderLayerConfig`, and `node_encoder_config`.

### Unproven

The configuration correction makes no claim about the training quality or performance of any
component combination. It verifies construction, execution, serialization, and export behavior.

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
