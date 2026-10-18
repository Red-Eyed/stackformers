# Transformer model development log

## 2026-10-18 — Construct packed layouts from concrete configs (5.0.0rc12)

### Observation and decision

The explicit conversion API in `67f5b40` preserves old checkpoints, but reconstructing a
packed model should not require constructing and replacing ordinary projections first.
Introduce separate `PackedSelfAttentionConfig`, `PackedCrossAttentionConfig`, and
`PackedSwiGLUConfig` records. Factories exhaustively match discriminated unions; constructors
accept only their matching concrete config. Existing presets delegate to these factories.
Shared config fields remain in private data bases without adding behavioral abstractions.

Persist `kind` in JSON so saved configs reconstruct the matching state-dict layout. Adapt
legacy untagged attention JSON to the ordinary variant at validation. Conversion and
unpacking create the destination config explicitly. Ordinary forward implementations and
checkpoint names remain intact; existing cache wrappers still require ordinary attention.
A converted model's state dict must be saved alongside a matching packed preset config.

### Verified result and limits

Focused checks cover factory dispatch, config transitions, legacy JSON, and strict checkpoint
reload/output parity for all four presets, all normalization topologies, independent attention
and SwiGLU layout selections, and supported device/dtype combinations. `just check` passes:
1,769 passed, 21 skipped, 172 expected failures, and 111 optional-case unexpected passes;
formatting, lint, and types also pass. Existing conversion and numerical parity tests pass.
This change does not alter projection equations or
establish new performance results; the previous complete-block experiment remains the evidence.

## 2026-10-18 — Opt-in projection packing (5.0.0rc11)

### Observation

Self-attention projects the same input three times; cross-attention projects its context
twice; SwiGLU projects gate and value independently. QuantSmith's isolated QKV measurements
suggested packing could reduce integer-projection overhead, but did not establish complete-block gains.

### Decision

Add `to_packed()` to the existing attention and SwiGLU classes. Preserve their original forward
implementations, parameter names, factory dispatch, and cache wrappers. Put experimental
`PackedSelfAttention`, `PackedCrossAttention`, and `PackedSwiGLU` in separate modules, using
ordinary `nn.Linear` and explicit split widths. No new protocol or abstract hierarchy is needed.

Load old checkpoints into the original classes, then convert. Conversion copies parameters,
preserves dtype/device, frozen parameters and training modes, and leaves the source untouched.
`unpack()` returns an independently owned original module with its original checkpoint names.
Projection groups must share trainability. Convert before creating an optimizer; optimizer
state, existing gradients, hooks and parameter aliases are not migrated. Sequence packing and
projection packing are independent: both projection layouts accept padded and packed inputs.

### Verified

`just check` passes: 1,614 tests passed, 21 skipped, 172 expected failures, and 111 optional-case
unexpected passes. New checks cover forward and gradient parity, MHA/GQA/MQA, masks, learned
bias, local visibility, checkpoint serialization, frozen/tied parameters, training dropout,
and fresh optimizer steps. Released attention references from 5.0.0rc1 retain exact checkpoint
keys and output/input-gradient behavior after conversion. A SwiGLU reference captured from
the installed 4.6.0 wheel verifies the old checkpoint transition independently of current code.

All 12 attention dynamic-export tests pass in QuantSmith's PyTorch 2.14 environment. Four
PyTorch 2.11 GQA cases remain expected failures only after confirming that the ordinary
implementation encounters the same symbolic stride-guard error. The other new export cases pass.

### Complete-block ORT CPU experiment

QuantSmith's existing exporter, integer lowering, profiler and timing runner were reused with
an experiment adapter under ignored `work_dir/projection_packing/`. This run uses the local
5.0.0rc11 source checkout (file hashes recorded in `source.json`), PyTorch 2.14.0, ORT 1.30.0,
ONNX 1.23.0, arm64 macOS 27.0.1, batch 1, width 768 and 12 heads. The causal encoder has one
RoPE/RMSNorm/SwiGLU layer and a final norm; cross-attention reprojects a 128-token context.
Neither workload uses a KV cache. Weights and inputs are identical across projection layouts.

Each candidate uses independent native opset-18/21 exports with dynamic sequence axes,
CPUExecutionProvider, ORT_ENABLE_ALL, one intra/inter-op thread, 10 warmups and three
alternating-order sweeps of 300 calls. Profiling uses a separate 10-call session. Initialization
and weight packing are excluded. Values below are opset-18 medians of repetition medians.

| Block/layout | Query tokens | FP32 µs | Native dynamic INT8 µs |
|---|---:|---:|---:|
| Encoder, unchanged | 1 | 393.90 | 137.10 |
| Encoder, attention packed | 1 | 389.25 | 125.71 |
| Encoder, SwiGLU packed | 1 | 391.21 | 120.79 |
| Encoder, both packed | 1 | 388.06 | 106.85 |
| Encoder, unchanged | 128 | 1758.96 | 3672.31 |
| Encoder, attention packed | 128 | 1780.42 | 2950.37 |
| Encoder, SwiGLU packed | 128 | 1897.25 | 2353.94 |
| Encoder, both packed | 128 | 1831.33 | 1637.06 |
| Cross-attention, unchanged | 1 | 241.56 | 727.67 |
| Cross-attention, KV packed | 1 | 260.29 | 263.67 |
| Cross-attention, unchanged | 128 | 590.44 | 1135.12 |
| Cross-attention, KV packed | 128 | 610.44 | 672.44 |

Combined packing improves the encoder's native INT8 path by 1.28× at one token and 2.24×
at 128 tokens, reducing integer projections from seven to four. FP32 improvements are not
consistent: combined packing is about 4% slower at 128 tokens, and packing context KV also
regresses FP32 here. Native INT8 cross-attention remains slower than its FP32 baseline despite
its improvement from packing. Keep all layouts explicit rather than changing defaults.

All 144 graph variants execute at query/context pairs (1,1), (2,7), (7,2), and (31,13):
576 graph/length validations. FP32 matches PyTorch; native dynamic and four fixed-grid integer
paths return finite FP32 outputs with correct shapes. All 72 graph pairs have identical hashes
between the one-token and 128-token measurements. Only FP32/native dynamic candidates are
timed in this run; fixed-grid integer paths are correctness diagnostics.

Native INT8 relative MSE against unchanged FP32 is about 0.000033 for the combined one-token
encoder and 0.000018 at 128 tokens. These synthetic errors are not task-accuracy budgets.

Raw reports, graphs, profiles, the generated adapter, and source fingerprints remain in
`work_dir/projection_packing/`. Re-run the adapter from that directory with the adjacent
QuantSmith checkout and its locked example dependencies:

```sh
PYTHONPATH=../..:../../../QuantSmith \
  uv run --locked --group examples --project ../../../QuantSmith benchmark.py
```

`environment.json` reports installed distribution metadata (including Stackformers 4.6.0 in
QuantSmith's pinned environment); `source.json` identifies the actual selected local 5.0.0rc11
code and its hashes. The QuantSmith dependency pin was not changed.

### Unproven

CUDA execution, trained-model task quality, optimizer-state migration, and cached execution of
the packed variants have not been established. Existing cache wrappers continue to use ordinary
attention. Quantization and workload-dependent precision selection remain deployment concerns.


## 2026-09-26 — Precise cache and vendor input contracts (5.0.0rc7)

### Observation

Cache admission accepted `object` despite receiving typed attention collaborators. The vendor
wrapper widened a callable's declared return type to `object`, and fake kernels hid their
known keyword request schema behind the same annotation.

### Decision

Reuse `SelfAttn` and `CrossAttn` for admission inputs. Capture the vendor callable's return type
with a type parameter alongside its ParamSpec, while still validating the actual response into
`Result[Tensor, VarlenFallback]`. Describe fake kernel keyword arguments with a TypedDict.
Keep broad annotations confined to genuinely arbitrary export/SDK values pending validation.
No tensor computations, error policy, or public model APIs change.

### Verified

Ruff and strict Pyrefly pass. Backend, core admission, and cached-decoder regressions pass:
86 passed, 6 expected failures, and 6 optional-case unexpected passes. These checks include
cached/eager output parity, export, invalid vendor payloads, fallback warnings, and error
identity. Production Python signatures contain no `object` annotations.
Negative checker probes reject unrelated collaborators, incorrect tensor arguments, missing
required arguments, and unexpected vendor keywords.

### Unproven

The experimental vendor kernel itself is faked on CPU; CUDA execution has not been measured.

## 2026-09-26 — Native NamedTuple Result records (5.0.0rc6)

### Observation

The external `returns` container added a dependency and an unchecked unwrap API. Cached layer
construction also recovered from anticipated validation failures by raising and catching them.

### Decision

Replace it with one project-owned `Result[T, E] = Ok[T] | Err[E]` alias and two immutable generic
NamedTuple records. Carry native payloads and original error causes without copying. Keep
recoverable core failures as data, validate cache admission before construction, and preserve
the existing public tensor/configuration, validation-exception, and fallback-warning contracts.

Use keyword class patterns. Positional patterns on these generic NamedTuples fail PyTorch 2.11
Dynamo's `__match_args__` lookup during strict tracing. Keyword patterns pass without compiler
disables, graph breaks, type suppressions, or casts.

### Verified

The full quality gate passes: 1273 passed, 21 skipped, 162 expected failures, and 87 optional-case
unexpected passes. Strict/non-strict dynamic export, exported NamedTuple outputs, tensor
identity and autograd, and full-graph compilation of both tensor and Result outputs pass.
Compiled recovery handles an empty input as data. Three cache-admission regressions prove that
expected rejection never invokes an executor constructor. Existing model/API compatibility
and ONNX export checks pass.

Ruff and strict Pyrefly pass without new suppressions. Four negative checker probes reject
unnarrowed payload access, missing variants, and wrong success/error types. The `returns`
dependency has been removed with uv; an extracted-wheel smoke test verifies the public Result
exports, native tensor autograd, preserved error identity, and dependency metadata.

### Unproven

Compilation checks use the CPU eager backend. CUDA execution and Inductor performance have
not been measured. Python type annotations do not guarantee that arbitrary code never raises.

## 2026-09-26 — Normalize the experimental kernel contract (5.0.0rc4)

### Observation

The injected kernel protocol exposed the vendor's raw return schema. Core execution had to
validate that response and translate vendor exceptions despite already returning a Result.

### Decision

Require injected kernels to return `Result[Tensor, VarlenFallback]`. Wrap the imported vendor
callable once, preserving its parameter signature, validating its unknown response, and
converting the existing RuntimeError/TypeError fallback cases into failures with original
exception identity. Keep warning policy and the public tensor-or-None API unchanged.

### Verified

All 16 backend tests pass, including loader response validation, normalized outcome identity,
tensor identity, original error identity, warnings, and propagation of unexpected exceptions.
Strict Pyrefly accepts the normalized protocol and signature-preserving wrapper without casts
or suppressions. A negative checker probe rejects a raw tensor-returning callable with the
correct input signature while accepting its normalized wrapper.

The full quality gate passes: 1261 passed, 21 skipped, 162 expected failures, and 87 optional-case
unexpected passes. Existing model/API compatibility, Torch export, and compilation checks pass.
The signature fallback test uses an incompatible callable so Python generates the TypeError.

### Unproven

These tests fake vendor execution on CPU; they do not exercise the CUDA kernel.

## 2026-09-26 — Variable-width encoder schedules and explicit heads (5.0.0rc3)

### Observation

The convenience preset required a head-dimension list, shared the other settings across all
layers, and inferred query heads from residual width. The in-progress schedule support also
changed the public head-dimension length error without changing its Result validation core.

### Decision

Accept scalar or per-layer head dimensions, query heads, causal flags, feed-forward multipliers,
dropout, and normalization placement. Omitted heads retain exact legacy inference; explicit
heads allow compressed or expanded attention independently of the residual width. Keep
schedule failures as Result data, restore legacy width diagnostics, and validate lists before
constructing modules. Use a closed TypedDict for the test settings so strict Pyrefly can check
keyword expansion without suppressions.

### Verified

The full quality gate passes: 1256 passed, 21 skipped, 162 expected failures, and 87 optional-case
unexpected passes.
Explicit compressed and expanded projections run padded and packed inputs with finite nonzero
gradients. Equivalent explicit head counts preserve the legacy factory config, strict
checkpoint loading, and exact outputs. Scheduled normalization selects the requested layer
types, and strict Torch export preserves the new independent geometries' eager output. Existing
required ONNX export cases, formatting checks, Ruff, and strict Pyrefly pass.

### Unproven

GPU execution and throughput of the new explicit-head geometries have not been measured.

## 2026-09-26 — Independent attention projection widths (5.0.0rc2)

Commit: `ce443ff`

### Observation

Attention projects model features into `heads * dim_head` query features and projects the
result back to the model width. Requiring `dim` divisible by `heads` rejected valid geometry
such as `dim=512`, `heads=12`, and `dim_head=64` before these projections could run.

### Decision

Remove the model-width divisibility check from both attention configs. Keep Result-based
validation for the required query/key-value head grouping. Retain the existing non-square
projection warnings and the convenience preset's inferred head-count policy.

### Verified

Twelve parameterized CPU regression cases cover compressed and expanded attention widths
for self- and cross-attention with MHA, GQA, and MQA. Padded and packed outputs agree, retain
the model width, and produce finite nonzero input and query-projection gradients. Invalid
key-value head grouping still produces the existing Pydantic validation errors.

The isolated full quality gate passes: 1229 passed, 21 skipped, 162 expected failures, and
87 optional-case unexpected passes. Sixty compatibility cases use frozen outputs,
input gradients, and checkpoint shapes captured from the verified `5.0.0rc1` wheel at commit
`c40fbaf`. Existing positional and keyword constructor/call APIs match those observations
under omitted defaults, square/compressed/expanded projections, MHA/GQA/MQA, Q/K normalization,
and causal self-attention across padded and packed inputs. The reference fixture identifies
its source release and is independent of current attention code.

### Unproven

The new geometries have not been exercised on GPU hardware or benchmarked for throughput.

## 2026-09-26 — Shared Result core with compatible public adapters

Commit: `738a3ad`

### Observation

Expected layout, geometry, cache-admission, and experimental backend failures were implicit
in core signatures. Local success/error records would also differ from the Result types used
by other libraries, requiring callers to recognize and translate each project's variants.

### Decision

Use `returns.result.Result`, `Success`, and `Failure` directly. Keep pure admission checks
separate from public exception and warning policy. Carry existing exceptions as failure data;
the public adapter raises the original object so its type, message, and cause survive. Preserve
Pydantic validation errors, native tensor signatures, and the existing tensor-or-None backend
fallback contract. Inject the optional backend kernel into the Result attempt and preserve a
typed fallback kind plus original failure detail. Keep impossible factory branches checked
with `assert_never`.

Use returns' ordinary constructor and matching API under strict Pyrefly. An isolated dependency
comparison passed strict PyTorch export with returns 0.28.0; Expression 5.7.0 failed during
tagged Result construction in strict export. The project uses no custom Result variants.

### Verified

Shared-library success identity/autograd and exception identity/cause tests pass. Dynamic
`torch.export` passes in both tracing modes and `torch.compile` passes with `fullgraph=True`
using the eager backend. The public experimental backend's warning/fallback tests pass.
`just check` passes on Python 3.11: 1155 passed, 21 skipped, 162 expected failures, and
87 optional-case unexpected passes. All 37 added regression cases pass. Ruff and strict
Pyrefly pass without additional suppressions. A negative checker probe rejects wrong success
and error types plus incompatible mapping callbacks while accepting typed ordinary mapping.

### Unproven

GPU kernel execution and accelerator compilation were not exercised. Result annotations make
expected outcomes visible but do not prove freedom from unexpected Python or PyTorch exceptions.
Returns' advanced composition and decorator APIs have not been checked with Pyrefly.

## 2026-09-26 — Empty-input contracts and strict Python checks (4.7.0b8)

Commit: `4fe2960`

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
