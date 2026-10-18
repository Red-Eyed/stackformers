# Feed-forward Development Log

## 2026-10-18 — Packed GEGLU and HardSwishGLU (5.0.0rc13)

### Observation and decision

GEGLU and HardSwishGLU share the same gate/value projection opportunity as SwiGLU. Add
independent `PackedGEGLU` and `PackedHardSwishGLU` modules and concrete config variants.
Factories select the layout; existing conversion helpers own weight copying and trainability
checks. Keep activation choices local to each variant without introducing a behavior base
class, protocol, or constructor mode flag. Original forwards and state-dict keys remain intact.

### Verification scope

Verified outputs, input/parameter gradients, seeded training dropout, dynamic export, strict
ordinary/packed checkpoint reloads, config JSON dispatch, frozen parameters, training modes,
and independent weight/config ownership across supported devices and dtypes. `just check`
passes: 1,799 passed, 21 skipped, 172 expected failures, and 111 optional-case unexpected
passes; formatting, lint, and types also pass. Packing does not
migrate optimizer state, existing gradients, hooks, or aliases. Throughput and trained-model
quality for these two variants remain unmeasured.

## 2026-10-18 — Concrete packed SwiGLU config (5.0.0rc12)

`PackedSwiGLUConfig` selects the packed layout through the existing discriminated factory.
`PackedSwiGLU` accepts only this concrete config; ordinary `SwiGLUConfig` keeps its existing
constructor and checkpoint layout. Conversion and unpacking change config variants explicitly.
JSON roundtrip, direct factory construction, conversion metadata, and preset checkpoint
reloads are verified alongside the existing numerical parity tests. The forward equations
are unchanged; no new throughput or model-quality claim is made.

## 2026-10-18 — Opt-in packed SwiGLU (5.0.0rc11)

### Observation and decision

Gate and value share an input and can use one projection. Keep the original `SwiGLU.forward`
intact and add `to_packed()` returning an independent `PackedSwiGLU`. Preserve SiLU, the
multiplication, dropout placement, inner width and output projection. Keep checkpoint
conversion in focused helpers, and use `unpack()` to restore ordinary checkpoint names.
Store a config snapshot without changing the ordinary state dict.

### Verified result

The full project check passes. Focused tests cover forward/input-gradient/parameter-gradient
parity, padded and flattened token layouts, training dropout, dynamic export, checkpoint
serialization, frozen/tied projection values, and fresh SGD steps. An independently captured
Stackformers 4.6.0 checkpoint/output/input-gradient reference passes before and after conversion.
The original implementation and its checkpoint names remain available.

### Unproven claims

Conversion does not migrate existing gradients, optimizer state, hooks or parameter aliases.
GPU throughput, memory savings, convergence and trained-model task quality remain unmeasured.
See the model-level DEVLOG for complete-block deployment measurements and their scope.


## 2026-08-14 — Standard GELU FFN

### Motivation

The module exposed gated SwiGLU and GEGLU networks and a non-gated ReLU² network, but not the
standard GELU Transformer FFN used as the parameter-count baseline in their documentation.

### Decision

Add `GELUFFN` as a distinct, bias-free implementation selected by `GELUConfig(kind="gelu")`.
It uses the standard non-gated hidden width `int(dim * mult)`, PyTorch's built-in
`nn.GELU(approximate="tanh")`, dropout after activation, and a second projection back to the
model dimension. Existing presets remain on SwiGLU; selecting GELU is explicit through injected
configuration.

### Verified result

Ruff, formatting, and Pyrefly checks pass. The full suite passes with 281 tests and 19 skips;
focused device/dtype-expanded tests verify the output shape, bias-free projections, hidden
width, GELU approximation, input gradients, config round-tripping, and factory dispatch.

### Unproven claims

No training-quality, convergence, throughput, or memory comparison with the other feed-forward
variants has been run.
