# All-valid inference and ONNX cache contracts

Projection packing, input validity, and cache storage are independent choices. None is
selected automatically from tensor values, a batch size of one, or evaluation mode.

## Packed projections

Load an ordinary checkpoint into the ordinary model, then call `sf.to_packed(model)` before
building wrappers. Alternatively, construct with packed attention/feed-forward configs and
load a matching packed checkpoint. `PackedInput` describes a sequence layout; it does not
select packed projections. Default presets still construct ordinary projections.

Packing retains parameter values, normalization, RoPE, output projections, dtype/device,
training state, and trainability. Floating-point rounding can differ. Converting back restores
the ordinary checkpoint layout exactly. Convert before constructing an optimizer. If wrappers
already exist, convert a containing module that registers them and the model together so
shared references remain shared. Export uses the selected modules without unpacking them.

At ONNX opsets 23 and 24, packed self-attention has one QKV MatMul followed by Split; each
packed GLU has one gate/value input MatMul followed by Split and a separate output projection;
cross-cache construction has one KV MatMul followed by Split. Repeated cross-attention calls
project only Q. Graph structure is verified separately from eager numerical parity.

## Explicit validity

Use `sf.PaddedInput(x, sf.AllValid(), abs_positions)` only when every token is valid. The
mask field is required and accepts either a boolean tensor or the explicit `AllValid()`
marker. There is no `None` mode or omitted-mask default. `SequenceInput` retains just its
padded and packed layouts. This replaces the prerelease `AllValidInput` record.

Encoder/full-prefix paths accept both mask variants through the existing stacks, in both
projection layouts. `padded_sdpa` owns their common dispatch. The marker's global attention
path excludes padding construction, empty-row detection and guards. Learned biases, sliding
windows and explicit pair masks are rejected for the marker; use a tensor mask for those
cases. Cross-attention query and context masks are handled independently.

`AllValid` is stateless and registered as a zero-leaf pytree, so it contributes no graph input.
`sf.materialize_padding_mask(input)` adapts to consumers requiring a boolean tensor: existing
tensors retain identity, and only the marker allocates an all-true mask. Attention does not
call this adapter on its all-valid path. Legacy `PaddedSequence` metadata remains tensor-only.

`sf.AllValidCachedDecoderWrapper` requires exactly one token and batch size one. Its self
cache includes only valid past tokens, and the new token is appended before attention.
Its cross cache must contain only valid context positions. These calls use
`attn_mask=None, is_causal=False`: an upper-left one-query causal mask would hide valid past
positions. Full-prefix self-attention still applies its configured causal visibility.
Coordinates retain their supplied meaning and are not inferred from cache length.

```python
decoder.eval()
sf.to_packed(decoder)
builder = sf.AllValidCrossAttentionCacheBuilder(decoder)
cached = sf.AllValidCachedDecoderWrapper(decoder)

with torch.no_grad():
    cross = builder(sf.PaddedInput(memory_x, sf.AllValid(), memory_positions))
    past = cached.empty_cache(dtype=torch.float32)
    result = cached(sf.PaddedInput(token_x, sf.AllValid(), token_positions), cross, past)
    past = result.self_kv_cache
```

The functional all-valid path supports opsets 23 and 24. It preserves incoming cache tensors
and returns separate per-layer pairs, avoiding the original API's stacking. It still copies
past keys/values through concatenation. The original `CachedDecoderWrapper` remains available
with its existing padded records, dense cache tensors, `step_i`, and fully masked-row behavior.
`empty_cache` derives each layer's geometry/device directly from its self-attention module;
cross-attention heads and preset metadata need not match. Both cache strategies share this
allocator and the existing per-layer K/V record. The snippet assumes float32 execution; use
the autocast projection dtype under autocast. The specialized all-valid wrappers require the
marker explicitly and reject even an all-true tensor mask.

When describing dynamic cache dimensions with `torch.export.ShapesCollection`, assign `{}`
to static tensor leaves as well as `{2: past_dim}` to self-cache leaves. PyTorch 2.11's ONNX
shape conversion otherwise mistakes a completely static NamedTuple of `None` entries for a
single tensor's axis list. Trace with at least two past positions and execute the exported
graph with zero and increasing past lengths.

## Fixed-capacity strategy at opset 24

`sf.PreallocatedCachedDecoderWrapper(decoder)` is a separate inference-only executor. Its
`allocate_cache(capacity, dtype=...)` allocates zero-filled, contiguous, disjoint K/V buffers
on the model device. Use the projection dtype, including bf16 when decoding under bf16
autocast. Each tensor has shape `(1, kv_heads, capacity, dim_head)`. No leading layer or K/V
selector dimension is stacked into the cache. Different layer head geometries remain separate.

`PreallocatedDecoderCache(kv, valid_length)` carries these buffers and an int64 length tensor
of shape `(1,)`. Length zero is valid. A call writes the new positioned K and raw V at the old
length, attends through old length plus one, and returns the incremented length. Pass the
returned record to the next call. Token coordinates remain independent of physical slots.

The ownership contract is:

- One decoding sequence exclusively owns the buffers and its current length. Calls cannot
  overlap. Independent branches or concurrent requests need separate buffers.
- Eager execution mutates K/V in place under `no_grad`. Previously retained records alias
  later writes; clone buffers when a persistent snapshot is required. The length input is
  not mutated; its returned replacement must accompany the updated buffers.
- Capacity is fixed for a decoding sequence. Before append, require
  `0 <= valid_length < capacity`. Overflow raises before eager mutation. There is no automatic
  growth, eviction or wrapping. Allocate a larger cache outside decoding
  if the application needs another capacity.
- Unused capacity must remain finite. Allocation zeros it; resetting the length permits reuse
  of finite previous values. Cross caches stay immutable and do not alias the writable buffers.
- Model parameters and source checkpoint names remain unchanged. The executor shares source
  modules and requires evaluation mode. Construction should precede compilation/export.

`validate_cache` checks eager host length, geometry, execution dtype/device, contiguity and
disjoint storage. Tensor-value checks are deliberately outside captured execution. **An ORT
or compiled host must validate the length before every call**, as well as buffer metadata at
allocation/binding. Export does not embed an overflow exception. Do not rely on an out-of-range
TensorScatter error to provide the same pre-mutation guarantee.

`attention/onnx_backend.py` owns the update and valid-length attention lowerings and their
eager PyTorch equivalents. The exported graph uses `TensorScatter` for each K/V update and `Attention`'s
`nonpad_kv_seqlen` input for valid-prefix attention. No padding mask is built and `is_causal`
is false. GQA consumes the smaller KV-head tensors directly, avoiding an explicit expansion
of the entire cache to query-head width. This exporter uses `torch.onnx.ops.symbolic`; the
verified environment is PyTorch 2.11, ONNX 1.22, and ONNX Runtime 1.29 CPU. Older PyTorch
versions without that symbolic API cannot export this strategy. Use `opset_version=24`
explicitly. The original functional path continues to support opset 23.

## Actual ORT buffer sharing

An in-place-looking PyTorch operation, a Scatter operator, and fixed shapes do not themselves
eliminate a copy. ONNX graphs are functional. ONNX's opset-24 cache pattern permits backend
aliasing only when the old buffer is no longer needed. See the
[ONNX cache specification](https://onnx.ai/onnx/technical/kv_cache.html).

With ordinary `session.run()`, the tested ORT 1.29 CPU TensorScatter allocated another output
and logged that in-place optimization was inactive. To request sharing:

1. Keep one correctly typed, contiguous CPU `OrtValue` alive for each writable K/V tensor.
2. Bind each cache graph input and its corresponding output to that **same OrtValue**.
3. Keep cross-cache buffers immutable and separate. Bind valid length separately and replace
   it with the returned length after each call. The scalar length is not a shared cache buffer.
4. Keep bound storage alive until the run completes. Do not resize, reuse or free it while ORT
   may access it. Match execution provider, device, dtype, shape and capacity to the graph.
5. Check the exported update wiring and actual output pointers on the deployed ORT/provider.
   This graph exposes each TensorScatter result directly as an output and its old cache
   directly as an input, without cache slicing or final stacking that could hide copies.

The relevant binding operations are:

```python
binding.bind_ortvalue_input(past_name, cache_value)
binding.bind_ortvalue_output(present_name, cache_value)
session.run_with_iobinding(binding)
```

I/O binding controls external buffers; it does not promise that arbitrary intermediate
operators avoid copies. Read the [ORT I/O binding contract](https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.OrtIoBinding.html).
The full decoder regression checks four successive updates from zero length, unchanged output
buffer pointers, untouched unused slots, and full-prefix numerical parity for ordinary/packed
MHA/GQA. That establishes cache-buffer reuse on the tested CPU runtime. It does not establish
zero temporary allocation throughout attention or a latency improvement on ARM phones.

## Validation and performance claims

Tests use deterministic synthetic values, explicit coordinates and controlled weights. They
cover all decoder normalization topologies, empty and multi-step caches, state preservation,
exact packing/checkpoint round trips, dtype coverage and CPU bf16 autocast. ONNX tests execute
opsets 23/24 on `CPUExecutionProvider`, inspect projection counts and splits, and reject
padding-related `IsInf`, comparisons, reductions and `Where` guards in all-valid graphs.

Use torch-harness's `write_text_graph` for graph summaries. Inspect operator wiring separately:
the summary intentionally omits it. These graph and storage results are not latency numbers.
ARM latency, peak memory and runtime allocator/copy behavior must be measured on the target
CPU and deployed ORT build before making deployment speed claims.
