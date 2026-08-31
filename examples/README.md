# Examples

Each example is a small, deterministic program built from the public Stackformers API. They run
on CPU and avoid external datasets, tokenizers, or model downloads.

| Example | What it demonstrates |
| --- | --- |
| [`padded_and_packed.py`](padded_and_packed.py) | One encoder and one set of weights processing equivalent padded and packed variable-length batches. |
| [`causal_language_model.py`](causal_language_model.py) | A decoder-only language model with token embeddings, causal attention, next-token loss, and an optimizer step. |
| [`image_encoder.py`](image_encoder.py) | Image patch projection, explicit row/column positions, RoPE-2D, pooling, and classification. |
| [`point_cloud_encoder.py`](point_cloud_encoder.py) | Continuous 3-D positions, coordinate centering, RoPE-ND, and translation invariance. |
| [`subclassed_encoder.py`](subclassed_encoder.py) | A focused preset-hook override and a fully user-owned `TransformerEncoderBase` subclass. |
| [`encoder_decoder_onnxruntime.py`](encoder_decoder_onnxruntime.py) | Separate encoder, cross-cache, and cached-decoder ONNX graphs with ONNX Runtime parity. |

Run any example from the repository root:

```bash
uv run examples/padded_and_packed.py
uv run examples/causal_language_model.py
uv run examples/image_encoder.py
uv run examples/point_cloud_encoder.py
uv run examples/subclassed_encoder.py
uv run examples/encoder_decoder_onnxruntime.py
```

## Encoder–decoder ONNX Runtime

[`encoder_decoder_onnxruntime.py`](encoder_decoder_onnxruntime.py) builds one ordinary
`TransformerEncoder` + `TransformerDecoder` model and exports the same modules along deployment
boundaries suitable for separate C++ ONNX Runtime sessions:

1. Eager PyTorch calls `model(source, target)` with the existing Stackformers API.
2. `encoder.onnx` runs once and returns encoder memory.
3. `decoder_cross_cache.onnx` consumes that memory once and returns one dense cross-attention K/V
   cache plus the context mask.
4. `cached_decoder.onnx` consumes one target token, the cross cache, a required growing self cache,
   and `step_i`; it returns the decoder output and the self cache for the next call.

Run the complete export and parity demonstration from the repository root:

```bash
uv run examples/encoder_decoder_onnxruntime.py
```

The ONNX tensors have explicit names suitable for equivalent C++ `Ort::Value` maps:

- Encoder inputs: `source_x`, `source_mask`, `source_positions`.
- Encoder outputs and cache-builder inputs: `memory_x`, `memory_mask`, `memory_positions`.
- Cache outputs and cached-decoder inputs: `cross_kv_cache`, `context_mask`.
- Other cached-decoder inputs: `target_x`, `target_mask`, `target_positions`, `self_kv_cache`,
  `step_i`.
- Cached-decoder outputs: `decoder_output`, `self_kv_cache_out`.

The cache builder is decoder-owned because its K/V projections use decoder weights. C++ retains
the immutable cross cache and passes the returned self cache into the next call. The first call
uses a real self-cache tensor with shape `(layers, 2, batch, kv_heads, 0, dim_head)`, never an
optional or `None`. Batch, source-length, and past-target-length axes are exported dynamically;
architecture-constrained axes remain static.
