# Stackformers

**Composable PyTorch transformers for research.**

Build vision, spatial, and sequence models with interchangeable attention, positional
encodings, normalization, and feedforward components. Start with a preset, then replace
individual components as your experiments evolve.

- **Experiment with architecture:** explicit, serializable configurations and structural
  protocols let you supply your own components.
- **Work with variable-length data:** run the same encoder weights on padded or packed
  token batches, with explicit token coordinates.
- **Connect research and deployment:** examples cover image patches, point clouds,
  causal modeling, custom components, and cached ONNX Runtime decoding.

## Try the current API

Python 3.11 or newer is required. This checkout is on the **5.0 release-candidate series**.
To run the examples shown here against the matching source:

```bash
git clone https://github.com/Red-Eyed/stackformers.git
cd stackformers
uv run examples/padded_and_packed.py
```

`uv run` prepares the project environment automatically. For a published release in an
existing project:

```bash
uv add stackformers
```

An unpinned install may resolve a different stable release whose API differs from this
checkout. Use the documentation matching your installed version. For experiments, pin the
exact version or commit you validated; see the [research guide](docs/research.md).

## Run your first encoder

```python
import torch
import stackformers as sf

model = sf.TransformerEncoder(
    sf.plain_encoder_config(dim=64, heads=1, num_layers=2, ff_mult=3.0)
).eval()

sequences = [torch.randn(8, 64), torch.randn(5, 64)]
padded = sf.PaddedTensor.from_list(sequences)
with torch.no_grad():
    padded_output = model(padded)              # (2, 8, 64)
    packed_output = model(padded.to_packed())  # (13, 64)
```

The two calls share model weights. Packed output contains valid tokens only; padded output
retains batch and sequence axes. See the [layout parity example](examples/padded_and_packed.py)
for a numerical comparison. Packed input does not guarantee faster execution: backend,
hardware, dtype, and length distribution matter.

## Choose an example

Examples run on CPU without dataset or model downloads. From a repository checkout,
`uv run` prepares the project environment automatically.

| Your task | Start here | Run |
| --- | --- | --- |
| Image patches and 2-D positions | [Image encoder](examples/image_encoder.py) | `uv run examples/image_encoder.py` |
| Continuous spatial coordinates | [Point-cloud encoder](examples/point_cloud_encoder.py) | `uv run examples/point_cloud_encoder.py` |
| Unequal sequence lengths | [Padded and packed](examples/padded_and_packed.py) | `uv run examples/padded_and_packed.py` |
| Replace a model component | [Custom encoders](examples/subclassed_encoder.py) | `uv run examples/subclassed_encoder.py` |
| Autoregressive token modeling | [Causal language model](examples/causal_language_model.py) | `uv run examples/causal_language_model.py` |
| Export an encoder–decoder | [ONNX Runtime guide](examples/README.md#encoderdecoder-onnx-runtime) | `uv run examples/encoder_decoder_onnxruntime.py` |

These are integration examples with synthetic inputs, not trained models or task-accuracy
benchmarks. Stackformers provides transformer components; your application owns tokenization,
data processing, training, and task heads.

## What can you change?

| Component | Included choices |
| --- | --- |
| Attention | Global self/cross attention, local self-attention, GQA/MQA |
| Token layout | Padded or packed, with explicit positions |
| Positional encoding | RoPE-1D, RoPE-2D, continuous-coordinate RoPE-ND, or none |
| Feedforward | SwiGLU, HardSwishGLU, GEGLU, GELU, ReLU² |
| Normalization | RMSNorm, LayerNorm; pre, post, sandwich, or reordered placement |
| Model assembly | Encoders, decoders, cross-attenders, variable-width encoders |
| Deployment | Projection packing, compile/export paths, cached decoder adapters |

Presets build components from discriminated Pydantic configurations. For deeper changes,
wire modules directly or implement an owning module's protocol. The
[configuration and API guide](docs/api-guide.md) covers both approaches.

## Compatibility and limits

- Global attention uses PyTorch SDPA. Eligible CUDA fp16/bf16 packed workloads use
  PyTorch's variable-length attention; other paths can fall back to padded execution.
- Compile and export support depend on the component and input path. Eager list/nested
  adapters and deployment cache strategies have separate contracts.
- Cached deployment supports specified built-in classes; custom forward overrides need
  their own deployment integration.
- No speedup or downstream accuracy improvement is claimed without a matching benchmark.

See [compatibility evidence](docs/compatibility.md) and
[inference contracts](docs/inference.md) before selecting a deployment path.

## Documentation

- [Examples and expected behavior](examples/README.md)
- [Research reproducibility and citation](docs/research.md)
- [API, layouts, configuration, and migration](docs/api-guide.md)
- [Deployment and inference](docs/inference.md)
- [Release history](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Adoption and evidence roadmap](docs/improvement-plan.md)

## Development

```bash
git clone https://github.com/Red-Eyed/stackformers.git
cd stackformers
just sync
just check
```

The [Justfile](Justfile) owns formatting, linting, type-checking, tests, and builds.

## Citation and license

If Stackformers contributes to your research implementation, cite the software using
[CITATION.cff](CITATION.cff) and report the exact version or commit used. No associated
research paper or conference endorsement is implied.

Released under the [MIT license](LICENSE).
