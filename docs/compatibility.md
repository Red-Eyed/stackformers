# Compatibility and evidence

Package metadata declares Python >=3.11 and PyTorch >=2.2. Those lower bounds are
installation constraints, not a claim that every optional export or experimental kernel
path works on every allowed version. The development environment may be substantially newer.

## Where to find evidence

| Capability | Evidence in this repository | Limit |
| --- | --- | --- |
| CPU eager examples | `tests/test_examples.py` and the runnable examples | Synthetic inputs; no task-accuracy result |
| Padded/packed numerical agreement | Layout parity example and attention tests | Numerical tolerance depends on dtype/backend |
| Device/dtype coverage | `device_dtype` fixture in `tests/conftest.py` | CUDA cases are collected only when CUDA is available |
| Compilation and export | Compile/export cases under `tests/` | Support applies to the exercised model and input paths |
| Cached ONNX Runtime decoding | Export example and inference tests | Read the strategy-specific opset and runtime contracts |

Run `just check` to evaluate the available local environment. Passing CPU checks does not
establish CUDA support. The [CI workflow](../.github/workflows/check.yml) configures
Python 3.11/3.12 on Linux CPU with locked dependencies. Workflow configuration is not a passing result; consult the
[Actions runs](https://github.com/Red-Eyed/stackformers/actions) for executed evidence.
This matrix does not establish minimum-PyTorch or GPU compatibility.

## Backend selection

Packed inputs avoid representing padding in their public token buffer. Execution can still
use a padded fallback. PyTorch's experimental variable-length attention is attempted for
eligible CUDA fp16/bf16 inputs; import availability, attention bias, and kernel failures can
cause fallback. CPU and float32 packed execution use the fallback path.

There are no published performance results in this repository. Measure representative
length distributions and include conversion costs where your application pays them.

## Export boundaries

List and nested-tensor constructors are eager adapters. Perform those conversions outside
the graph unless a specific path is documented as exportable. Custom modules need their own
compile/export validation. Cached adapters accept supported built-in implementations;
they cannot infer the behavior of arbitrary overridden forwards.

See [inference contracts](inference.md) for ONNX opsets, dynamic axes, cache ownership, and
runtime binding requirements. Declaring a minimum PyTorch version does not supply the ONNX
and ONNX Runtime APIs required by newer deployment strategies.
