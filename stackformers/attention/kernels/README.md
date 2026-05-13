# attention/kernels

`AttnKernel` implementations — one file per kernel. Padded kernels operate on `(b, h, n, dh)` tensors; packed kernels operate on `(nt, h, dh)` flat tensors with cumulative sequence lengths.

Packed kernels use `torch.nn.attention.varlen.varlen_attn` on CUDA with fp16/bf16 and fall back to a per-sequence loop otherwise.

## Adding a new kernel

1. Add a config class with a `kind: Literal[...]` discriminator to `config.py` and include it in the `KernelConfig` union.
2. Implement the class satisfying `AttnKernel` structurally — do not import the protocol.
3. Add a `case` branch in `factory.py::build_kernel`.
