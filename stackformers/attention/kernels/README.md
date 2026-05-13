# attention/kernels

One file per `AttnKernel` implementation. Add new kernels here without touching any other module.

## Files

| File | Contents |
|------|----------|
| `sdpa.py` | `SDPAKernel` — padded full attention |
| `windowed.py` | `WindowedSDPAKernel` — padded sliding-window attention |
| `varlen.py` | `VarlenSDPAKernel` — packed full attention |
| `varlen_windowed.py` | `VarlenWindowedSDPAKernel` — packed sliding-window attention |
| `config.py` | `SDPAKernelConfig`, `WindowedSDPAKernelConfig`, `VarlenSDPAKernelConfig`, `VarlenWindowedSDPAKernelConfig`; discriminated union `KernelConfig` |
| `factory.py` | `build_kernel(config, causal, dropout) -> AttnKernel` — dispatches on `kind` |
| `_mask.py` | `build_window_mask` — shared helper, not public API |

## Implementing a new kernel

Satisfy the `AttnKernel` protocol — do not import it:

```python
def forward(
    self,
    q: Tensor,          # (b h n dh) padded  or  (nt h dh) packed
    k: Tensor,          # (b h s dh) padded  or  (nt h dh) packed
    v: Tensor,          # (b h s dh) padded  or  (nt h dh) packed
    q_seq_info: SequenceInfo,
    k_seq_info: SequenceInfo | None,
    attn_bias: Tensor | None,
) -> Tensor: ...
```

Then add a config class with a `kind: Literal["your_kind"]` field to `config.py`, include it in the `KernelConfig` union, and add a `case` branch in `factory.py::build_kernel`.
