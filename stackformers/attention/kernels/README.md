# attention/kernels

One file per `AttnKernel` implementation. Add new kernels here without touching any other module.

## Implementing a new kernel

Satisfy the `AttnKernel` protocol — do not import it:

```python
def forward(
    self,
    q: Float[Tensor, "b h n dh"],   # or "nt h dh" for packed kernels
    k: Float[Tensor, "b h s dh"],
    v: Float[Tensor, "b h s dh"],
    attn_mask: ...,
    attn_bias: ...,
    is_causal: bool,
) -> Float[Tensor, "b h n dh"]: ...
```

Packed kernels use `(nt, h, dh)` tensors and accept `cu_seqlens` instead of `attn_mask`.

## Internal helper

`_mask.py` — shared utilities for building key-padding masks from `SequenceInfo`. Not a public API.
