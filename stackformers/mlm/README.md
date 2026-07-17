# mlm

A masked-token-reconstruction auxiliary loss, wrapped around any encoder that satisfies `EncoderLike`. Domain-agnostic — it has no idea what a token represents.

## Usage

```python
import torch
from stackformers import (
    MLMWrapper,
    MLMWrapperConfig,
    TransformerEncoder,
    make_padded_input,
    plain_encoder_config,
)

encoder = TransformerEncoder(plain_encoder_config(dim=512, heads=8, num_layers=6))
mlm_wrapper = MLMWrapper(MLMWrapperConfig(dim=512, mask_ratio=0.15))

x = torch.randn(2, 128, 512)
mask = torch.ones(2, 128, dtype=torch.bool)
input = make_padded_input(x, mask)

res = mlm_wrapper(input, encoder)

task_loss = my_task_head(res.out)  # res.out is always the clean, unmasked encoder output
loss = task_loss + 0.1 * res.mlm_loss  # zero automatically in eval — nothing to branch on
```

One call site covers both modes: in training, `mlm_wrapper` runs the encoder twice internally — once clean (returned as `res.out`) and once on a separately-masked copy (used only to compute `res.mlm_loss`) — so masking never reaches whatever consumes `res.out`. In eval, it runs the encoder once and `res.mlm_loss` is a constant zero. Pretraining with no main task yet — just drop `task_loss` and train on `res.mlm_loss` alone.

`mlm_wrapper` and `encoder` are separate modules — `encoder` is passed in at call time, never stored, so `mlm_wrapper.parameters()` holds only `mask_token`, `masking_strategy`, and `head`. Build one optimizer over both: `itertools.chain(encoder.parameters(), mlm_wrapper.parameters())`. And since they're separate modules, `.train()`/`.eval()` on one does not propagate to the other — call both, or register both as submodules of a shared parent so one call does.

## Design

**`out` is always clean; only `mlm_loss` ever reflects masking.** `forward(input, encoder)` runs `encoder(input)` unconditionally for `out`. In training it additionally runs `encoder` a second time on a masked copy, purely to produce `mlm_loss` — that corrupted copy is never returned. This means the main pipeline sees byte-identical output whether or not the aux loss is being trained alongside it, and callers never need to special-case which pass produced `out`.

**Training/eval is the one branch `forward()` is allowed, because it isn't tensor control flow.** `self.training` gates whether the second (masked) pass runs — the same mechanism `nn.Dropout` and `nn.BatchNorm` already use. It lets callers invoke `mlm_wrapper(input, encoder)` unconditionally in both modes, instead of an external `if training: ...` at every call site.

**A wrapper over an encoder supplied at call time, not one it owns.** `MLMWrapper` never stores an encoder — `forward` takes one as an argument, so nothing is registered as a submodule and `mlm_wrapper.parameters()` never includes the encoder's weights.

**The mask token is owned state, not an injected collaborator.** It's the wrapper's own learned parameter, analogous to a `nn.Linear`'s weight — not swappable behavior, so it isn't a `Protocol`.

**`input.x` is detached once, before it is used for anything — not only as the target.** The masked pass replaces `input` with `input._replace(x=input.x.detach())` up front, so both the unmasked context fed to the encoder and the reconstruction target come from that same detached copy. Detaching only the target is not sufficient: self-attention still mixes the live, undetached unmasked positions into the masked positions' predictions, so a gradient path back to whatever produced `input.x` would otherwise survive at every unmasked position — worse, it grows with the fraction of tokens left unmasked. Severing the whole input removes the standard collapse shortcut for self-supervised regression targets (drive every token toward one constant vector, and reconstruction becomes trivial) — not just the degenerate all-masked case, but the realistic partial-masking case a target-only detach would silently miss. See `tests/mlm/test_wrapper.py::test_wrapper_severs_gradient_to_upstream_input_under_partial_masking` for the regression test.

**No layout dispatch in the wrapper.** `MaskingStrategy` returns a boolean tensor shaped like `input.x`'s leading dims — `(b, n)` for `PaddedInput`, `(nt,)` for `PackedInput`. `torch.where` and boolean advanced indexing both broadcast/gather correctly against either shape, so `MLMWrapper.forward` never matches on the sequence variant; only `RandomMasking` does, internally.

**Random masking needs no packed-sequence boundary awareness.** Each token's masking decision is independent of every other token's, so document identity (`cu_seqlens`) never enters the decision. Boundary-awareness only becomes necessary for a contiguous-span (blockwise) strategy, which could otherwise straddle two packed documents — add that logic when that strategy is added, not before.

## Extending

To add a masking strategy (e.g. blockwise), write an `nn.Module` satisfying `MaskingStrategy` and pass it as `MLMWrapper(..., masking_strategy=...)`.

To add a reconstruction target type (e.g. a discretized/codebook target for BEiT-style cross-entropy, or a contrastive target), write an `nn.Module` satisfying `ReconstructionHead` and pass it as `MLMWrapper(..., head=...)`. Prediction and loss are scored together by one call — they're a coupled choice (a cross-entropy loss over a raw regression output isn't meaningful), not two independently pluggable axes.
