# Using Stackformers in research

Stackformers supplies model components. Your experiment owns the dataset, preprocessing,
training loop, evaluation, and scientific claims. The bundled synthetic examples demonstrate
integration and numerical properties; they are not evidence of downstream accuracy.

## Record the implementation

For each experiment, retain:

- The exact Stackformers version and, for an unreleased checkout, Git commit and local patch.
- The serialized model configuration and the application code that reconstructs the model.
- Model weights, optimizer/scheduler state for resumed training, and random-number state.
- Dataset versions, splits, preprocessing, coordinate units, and augmentation settings.
- Python, PyTorch, accelerator/runtime versions, device, dtype, seeds, and determinism settings.
- The application dependency lockfile and the command used to run training and evaluation.

Use `uv add "stackformers==<validated-version>"` in the consuming project, substituting
the actual version you tested. Commit that project's `uv.lock`; reproduce its environment
with `uv sync --locked`. A repository checkout can use `uv sync --locked --group dev`.
The library's lockfile alone does not describe your application's full environment.

Stable installations use the published stable release by default. To evaluate a newer
prerelease, use `pip install --upgrade --pre stackformers` or
`uv add --upgrade --prerelease allow stackformers`, then pin the exact version you validate.
Consult the [release history](../CHANGELOG.md) before upgrading and rerun your experiment's
parity checks; prereleases can change interfaces before the corresponding stable release.

## Save configuration and weights together

For the built-in encoder preset:

```python
from pathlib import Path

import torch
import stackformers as sf

config = sf.plain_encoder_config(dim=64, heads=1, num_layers=2, ff_mult=3.0)
model = sf.TransformerEncoder(config)
Path("encoder.json").write_text(config.model_dump_json(indent=2))
torch.save(model.state_dict(), "encoder.pt")

restored_config = sf.TransformerEncoderConfig.model_validate_json(
    Path("encoder.json").read_text()
)
restored = sf.TransformerEncoder(restored_config)
restored.load_state_dict(torch.load("encoder.pt", weights_only=True))
restored.eval()
```

Custom collaborators require their own reconstruction code. A state dict does not encode
arbitrary Python implementations. If you pack projection layouts, save the resulting model's
configuration after conversion; see the [API guide](api-guide.md#convert-projection-layouts).

## Report what changed in an ablation

Keep data splits, training budget, evaluation, and unrelated architecture settings fixed.
Save a full configuration for each run and describe parameter-count or compute changes
introduced by a substituted component. Serializable configuration helps document an
experiment; it does not guarantee deterministic results across environments.

For coordinate-based models, record coordinate centering, scaling, units, and positional
encoding parameters. Follow the [positional encoding guide](../stackformers/positional/README.md).

## Cite the software

[CITATION.cff](../CITATION.cff) provides software citation metadata. Report the exact version
or commit alongside that citation. No DOI or associated Stackformers research paper is
currently declared in this repository. Cite the original methods separately when relevant;
a software citation does not replace attribution to their authors.

If a paper uses Stackformers as an implementation dependency, describe that relationship
accurately. Use in a paper does not establish the library as a scientific contribution or
validate its performance. Follow the venue's current anonymity and artifact rules for
submission materials; keep paper-specific publicity coordinated with the paper's release.
