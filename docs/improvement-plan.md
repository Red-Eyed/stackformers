# Repository positioning and adoption plan

## Audience and promise

Primary audience: PyTorch researchers who need to change transformer architecture without
rewriting the surrounding model. Vision and spatial workloads are prominent examples;
sequence modeling and deployment remain supported use cases.

Lead with composable experiments, explicit coordinates, variable-length inputs, and
reproducible configuration. Support architectural claims with executable examples.
Measure performance before advertising speedups. Describe any paper usage as a dependency
relationship unless the paper establishes a broader contribution.

## Foundation implemented

- Research-oriented README with a runnable first encoder and vision/spatial entry points.
- Separate API reference, research guide, compatibility scope, and contribution instructions.
- Current tensor API in runnable examples and software citation metadata.
- Package discovery metadata and a configured Python 3.11/3.12 CPU quality-gate workflow.
- Public GitHub description and topics, plus bug-report, feature-request, and PR templates.
- An [upload-ready social preview](assets/social-preview.md) and README status badges.

CI results must be checked after pushing. Local checks do not establish hosted CI success
or GPU compatibility.

## Next: evidence before promotion

1. Establish the actual supported PyTorch floor. Test the declared minimum separately from
   the current locked environment and either document path-specific requirements or update
   metadata based on results. Do not infer compatibility from an install resolver.
2. Add a controlled component-substitution experiment. Keep data, training budget, and
   evaluation fixed; publish full configurations, parameter counts, and multiple-seed results
   if making accuracy claims. The current subclassing example demonstrates mechanics only.
3. Build a reproducible padded/packed benchmark on representative research hardware. Include
   equal-length and skewed length distributions, CPU and CUDA where available, fp32/fp16/bf16
   as supported, exact versions, warmup, synchronization, latency distributions, throughput,
   and peak device memory. Separate preconverted execution from conversion-inclusive timing;
   report backend/fallback behavior and verify valid-token numerical parity first.
4. Freeze an exact dependency version for the consuming paper project. Retain its config,
   checkpoints, environment, and reproduction commands; test reconstruction in that project.
5. Decide when the public API is stable enough for a non-prerelease version using compatibility
   evidence. Do not remove the release-candidate label for presentation alone.

## Discovery and launch preparation

Published GitHub description:

> Composable PyTorch transformers for vision, spatial, and sequence research.

Published topics: `pytorch`, `transformers`, `computer-vision`, `point-cloud`,
`attention`, `research`, `deep-learning`.

Draft introduction:

> Stackformers is a PyTorch library for researchers who modify transformer architectures.
> Start with an encoder or decoder preset, then replace attention, positional encoding,
> normalization, or feedforward components. CPU examples cover image patches, continuous
> point coordinates, padded and packed batches, and custom components. Configuration and
> deployment guides document the boundaries and current limitations.

Publish an accompanying technical walkthrough showing one useful substitution, why it is
needed, the changed code/configuration, and what remains controlled. Use the benchmark report
for any efficiency claim. This introduction has not been posted externally. The description
and topics above are live; the social preview still requires upload in repository settings.

## CVPR-related use

Treat Stackformers as implementation infrastructure unless the research itself evaluates it.
During review, coordinate disclosure and publicity with the paper authors and the applicable
venue rules. Add a named project case study only after its dependency relationship can be
publicly attributed. Include a link to reproduction artifacts and the exact library version;
do not imply acceptance, endorsement, or performance validation from indirect usage.

## Evaluate progress

Check whether a new user can run a relevant example in five minutes, reconstruct a saved
model, identify compatibility limits, and make a component substitution. Track recurring
onboarding questions, successful external reproductions, and substantive adoption reports.
Stars and traffic can show reach; they do not establish research usefulness.
