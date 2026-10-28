# GitHub social preview

`social-preview.jpg` is the upload-ready repository preview: 1774 × 887 pixels,
opaque JPEG, below GitHub's 1 MB limit. It was generated with the built-in image
generation tool and re-encoded as JPEG without resizing or changing its contents.

Upload it at [repository settings](https://github.com/Red-Eyed/stackformers/settings)
under **Social preview → Edit → Upload an image**. Neither the repository REST
update nor the GraphQL `UpdateRepositoryInput` exposes a social-image upload field.
See [GitHub's instructions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/customizing-your-repositorys-social-media-preview).

## Generation prompt

```text
Use case: ads-marketing.
Asset type: GitHub repository social preview for Stackformers, landscape 2:1, ideally 1280 by 640 pixels, opaque background, PNG suitable for upload under 1 MB.
Primary request: Create one polished, restrained editorial graphic for an open-source PyTorch transformer research library. The audience is computer vision and machine learning researchers.
Visual: an elegant architectural illustration of a few interchangeable stacked transformer layers, represented as precise thin panels with subtle token grids and small coordinate points, arranged as one coherent sculptural stack. Keep it abstract and clearly an illustration rather than a scientifically labeled architecture diagram. Use a clean off-white background, charcoal type and graphite panels with subtle dimensional shading. Lots of negative space, no decorative noise.
Typography and composition: very large crisp readable project name; smaller clear subtitle; one compact line naming application domains. All text within generous safe margins, readable at thumbnail size. Balance the type with the stack illustration, with no overlapping elements.
Exact text, verbatim:
"Stackformers"
"Composable PyTorch transformers for research."
"Vision · Spatial · Sequences"
Constraints: no other words, no invented logos or badges, no conference names, no CVPR affiliation, no performance claims, no code snippets, no watermark. Final artwork should be clean and high contrast.
```
