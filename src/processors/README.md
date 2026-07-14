# Processors

Reusable CPU-side processing primitives live here:

- `text/`: tokenization, chat templates, incremental detokenization helpers;
- `image/`: decode, resize, normalize, tiling/patch preparation;
- `audio/`: decode, channel conversion, resampling, feature extraction/framing.

Model-specific ordering, constants, special tokens, and position rules stay in a
model adapter. Processors must be deterministic and independently testable.
