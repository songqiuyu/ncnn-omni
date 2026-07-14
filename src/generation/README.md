# Generation engines

- `text/`: prefill/decode loop, KV state, logits processors, sampler pipeline,
  stopping, and incremental text/token events.
- `audio/`: multi-codebook/feedback generation, code chunking, and streaming codec
  orchestration.

Text and audio generation share lifecycle utilities but not one forced
`generate()` implementation.
