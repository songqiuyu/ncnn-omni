# Initial implementation roadmap

The roadmap is organized as vertical slices. Each milestone must produce a
working end-to-end path and parity evidence; building many abstractions without a
model exercising them is explicitly avoided.

## M0 — Contracts and package inspection

- Define experimental `Status`, `Result`, media values, `OutputEvent`, capability,
  manifest, module, stage, engine, and session headers.
- Implement package parsing/validation and `ncnn-omni-inspect`.
- Implement device capability reporting without loading a model.
- Add schema/ownership/unit tests.

Exit gate: an ASR package is validated, its module I/O and estimated memory are
printed, and malformed/path-traversal packages are rejected.

## M1 — Qwen3-ASR CPU vertical slice

- Audio decode/resample/feature processor.
- `NcnnModule` load/run/unload with checked return codes.
- Text embedding, decoder/KV, LM head, sampler, incremental detokenizer.
- Serial pipeline and event API.
- Module-level and end-to-end PyTorch parity harness.

Exit gate: fixed fixtures match the agreed tolerances; long prefill and single
decode are reported separately; cancellation and repeated-session tests pass.

## M2 — Resource and Vulkan policy

- Per-module load options and capability resolution.
- CPU/Vulkan parity matrix and fallback metrics.
- Blob/workspace/session memory accounting.
- Residency policies and constrained-memory loading tests.
- Android and desktop smoke examples.

Exit gate: device policy is explicit, reproducible, and does not silently switch
numerical mode or backend.

## M3 — VLM architecture test

- Image processor and vision encoder/projector modules.
- Ordered multi-part request and embedding mixer.
- Model-specific multimodal position planner.
- Multi-image support; bounded video frame sampling may follow.

Exit gate: VLM is added as a model adapter without changes that introduce a VLM
branch or model-family enum in core.

## M4 — Qwen3-TTS architecture test

- Reference-audio processor/encoder where required.
- Audio multi-codebook generation engine.
- Bounded streaming edge and codec/vocoder stage.
- `AudioChunk` public events, first-audio latency and RTF metrics.
- Producer/consumer backpressure and mid-stream cancellation tests.

Exit gate: streaming audio output works without changing the text AR contract or
buffering the complete waveform.

## M5 — Stable API and platform bindings

- Freeze a narrow C ABI after the three architecture tests.
- Android JNI/Kotlin and Apple Swift/Objective-C wrappers.
- Optional Python binding for validation/tooling, not as a device dependency.
- Desktop multi-session admission and scheduler policy.
- Package compatibility matrix and release tooling.

## Deferred optimizations

These require profiles and correctness baselines before implementation:

- device-resident zero/low-copy handoff between ncnn modules;
- chunked prefill and prefix/media embedding caches;
- KV cache quantization/compression;
- speculative or multi-token decoding;
- compatible-request batching;
- memory-mapped/chunked weight packaging;
- diffusion/flow generation engines;
- NPU or non-ncnn tensor backends.
