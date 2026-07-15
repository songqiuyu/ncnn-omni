<p align="center">
  <img src="docs/banner.png" alt="ncnn-omni banner" width="100%">
</p>

[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE) [![Language](https://img.shields.io/badge/Language-简体中文-green)](README_CN.md) [![Model](https://img.shields.io/badge/ncnn_omni-HuggingFace-yellow)](https://huggingface.co/songqiuyu)

# ncnn-omni

ncnn-omni is an early-stage, lightweight C++ runtime for on-device multimodal
model inference using [ncnn](https://github.com/Tencent/ncnn) as the tensor graph
backend.

The project is designed for phones, PCs, and embedded devices. Its architecture
supports different model topologies instead of treating every multimodal model as
a vision extension of a text LLM. Initial architecture-validation targets are
Qwen3-ASR, a Qwen VLM, and Qwen3-TTS.

## Supported Models

| Model | Task | ncnn Model |
|---|---|---|
| Qwen3-ASR-0.6B | Automatic speech recognition | [Qwen3-ASR-0.6B-ncnn](https://huggingface.co/songqiuyu/Qwen3-ASR-0.6B-ncnn) |

More converted ncnn models and related artifacts are available on
[songqiuyu's Hugging Face profile](https://huggingface.co/songqiuyu).

## Current status

The first CPU FP32 Qwen3-ASR-0.6B vertical slice is runnable. It includes WAV
loading, Whisper-compatible Log-Mel extraction, all five ncnn modules, greedy
KV-cache decoding, full-frontend numerical parity, and exact generated-token
parity tooling. The API remains experimental.

- [Design document index](docs/README.md)
- [Architecture overview](docs/architecture/overview.md)
- [Inference engine survey](docs/research/inference-engine-survey.md)
- [Model package proposal](docs/architecture/model-package.md)
- [Implementation roadmap](docs/roadmap.md)
- [Qwen3-ASR first-run guide](docs/guides/qwen3-asr-first-run.md)
- [Audio frontend parity report](docs/diagnostics/qwen3-asr-audio-frontend-parity.md)
- [Long-audio root-cause report](docs/diagnostics/qwen3-asr-long-prefill-divergence.md)

## Highlights

- **Pure C++ deployment:** PyTorch and Transformers are used only by offline
  validation tools, never by the runtime.
- **Five verified ncnn graphs:** Audio Conv, Audio Transformer, token embedding,
  text decoder/KV cache, and LM head retain independent differential tests.
- **Correct graph-external semantics:** Whisper final-hop compatibility, biased
  Conv tail padding, CPU/SDPA global audio attention, prompt fusion, RoPE, and
  sentinel KV masking are explicit C++ logic.
- **Strict evidence:** normalized PCM is sample-exact, complete Log-Mel tensors
  stay within `2.23e-5` max error, and the current long-audio cases match every
  greedy token from Transformers.
- **Correctness-first and inspectable:** deterministic CPU FP32 is the baseline;
  backend and precision optimizations must establish their own parity results.
- **Lean hot paths:** Whisper DFT coefficients, RoPE frequencies, and audio
  positions are precomputed; Conv outputs are written directly into the Audio
  Transformer input without an intermediate tensor copy.

## Current structure

```text
include/ncnn_omni/   compact public Qwen3-ASR API and Result type
src/models/          Qwen3-ASR orchestration and generation loop
src/processors/      WAV, Qwen2 tokenizer, and Whisper Log-Mel
src/runtime/ncnn/    checked ncnn module loading and invocation
examples/asr/        runnable CLI
tools/parity/        frontend and end-to-end differential tools
tests/unit/          deterministic processor/tokenizer tests
docs/                architecture, research, guides, and evidence
```

Only implemented code is represented by physical directories. The planned
multimodal engine, VLM, TTS, bindings, and platform layers remain design work in
the architecture documents until real implementations justify those boundaries.

## Star History

<p align="center">
  <a href="https://www.star-history.com/songqiuyu/ncnn-omni">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=songqiuyu/ncnn-omni&type=Date&theme=dark" />
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=songqiuyu/ncnn-omni&type=Date" />
      <img alt="ncnn-omni Star History Chart" src="https://api.star-history.com/svg?repos=songqiuyu/ncnn-omni&type=Date" />
    </picture>
  </a>
</p>
