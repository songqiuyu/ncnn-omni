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
KV-cache decoding, and token-level parity tooling. The API remains experimental.

- [Design document index](docs/README.md)
- [Architecture overview](docs/architecture/overview.md)
- [Inference engine survey](docs/research/inference-engine-survey.md)
- [Model package proposal](docs/architecture/model-package.md)
- [Implementation roadmap](docs/roadmap.md)
- [Qwen3-ASR first-run guide](docs/guides/qwen3-asr-first-run.md)

## Proposed structure

```text
include/ncnn_omni/   public C++ contracts
src/core/            engine, session, resources
src/runtime/ncnn/    ncnn-only execution adapter
src/pipeline/        stages, topology, queues, schedulers
src/processors/      text, image, and audio processing
src/generation/      text and audio generation engines
src/models/          model-family adapters
bindings/            C and platform-language bindings
tools/               conversion, inspection, benchmarking
tests/               unit, integration, and parity tests
```

Implementation starts with a CPU Qwen3-ASR vertical slice. Public interfaces stay
experimental until ASR, VLM, and streaming TTS validate the shared abstractions.

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
