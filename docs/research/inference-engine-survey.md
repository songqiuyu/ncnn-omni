# Multimodal inference engine survey

Last reviewed: 2026-07-14

## Purpose

This survey is not a feature checklist. It extracts architecture patterns useful
to a lightweight, local, ncnn-based runtime. Server-oriented optimizations are
only adopted when they also make sense for a single phone or PC process.

## Systems reviewed

| System | Primary setting | Useful architecture ideas | What ncnn-omni should not copy directly |
|---|---|---|---|
| vLLM | High-throughput serving | Model-dispatched multimodal processor registry; placeholder ranges; profiling inputs; processor/embedding caching | Multi-process API/engine/worker split and server-sized cache defaults |
| SGLang / SGLang-Omni | Distributed LLM and omni serving | Coordinator, typed stages, different schedulers for AR/non-AR/streaming work, model-local adapters, stream fan-out/fan-in | ZMQ/NCCL relay fabric, distributed placement, and a scheduler process per stage |
| TensorRT-LLM | NVIDIA serving | Explicit processor → encoder → embedding integration; asynchronous request executor; overlap CPU preprocessing with GPU encoding | CUDA/TensorRT-specific engine building and in-flight batching as a baseline requirement |
| llama.cpp `libmtmd` | Local CPU/GPU inference | Small C/C++ surface; multimodal projector isolated from the text model; capability discovery; separate model artifacts | Treating all multimodality as input-to-text; it does not generalize by itself to audio output or diffusion |
| MLC LLM | Cross-platform compiled runtime | Separate model compilation/artifacts from a shared runtime; one engine API across native language bindings; local memory modes | TVM/JIT compiler architecture, which would duplicate ncnn/pnnx and greatly enlarge the project |
| ExecuTorch | Edge model execution | Clear preparation/runtime/execution phases; small runtime; backend delegation and explicit customization boundaries | Recreating a general graph compiler or delegate ecosystem above ncnn |
| LiteRT-LM | Product-grade edge LLM | A pipeline layer that stitches multiple runtime models with processors; packaged models with metadata; C++ core plus Kotlin/Swift/etc. bindings | LiteRT-specific compiled model APIs and NPU delegation machinery |
| MNN-LLM | Mobile/PC local inference | Evidence that multimodal, ASR, TTS and diffusion need separate orchestration above a mobile tensor runtime; async token-to-wave pipelines; scoped runtime resources | Model-family branching in a single LLM hierarchy or backend-specific APIs leaking to applications |

## Findings worth adopting

### 1. Multimodal processing is a registered, model-specific concern

vLLM dispatches preprocessing by model using a multimodal registry, lazily creates
processors, and separates processed media caching from model execution. This is a
better boundary than embedding image/audio preprocessing inside a generic LLM
class. See the official [vLLM multimodal registry](https://docs.vllm.ai/en/stable/api/vllm/multimodal/registry/)
and [multimodal cache documentation](https://docs.vllm.ai/en/latest/configuration/optimization/#multi-modal-caching).

Adoption for ncnn-omni:

- Processors are registered by model adapter and modality.
- Raw-media cache, processed-feature cache, and KV cache are separate budgets.
- Cache defaults are small or disabled on phones and must be bounded.
- A processor declares worst-case shapes so loading can estimate memory.

### 2. Omni workloads need different execution loops

SGLang-Omni separates non-autoregressive stages, autoregressive stages, and a
streaming code-to-wave scheduler. Its model-specific code stays under a model
directory while pipeline/coordinator concerns stay framework-owned. See its
[architecture](https://sgl-project.github.io/sglang-omni/developer_reference/main.html),
[pipeline design](https://sgl-project.github.io/sglang-omni/developer_reference/pipeline.html),
and [TTS integration guide](https://sgl-project.github.io/sglang-omni/developer_reference/tts_model_integration.html).

Adoption for ncnn-omni:

- `Stage` and `Module` are different concepts: a stage owns control flow; a module
  owns a bounded computation.
- Text AR generation, audio multi-codebook generation, and a vocoder do not share
  one forced `generate()` implementation.
- Pipeline topology is model-local, while lifecycle, cancellation, streaming, and
  resource ownership are framework-owned.
- Unlike SGLang-Omni, the initial runtime stays in-process. A local queue replaces
  distributed control/data planes.

### 3. Processor, encoder, and fusion are first-class steps

TensorRT-LLM explicitly describes multimodal inference as input processor,
multimodal encoder, and integration with the LLM decoder. It also overlaps CPU
preprocessing with GPU encoding and hashes raw media for reuse. See the official
[TensorRT-LLM multimodal design](https://nvidia.github.io/TensorRT-LLM/features/multi-modality.html).

Adoption for ncnn-omni:

- The pipeline must expose processor, encoder/projector, and embedding fusion as
  separately measurable stages.
- CPU work may run ahead of ncnn GPU work, but only under a bounded queue.
- Media hashing is optional and never retains unbounded raw user content.

### 4. Edge engines package artifacts separately from orchestration

MLC compiles model logic for a target while using a common runtime and API across
desktop and mobile. LiteRT-LM explicitly positions itself as an orchestration
layer that stitches processors, vision encoders, and text decoders over LiteRT.
See [MLC's runtime architecture](https://blog.mlc.ai/2024/06/07/universal-LLM-deployment-engine-with-ML-compilation)
and the [LiteRT-LM repository](https://github.com/google-ai-edge/LiteRT-LM).

Adoption for ncnn-omni:

- Conversion is an offline toolchain; the device runtime never depends on PyTorch
  or pnnx.
- A versioned package binds ncnn `.param/.bin` artifacts to processors, I/O names,
  capabilities, and a model adapter.
- The runtime core is C++; stable bindings build above a narrow C API.
- ncnn-omni does not add a compiler. pnnx/ncnnoptimize remain the graph toolchain.

### 5. Local runtimes benefit from a small, capability-aware API

llama.cpp isolates multimodal handling in `libmtmd`, ships projector artifacts
separately, and exposes model capabilities to callers. See the official
[llama.cpp multimodal documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md).

Adoption for ncnn-omni:

- Applications query capabilities before constructing a request.
- Convenience task APIs sit above one event/session core.
- Model-specific prompt and media rules do not leak into application code.

### 6. Edge runtime boundaries should be layered and replaceable

ExecuTorch argues that one monolithic solution is unsuitable for devices with
diverse hardware and power limits, and separates program preparation, runtime
preparation, and execution. See its official
[architecture overview](https://docs.pytorch.org/executorch/stable/getting-started-architecture).

Adoption for ncnn-omni:

- Offline conversion, package validation, runtime loading, and request execution
  are separate phases.
- ncnn is the only tensor graph backend in v1, but its adapter boundary is explicit
  so platform code and model code do not depend on raw `ncnn::Net`.
- Extensibility is achieved through processors, model adapters, and generation
  policies, not through a general-purpose plugin ABI in the first release.

## ncnn constraints that change the design

ncnn is a compact C++ inference library with CPU/mobile optimizations, Vulkan,
explicit blob/workspace allocators, multi-input/output graphs, fp16/int8 paths,
custom layers, and direct memory-backed model loading. These properties are a good
fit for an endpoint runtime; see the official [ncnn project](https://github.com/Tencent/ncnn).

They also imply several boundaries:

1. **ncnn executes subgraphs, not the whole product workflow.** Token loops, media
   framing, dynamic branching, streaming, and state ownership belong in C++.
2. **`ncnn::Net` configuration is load-time state.** Backend/precision policy is
   resolved before loading each module; it is not toggled per token.
3. **`ncnn::Extractor` is invocation state.** A loaded module can create an
   extractor per run; request state must not be stored in a shared extractor.
4. **CPU/Vulkan transitions are not free.** Device placement is per module and
   transfer points must remain visible in profiling.
5. **Vulkan is a capability, not a guarantee of speed.** Unsupported paths may
   fall back to CPU and mobile CPUs can be faster for small stages; see the
   official [ncnn Vulkan FAQ](https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-vulkan).
6. **Memory is the first scheduling constraint.** Allocator pools, model residency,
   activation peaks, KV state, and media caches need independent budgets.

## Rejected directions

### Copy a server engine scheduler

Continuous batching, distributed prefill/decode separation, shared-memory IPC,
and worker processes solve server utilization. They add latency, memory, and
failure modes on a phone. ncnn-omni starts with an in-process serial scheduler and
an optional bounded async pipeline. PC multi-session scheduling is a later policy,
not a core assumption.

### Make every model a generic declarative DAG

A completely generic graph language would duplicate part of ncnn and still fail
to express model-specific decode feedback cleanly. The manifest declares modules
and a known pipeline adapter. The adapter builds a typed staged pipeline in C++.

### Define multimodality as encoder embeddings plus a text decoder

That works for ASR/VLM/OCR understanding, but not for multi-codebook TTS,
streaming vocoders, speech-output omni models, or future diffusion pipelines.
Generation engines and output events therefore support non-text results from day
one.

### Expose `ncnn::Mat` in the public task API

It couples applications and bindings to ncnn layouts and ownership. Public APIs
use text/image/audio/video values; internal runtime ports use `TensorHandle`, which
may wrap an `ncnn::Mat` or a device representation.

## Resulting direction

ncnn-omni should be a small in-process orchestration runtime over ncnn modules:

```text
typed request -> model adapter -> staged pipeline -> ncnn modules
      ^                |               |                |
      |          package manifest   session state   CPU/Vulkan
      +----------- typed output events / cancellation --------+
```

The detailed contract is defined in [Architecture overview](../architecture/overview.md).
