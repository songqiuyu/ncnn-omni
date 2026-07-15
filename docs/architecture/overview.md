# ncnn-omni architecture

Status: design baseline, pre-implementation

## 1. Mission

ncnn-omni is a lightweight, cross-platform C++ runtime for local multimodal model
inference. It uses ncnn as its only tensor graph backend and provides the dynamic
orchestration that static ncnn subgraphs intentionally do not provide.

Initial architecture-validation models are:

- Qwen3-ASR: audio input → text output;
- one Qwen VLM: image/video input → text output;
- Qwen3-TTS: text/reference audio input → streaming audio output.

Supporting all three without changing the core contracts is the acceptance test
for the architecture. Model support is not considered extensible if each new
family adds modality enums or branches to a central LLM class.

## 2. Goals and non-goals

### Goals

- Run fully offline on Android, iOS/macOS, Windows, Linux, and suitable embedded
  devices supported by ncnn.
- Represent text, image, audio, and video inputs and text, audio, token, tensor,
  and metrics outputs.
- Support batch and streaming pipelines, cancellation, and bounded backpressure.
- Make model-specific code local to a model adapter.
- Keep ncnn module execution, pipeline orchestration, session state, and public
  task APIs separate.
- Make memory use and CPU/Vulkan placement explicit and measurable.
- Provide module-level and end-to-end PyTorch parity tests.
- Keep the device runtime independent from Python, PyTorch, ONNX, and pnnx.

### Non-goals for v1

- Distributed execution, tensor parallelism, or server-scale continuous batching.
- A general ML compiler or an alternative to pnnx/ncnnoptimize.
- A user-programmable graph language.
- Loading arbitrary native plugins on mobile.
- Hiding every model behind one universal `forward()` or `generate()` method.
- Promise of zero-copy transfer across independently loaded ncnn networks before
  it is validated per backend.

## 3. Design principles

1. **Composition over model-type inheritance.** A model adapter assembles stages
   and modules; the core never switches on `Qwen3ASR`, `VLM`, or `TTS`.
2. **Typed boundaries.** Media, tensor ports, session state, and output events have
   explicit types and ownership.
3. **Static compute, dynamic orchestration.** ncnn owns bounded tensor graphs;
   C++ owns loops, branching, streaming, state, and lifecycle.
4. **One process first.** Phones are the baseline. Concurrency is optional and
   bounded; no IPC is required for a normal inference.
5. **Memory before throughput.** A request is admitted only if its estimated peak
   fits the configured budget.
6. **Streaming is a core contract.** It is not simulated by buffering the entire
   output and calling a final callback.
7. **Backend decisions are observable.** CPU/Vulkan placement, transfer time,
   allocator peaks, fallbacks, and numerical mode appear in metrics.
8. **Errors are values.** Public C/C++ boundaries use status/result objects; no
   exception crosses the C ABI or platform binding.
9. **Manifest is declarative; behavior is code.** The package declares artifacts
   and selects a known adapter. Complex decode feedback stays in tested C++.

## 4. System layers

```text
Application / CLI / Android / Apple / Python
                       |
                Convenience task APIs
       Transcriber | VisionChat | SpeechSynthesizer
                       |
                 Engine + Session API
         requests, events, cancellation, metrics
                       |
            Pipeline Engine / Coordinator
       lifecycle, routing, bounded queues, fan-in/out
                       |
    +------------------+---------------------+
    |                  |                     |
Processor stages   Generation stages     Utility stages
text/image/audio   text AR/audio AR      mixer/transfer/sink
    |                  |                     |
    +------------------+---------------------+
                       |
               Module / Tensor runtime
        NcnnModule, TensorHandle, StateStore
                       |
            Device and Resource Manager
      ncnn Net/Extractor, CPU/Vulkan, allocators
                       |
                     ncnn
```

### 4.1 Public API layer

The public API deals in application data, never raw ncnn objects.

Conceptual request types:

```cpp
using InputPart = std::variant<TextInput, ImageInput, AudioInput, VideoInput>;

struct GenerateRequest {
    std::vector<InputPart> parts;   // preserves interleaving/order
    GenerationOptions generation;
    RequestOptions runtime;
};
```

Conceptual output events:

```cpp
using OutputEvent = std::variant<
    TextDelta,
    AudioChunk,
    TokenChunk,
    TensorResult,
    MetricsSnapshot,
    Completed,
    ErrorEvent>;
```

Rules:

- Input order is preserved because omni prompts may interleave modalities.
- `AudioChunk` includes sample rate, channel count, sample format, sequence number,
  and final flag.
- Events are immutable once emitted.
- A request has one terminal event: `Completed` or `ErrorEvent`.
- Task APIs are thin adapters over the event/session core, not separate runtimes.

### 4.2 Engine and Session

`Engine` owns process-level services:

- device discovery and capability probing;
- package registry and validation;
- loaded model/pipeline instances;
- shared read-only resources and bounded caches;
- scheduler factory and metrics sink.

`Session` owns all mutable inference state for one logical interaction:

- text KV caches and positions;
- cross-attention or multimodal caches;
- streaming encoder windows;
- audio generator/codebook state;
- vocoder overlap/history;
- conversation and prompt state;
- cancellation token and request-local allocator handles.

A loaded model may create multiple sessions. No mutable request state is stored in
the model adapter, `ncnn::Net`, or a shared `ncnn::Extractor`.

Conceptual lifecycle:

```cpp
auto model = engine.load_package(path, load_options);
auto session = model.create_session(session_options);
auto request_id = session.submit(request);
while (auto event = session.next_event(request_id)) { /* consume */ }
session.cancel(request_id);
```

The first implementation may execute synchronously internally, but the API must
not prevent a later bounded asynchronous executor.

### 4.3 Model adapter and registry

A model adapter is selected by the package's stable adapter identifier, for
example `qwen3_asr`, not by filename guessing.

```cpp
class ModelAdapter {
public:
    virtual Result<PipelineDefinition> build(
        const ModelManifest& manifest,
        RuntimeServices& services) = 0;
    virtual CapabilitySet capabilities() const = 0;
};
```

Registry rules:

- Built-in adapters register statically for mobile-friendly linking.
- Registration maps `(adapter_id, supported_manifest_range)` to a factory.
- Package data cannot name arbitrary native libraries.
- Missing adapters or incompatible manifest versions fail before loading weights.
- All model-family prompt rules, special token placement, position planning, and
  feedback behavior stay inside its adapter directory.

### 4.4 Pipeline, Stage, and Module

These are intentionally separate concepts.

**Pipeline** is a typed topology plus lifecycle policy.

**Stage** is a schedulable unit that consumes messages, owns control flow, and may
call zero or more modules.

**Module** is a bounded computation, usually one ncnn `.param/.bin` pair.

```cpp
class Module {
public:
    virtual Status load(DeviceContext&, const ModuleSpec&) = 0;
    virtual Result<TensorMap> run(
        const TensorMap& inputs,
        ModuleState* state,
        RunContext& context) = 0;
    virtual void unload() = 0;
};

class Stage {
public:
    virtual Status open(SessionContext&) = 0;
    virtual Status accept(const StageMessage&, StageEmitter&) = 0;
    virtual Status flush(StageEmitter&) = 0;
    virtual void cancel() = 0;
    virtual void close() = 0;
};
```

Required initial stage categories:

- `ProcessorStage`: tokenizer, image transform, audio resample/log-mel, video
  sampling, prompt construction;
- `NcnnStage`: one-shot ncnn encoder/projector/codec invocation;
- `EmbeddingMixerStage`: inserts or combines modality and text embeddings;
- `TextARStage`: prefill/decode loop, logits processing, sampling, detokenization;
- `AudioARStage`: multi-codebook or feedback autoregressive generation;
- `StreamingCodecStage`: code chunks to waveform with persistent overlap state;
- `TransferStage`: explicit host/device transition when unavoidable;
- `SinkStage`: converts internal results to public output events.

Not every stage is a thread. The serial scheduler calls them directly. Stage is an
ownership and scheduling boundary, not a mandate for concurrency.

### 4.5 Messages, ports, and backpressure

Pipeline edges carry one of:

- `Payload`: a request-scoped immutable value;
- `StreamChunk`: ordered chunk with sequence number;
- `StreamEnd`: explicit normal end;
- `Failure`: error with stage context;
- `Cancel`: cooperative cancellation.

Each port declares its type, streaming mode, and optional shape constraints.
Pipeline construction validates connections before any model runs.

Every streaming edge has a bounded queue. When a TTS vocoder is slower than the
audio-token generator, the producer pauses at a safe step instead of growing
memory indefinitely. Cancellation closes all downstream queues and releases
request-owned state exactly once.

### 4.6 Generation engines

Generation is a policy family rather than one text-specific loop.

#### Text autoregressive engine

Responsibilities:

- chunked/full prefill according to memory policy;
- single-token or supported multi-token decode;
- KV/state updates;
- logits processors and sampler pipeline;
- stop criteria and incremental detokenization;
- text/token events.

#### Audio autoregressive engine

Responsibilities:

- one or multiple codebooks per step;
- model-specific feedback tensors;
- codebook sampling and stop policy;
- chunk formation for a downstream codec;
- code/token timing events.

#### Non-AR engines

Diffusion or flow models may be added later as separate stage/engine families.
They must not be forced through a fake text-token API.

## 5. ncnn runtime integration

### 5.1 NcnnModule ownership

`NcnnModule` owns:

- one `ncnn::Net` or a tightly coupled set explicitly declared by the adapter;
- resolved input/output blob bindings;
- load-time `ncnn::Option`;
- module capability and memory profile;
- weight residency state.

For each invocation it creates a fresh `ncnn::Extractor`, binds inputs, extracts
declared outputs, checks every return code, and destroys the extractor after the
run. A future optimized path may pool invocation objects only after thread-safety
and state-reset tests exist.

### 5.2 Tensor boundary

Internal modules exchange `TensorHandle`, which records:

- logical dtype and shape;
- layout/semantic axes;
- host or device residence;
- ownership and lifetime;
- underlying ncnn representation.

`ncnn::Mat`/`VkMat` stay inside the runtime implementation. v1 may materialize a
host `ncnn::Mat` between separately loaded networks where required. Device-resident
handoff is an optimization, not an undocumented assumption; any synchronization
or copy must be visible in stage metrics.

### 5.3 Load-time options

Backend, precision, packing, thread count, and allocator choices are resolved per
module before `load_param/load_model`. A module cannot silently mutate these
options for an active session.

The resolver considers:

- package constraints and numerical support;
- detected CPU/Vulkan capabilities;
- user policy (`latency`, `memory`, `quality`, `battery`);
- measured/device-profile overrides;
- current memory budget.

Unsupported Vulkan configurations fall back only when the policy allows it. The
fallback is reported; correctness tests run separately for CPU and Vulkan.

### 5.4 Allocators and memory budgets

Memory categories are accounted separately:

```text
resident weights
module pipelines / Vulkan resources
temporary workspace
blob activations
session state / KV cache
processed media cache
raw media buffers
stream queues
```

`ResourceManager` owns budgets and residency decisions. Allocator objects are
scoped to a device/worker or session according to ncnn's safety requirements;
they are not casually shared across concurrent sessions.

Models declare estimated memory profiles in the package. Runtime measurement
updates metrics but never treats a stale estimate as a safety guarantee.

### 5.5 Module residency

Initial policies:

- `resident`: keep loaded for the model lifetime;
- `session`: load at session creation and unload at close;
- `on_demand`: load before a stage and evict at a safe boundary;
- `exclusive_group`: modules may not be resident together under a small budget.

For example, a phone TTS pipeline may retain the AR generator while loading the
vocoder on demand; a desktop may keep both resident and overlap them.

## 6. Scheduling profiles

### 6.1 Mobile serial profile — v1 default

One request advances through the pipeline in-process. Streaming stages alternate
cooperatively. This minimizes memory, locks, and driver concurrency issues.

### 6.2 Mobile bounded async profile

Optional workers overlap safe operations such as:

- CPU media preprocessing with ncnn module preparation;
- audio-token generation with chunked vocoding;
- output encoding/playback with the next inference step.

All queues are bounded, and device modules may remain serialized if a driver or
memory profile requires it.

### 6.3 Desktop multi-session profile

A request queue schedules multiple sessions using memory-aware admission. It may
batch compatible processor/encoder calls later, but no core interface assumes
continuous batching. Prefill/decode batching is only added after converted ncnn
graphs and cache layouts provide a real benefit.

### 6.4 Cancellation and shutdown

Cancellation is cooperative at stage boundaries and inside AR loops. The runtime
guarantees:

- no new downstream work after cancellation is observed;
- exactly one terminal event;
- request-owned queues and state are released;
- shared model state remains valid for other sessions;
- partially generated text/audio may be marked incomplete but is not retracted.

## 7. Reference pipelines

### 7.1 Qwen3-ASR

```text
AudioInput
  -> AudioProcessor (decode/resample/features)
  -> AudioEncoder [ncnn]
  -> Projector [ncnn or fused]
  -> Prompt/TextProcessor
  -> TextEmbedding [ncnn]
  -> EmbeddingMixer + PositionPlanner
  -> TextAR (Decoder + LM head [ncnn])
  -> TextDelta / Completed
```

ASR-specific details—audio framing, special tokens, timestamp behavior, and
position rules—belong to `src/models/qwen3_asr`.

### 7.2 VLM

```text
Image/VideoInput
  -> Image/VideoProcessor
  -> VisionEncoder [ncnn]
  -> Projector/Merger [ncnn]
  -> Prompt/TextProcessor
  -> EmbeddingMixer + multimodal position plan
  -> TextAR
  -> TextDelta / Completed
```

Multi-image and video are ordered collections of media items, not new overloads.
The processor decides token budgets and frame sampling under request limits.

### 7.3 Qwen3-TTS

```text
TextInput + optional ReferenceAudio
  -> Text/Reference Processor
  -> optional ReferenceAudioEncoder [ncnn]
  -> AudioAR / multi-codebook generator [ncnn loop]
  -> bounded CodeChunk stream
  -> StreamingCodec/Vocoder [ncnn]
  -> AudioChunk stream / Completed
```

This pipeline is deliberately not expressed as `LLM -> string`. Audio generation
and vocoding can retain independent state and residency policies.

## 8. Public capability model

Capabilities are structured, not a single `is_multimodal` flag:

```text
input modalities: text, image, audio, video
output modalities: text, audio, tensor
tasks: transcription, chat, vision_chat, speech_synthesis
streaming: input_audio, output_text, output_audio
limits: context, media count, duration, resolution, channels
backends: cpu, vulkan (per module/profile)
numerics: fp32, fp16 storage/arithmetic, int8, model-specific weight quantization
```

Applications can reject unsupported requests before allocating large buffers.

## 9. Error and observability model

`Status` contains a stable code, message, component/stage id, optional ncnn return
code, and causal context. Initial code groups:

- invalid request/package/manifest;
- unsupported capability/backend/numerics;
- model artifact missing or corrupt;
- out of memory/resource admission denied;
- preprocessing/module/generation failure;
- cancelled/timeout;
- internal contract violation.

Per-request metrics include:

- processor, encoder, prefill, decode, codec, and transfer latency;
- TTFT, tokens/s, first-audio latency, RTF;
- input/output token and media sizes;
- peak memory by category where measurable;
- module device/precision and fallbacks;
- cache hit/miss and queue stalls.

Metrics collection can be disabled or sampled. Raw user media and prompt contents
are never logged by default.

## 10. Threading and ownership invariants

- A `Session` is single-writer; its public handle may be called from multiple
  threads only through the engine's serialized request queue.
- Loaded module weights are read-only after successful load.
- Each active invocation owns its extractor and temporary outputs.
- Stage messages are immutable after emission.
- Shared cache entries are immutable and reference counted.
- A resource is destroyed on the same device/runtime context required by ncnn.
- Callback code never runs while holding internal scheduler or allocator locks.

These invariants must be enforced by tests before enabling multi-session mode.

## 11. Source dependency rules

Allowed dependency direction:

```text
bindings/examples -> public api
public api        -> core
models            -> pipeline + processors + generation + runtime interfaces
pipeline          -> core + runtime interfaces
processors        -> core
generation        -> core + runtime interfaces
runtime/ncnn      -> core + ncnn
platform          -> core
```

Forbidden:

- `core` depending on ncnn, a model family, or platform UI code;
- generic processors switching on model family;
- `runtime/ncnn` parsing model-specific prompt/config fields;
- bindings reaching into `src/models` or raw ncnn;
- one model adapter including another model adapter's private headers.

## 12. Target repository layout

```text
ncnn-omni/
├── CMakeLists.txt                 # added with the first compilable core
├── include/ncnn_omni/
│   ├── api/                       # stable public C++ API
│   ├── core/                      # Status, media values, events, capabilities
│   └── runtime/                   # module/tensor/device abstract contracts
├── src/
│   ├── core/                      # engine/session/resource implementation
│   ├── runtime/ncnn/              # only layer that directly uses ncnn execution
│   ├── pipeline/                  # topology, stages, queues, schedulers
│   ├── processors/{text,image,audio}/
│   ├── generation/{text,audio}/
│   ├── models/{qwen3_asr,qwen_vl,qwen3_tts}/
│   └── platform/                  # filesystem, mmap, clocks, device probes
├── bindings/{c,android,apple,python}/
├── tools/{convert,inspect,benchmark}/
├── examples/{asr,vlm,tts}/
├── tests/{unit,integration,parity,fixtures}/
├── docs/{architecture,research,adr}/
└── third_party/
```

This is a target decomposition, not a requirement to keep placeholder folders.
The repository creates each directory only when a working vertical slice needs
it; until then the boundary remains a documented design rather than empty files.

## 13. Architecture acceptance tests

The core design is accepted only when:

1. Qwen3-ASR CPU inference runs through public API → pipeline → ncnn modules with
   module-level parity artifacts.
2. A VLM adapter is added without changing core modality enums or adding a VLM
   branch to the engine.
3. Qwen3-TTS streams audio through a bounded code-to-wave edge without changing
   the text generation contract.
4. CPU and Vulkan placement are selected per module and reported.
5. Cancellation frees every session-owned state in ASR prefill, text decode, audio
   generation, and vocoder phases.
6. A constrained-memory profile can evict/load modules at safe stage boundaries.
7. The same model package is validated by desktop CLI and at least one mobile
   binding without application-side knowledge of prompt special tokens.

Until these pass, public headers remain versioned as experimental.
