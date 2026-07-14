# ncnn-omni model package

Status: proposed format version 1

## Purpose

A package is the deployable contract between offline conversion and the device
runtime. It binds converted ncnn graphs to a known model adapter, processors,
tensor I/O, capabilities, numerical constraints, and integrity metadata.

The runtime does not infer behavior from filenames and does not import Hugging
Face Python code on device.

## Recommended layout

```text
qwen3-asr-0.6b-ncnn/
├── manifest.json
├── modules/
│   ├── audio_encoder.param
│   ├── audio_encoder.bin
│   ├── text_embedding.param
│   ├── text_embedding.bin
│   ├── decoder.param
│   ├── decoder.bin
│   ├── lm_head.param
│   └── lm_head.bin
├── tokenizer/
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── processors/
│   └── audio.json
├── metadata/
│   ├── source.json
│   ├── conversion.json
│   ├── checksums.json
│   └── memory-profiles.json
└── tests/
    ├── parity.json
    └── fixtures/                 # optional small redistributable fixtures
```

Large assets may use platform package containers later, but their logical paths
and manifest contract remain the same.

## Manifest sketch

```json
{
  "manifest_version": 1,
  "package_id": "qwen3-asr-0.6b-ncnn",
  "model": {
    "family": "qwen3-asr",
    "adapter": "qwen3_asr",
    "source": "Qwen/Qwen3-ASR-0.6B"
  },
  "capabilities": {
    "tasks": ["transcription"],
    "inputs": ["audio"],
    "outputs": ["text"],
    "streaming_outputs": ["text"]
  },
  "modules": [
    {
      "id": "audio_encoder",
      "format": "ncnn",
      "param": "modules/audio_encoder.param",
      "weights": "modules/audio_encoder.bin",
      "inputs": [
        {"logical_name": "features", "blob": "in0", "dtype": "f32", "layout": "T,F"}
      ],
      "outputs": [
        {"logical_name": "audio_embeddings", "blob": "out0", "dtype": "f32", "layout": "T,H"}
      ],
      "execution": {
        "allowed_devices": ["cpu", "vulkan"],
        "preferred_device": "cpu",
        "residency": "session"
      }
    }
  ],
  "pipeline": {
    "adapter_config": "processors/audio.json"
  },
  "compatibility": {
    "ncnn_min_version": "TBD",
    "ncnn_omni_manifest_min": 1,
    "ncnn_omni_manifest_max": 1
  },
  "integrity": {
    "checksums": "metadata/checksums.json"
  }
}
```

Blob names above are examples only. The converted graph's actual names are stored
explicitly and resolved once during loading.

## Required sections

### Identity and compatibility

- `manifest_version`: package schema version;
- `package_id`: stable human-readable package identity;
- `model.family`: upstream family for diagnostics;
- `model.adapter`: built-in ncnn-omni adapter identifier;
- compatible ncnn-omni manifest range;
- minimum tested ncnn version and conversion tool versions.

### Capabilities and limits

- supported tasks and input/output modalities;
- streaming input/output capabilities;
- maximum context, media count, resolution, duration, channels, and sample rates;
- required/optional modalities and valid combinations.

### Modules

Each module declares:

- ncnn parameter and weight paths;
- logical ports mapped to concrete blob names;
- dtype, layout, rank, and dynamic axes;
- persistent state ports, if any;
- allowed numerical modes and devices;
- residency policy and exclusive groups;
- optional expected memory profile.

The manifest does not encode arbitrary execution code. `model.adapter` interprets
these modules and builds a known typed pipeline.

### Processor and generation configuration

Configuration may include:

- tokenizer and chat template resources;
- audio sample rate, feature/window settings, padding rules;
- image normalization, resize, patch and token budget rules;
- special token ids and stop tokens;
- RoPE/mRoPE position-planning values;
- sampling defaults and safe limits;
- audio codebook and codec framing parameters.

These values are validated by the selected adapter. Unknown required fields fail
loading; optional extension fields are namespaced.

## Versioning rules

- Adding an optional field is backward compatible.
- Changing field meaning, required structure, or ownership semantics requires a
  new `manifest_version`.
- Adapter-specific config has its own `config_version`.
- Runtime rejects packages newer than its supported range before loading weights.
- Conversion metadata records source revision, export script revision, pnnx/ncnn
  versions, precision, quantization, and known parity results.

## Integrity and safety

Before loading any ncnn graph, the package loader:

1. parses JSON with depth and size limits;
2. rejects absolute paths and `..` traversal;
3. verifies every required file exists and is a regular allowed asset;
4. verifies declared size/checksum where available;
5. enforces package and per-file size limits;
6. validates adapter and manifest compatibility;
7. validates port uniqueness and module references;
8. computes an admission estimate before allocating large buffers.

Packages cannot load arbitrary shared libraries or execute scripts. Mobile builds
only use statically registered adapters.

## Packaging policy

- Keep ncnn graphs modular where it enables safe residency or independent parity
  testing; do not split every trivial operator into a separate graph.
- Fusion decisions are conversion-time and documented in `conversion.json`.
- Tokenizer/config assets are package-local to make inference reproducible.
- Parity metadata distinguishes CPU fp32, CPU fp16, Vulkan fp16, and quantized
  results rather than claiming one universal tolerance.
- Checksums ensure integrity, not model authenticity. Signed package indexes may
  be added later above this format.

## Open questions to resolve during the ASR vertical slice

- Exact ncnn version identifier to store: release tag, commit, or feature level.
- Portable representation of dynamic axes and layout semantics.
- Whether very large `.bin` files should be chunked for mobile asset systems.
- Which ncnn loading mode gives the best mmap/copy behavior on each platform.
- How to declare device-resident cross-module handoff once experimentally proven.
- Whether tokenizer resources should stay in upstream JSON or be compiled into a
  compact ncnn-omni format after correctness is established.
