# ncnn-omni design documents

These documents define the architecture before implementation starts. The design
targets on-device multimodal inference with ncnn on phones, PCs, and embedded
devices.

## Start here

- [Inference engine survey](research/inference-engine-survey.md): what was learned
  from existing server and edge runtimes, and what is deliberately not copied.
- [Architecture overview](architecture/overview.md): goals, layers, interfaces,
  execution model, resource policy, and extension rules.
- [Model package format](architecture/model-package.md): the proposed versioned
  package and manifest contract.
- [Initial roadmap](roadmap.md): staged delivery and architecture validation gates.
- [ADR-0001](adr/0001-compositional-staged-pipeline.md): why ncnn-omni uses a
  compositional staged pipeline instead of a monolithic model class.

## Status

This is a pre-implementation design baseline. Interface names are proposals, not
an ABI promise. The first ABI should only be frozen after Qwen3-ASR, one VLM, and
Qwen3-TTS have validated the shared abstractions.
