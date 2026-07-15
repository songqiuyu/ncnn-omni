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
- [Qwen3-ASR phase 1 plan](plans/qwen3-asr-phase1.md): concrete implementation
  scope, five-model wiring, milestones, risks, and parity acceptance criteria.
- [Qwen3-ASR first-run guide](guides/qwen3-asr-first-run.md): build, runtime
  resources, CLI usage, current constraints, and parity commands.
- [Qwen3-ASR long-prefill divergence](diagnostics/qwen3-asr-long-prefill-divergence.md):
  falsification experiments, confirmed root causes, fixes, and regression
  evidence for long-audio token divergence.
- [Qwen3-ASR audio frontend parity](diagnostics/qwen3-asr-audio-frontend-parity.md):
  normalized PCM, partial-hop compatibility policy, full Log-Mel tensor
  metrics, and reproduction command.
- [ADR-0001](adr/0001-compositional-staged-pipeline.md): why ncnn-omni uses a
  compositional staged pipeline instead of a monolithic model class.

## Status

The first Qwen3-ASR CPU FP32 vertical slice is implemented. Interface names remain
proposals, not an ABI promise. The first ABI should only be frozen after
Qwen3-ASR, one VLM, and Qwen3-TTS have validated the shared abstractions.
