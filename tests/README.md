# Tests

- `unit/`: manifests, typed ports, processors, queues, state and samplers;
- `integration/`: package load, pipeline lifecycle, cancellation, resource policy;
- `parity/`: PyTorch-versus-ncnn module and end-to-end numerical comparisons;
- `fixtures/`: small deterministic inputs and reference metadata.

Parity reports separate preprocessing, encoder/projector, prefill, single-step
decode, accumulated generation, and final normalized output. CPU/Vulkan and each
precision mode have independent tolerances.
