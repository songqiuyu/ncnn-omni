# Source layout

| Directory | Responsibility |
|---|---|
| `core/` | Engine/session lifecycle, package registry, capabilities, resources, metrics |
| `runtime/ncnn/` | ncnn module loading and invocation, tensor bridge, allocators, device policy |
| `pipeline/` | Typed topology, stage lifecycle, queues, serial/async schedulers |
| `processors/` | Reusable text/image/audio preprocessing primitives |
| `generation/` | Text AR, audio AR, sampler, detokenizer, streaming codec control |
| `models/` | Model-local pipeline assembly and special behavior |
| `platform/` | Filesystem/assets, mmap, clocks, threads, and device probing |

Dependency rules are normative in
[`docs/architecture/overview.md`](../docs/architecture/overview.md#11-source-dependency-rules).
