# ADR-0001: Compositional staged pipeline

- Status: accepted for initial implementation
- Date: 2026-07-14

## Context

The project must support audio-to-text, image/video-to-text, text/reference-audio
to audio, and future mixed-input/mixed-output models. ncnn provides efficient
static tensor graph execution but does not own token loops, media streaming,
model-specific feedback, or multi-module lifecycle.

A conventional LLM class with optional vision/audio branches works for encoder
embedding injection, but assumes one text autoregressive decoder and one output
type. Qwen3-TTS and speech-output omni models violate that assumption.

## Decision

Use a compositional, typed, in-process staged pipeline:

- ncnn graphs are represented by `Module` implementations;
- control-flow and streaming units implement `Stage`;
- a model-local `ModelAdapter` builds the pipeline from a versioned package;
- framework-owned `Engine` and `Session` manage lifecycle and mutable state;
- output is a typed event stream, not only text tokens;
- serial scheduling is the baseline, with bounded async and desktop multi-session
  schedulers added as policies.

## Consequences

Positive:

- ASR, VLM, TTS, and future generation engines can use different topologies.
- ncnn-specific objects stay below the runtime interface.
- streaming, cancellation, resource budgets, and observability are common.
- model-specific behavior has a clear home.
- mobile execution remains single-process and can stay serial.

Costs:

- More interfaces than a single demo-oriented model class.
- Typed ports and session-state ownership require careful validation.
- Cross-module tensors may initially incur host materialization until ncnn device
  handoff is proven.
- The project must resist turning the stage system into an over-general workflow
  engine.

## Rejected alternatives

1. **Monolithic `OmniModel` inheritance tree:** creates model/modality branching
   and conflates loading, processing, generation, and output.
2. **Fully declarative arbitrary DAG:** duplicates graph-runtime concerns and does
   not cleanly express decode feedback.
3. **Server-style process-per-stage architecture:** inappropriate overhead and
   memory cost for the phone baseline.
4. **One text generation API for all outputs:** cannot correctly represent
   multi-codebook audio generation, streaming codecs, or diffusion.

## Validation

This decision is revisited if Qwen3-ASR, one VLM, and Qwen3-TTS cannot be added
without model-family conditions in core or incompatible changes to the event,
session, module, and stage contracts.
