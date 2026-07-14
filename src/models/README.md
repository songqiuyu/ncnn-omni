# Model adapters

Each model family owns its pipeline assembly and model-specific behavior:

- `qwen3_asr/`: first CPU vertical slice;
- `qwen_vl/`: second architecture test for image/video-to-text;
- `qwen3_tts/`: third architecture test for streaming audio output.

Adapters may use public runtime, processor, generation, and pipeline contracts.
They must not add model-family branches to core. Shared behavior is promoted out
only after at least two adapters demonstrate the same contract.
