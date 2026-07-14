# Qwen3-ASR adapter

First vertical slice: audio processor/encoder/projector, embedding and position
assembly, text prefill/decode, and text output. All converted module boundaries
must have PyTorch parity fixtures before end-to-end claims are made.
