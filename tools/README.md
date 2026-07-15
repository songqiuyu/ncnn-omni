# Validation tools

- `parity/qwen3_asr_frontend_dump.cpp` exports normalized PCM and the complete
  C++ Log-Mel tensor from the production frontend implementation.
- `parity/qwen3_asr_frontend_parity.py` compares that output with Qwen3-ASR's
  preprocessing path and writes a machine-readable report.
- `parity/qwen3_asr_e2e.py` compares every generated greedy token with the
  Transformers reference.
- `parity/qwen3_asr_reference.py` contains the small compatibility shim shared
  by the Python reference tools.

Python, PyTorch, and Transformers are development-only dependencies. The C++
runtime and CLI do not depend on them.
