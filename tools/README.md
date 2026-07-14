# Tools

- `convert/`: offline export/package assembly helpers around pnnx/ncnn tooling;
- `inspect/`: manifest, capability, tensor-I/O, and memory-profile inspection;
- `benchmark/`: stage timing, TTFT, tokens/s, RTF, memory, and backend comparison.

Conversion tools may depend on Python/PyTorch/pnnx. The runtime library must not.
