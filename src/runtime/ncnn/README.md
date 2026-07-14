# ncnn runtime adapter

This is the only execution layer that directly owns and invokes `ncnn::Net` and
`ncnn::Extractor`.

Planned responsibilities:

- resolve per-module load options and CPU/Vulkan placement;
- load `.param/.bin` artifacts with checked errors;
- map logical tensor ports to ncnn blob names;
- create invocation-local extractors;
- bridge internal tensor handles and ncnn representations;
- scope blob/workspace allocators and report memory/transfer metrics;
- expose no model-family prompt, tokenizer, or generation behavior.
