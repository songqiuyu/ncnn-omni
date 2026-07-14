# Public headers

This tree will contain the experimental public C++ contracts:

- `api/`: engine, loaded model, session, task convenience APIs;
- `core/`: status/result, media values, requests, events, capabilities;
- `runtime/`: module, tensor, device, and execution-policy interfaces.

Public headers must not include model-private headers. Raw `ncnn::Net`,
`ncnn::Extractor`, `ncnn::Mat`, and `VkMat` are not part of the task API.

Headers are added during M0/M1 and remain experimental until all three reference
pipelines validate them.
