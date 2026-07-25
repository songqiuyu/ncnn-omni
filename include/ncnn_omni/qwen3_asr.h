#pragma once

#include "ncnn_omni/asr.h"
#include "ncnn_omni/audio.h"
#include "ncnn_omni/processors/wav.h"
#include "ncnn_omni/status.h"

#include <memory>
#include <string>

namespace ncnn_omni {

class Qwen3Asr {
public:
    Qwen3Asr();
    ~Qwen3Asr();
    Qwen3Asr(Qwen3Asr&&) noexcept;
    Qwen3Asr& operator=(Qwen3Asr&&) noexcept;

    Qwen3Asr(const Qwen3Asr&) = delete;
    Qwen3Asr& operator=(const Qwen3Asr&) = delete;

    // Load and retain all five ncnn modules. A loaded instance can be reused for
    // multiple transcriptions without re-reading weights. Calls on one instance
    // are serialized; request-local decoder/KV state is never shared.
    Result<bool> load(const std::string& model_dir,
                      const std::string& assets_dir,
                      int num_threads = 0);

    Result<AsrResult> transcribe(const AudioBuffer& audio,
                                const AsrOptions& options = {});

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ncnn_omni
