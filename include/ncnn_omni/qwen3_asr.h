#pragma once

#include "ncnn_omni/status.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ncnn_omni {

struct AudioBuffer {
    std::vector<float> samples;
    int sample_rate = 0;
    int channels = 0;
};

struct AsrOptions {
    int max_new_tokens = 256;
    std::string context;
    std::string language;
};

struct AsrResult {
    std::string text;
    std::string language;
    std::string raw_text;
    std::vector<int32_t> token_ids;
    double frontend_ms = 0.0;
    double audio_encoder_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
};

Result<AudioBuffer> load_pcm_wav(const std::string& path);

class Qwen3Asr {
public:
    Qwen3Asr();
    ~Qwen3Asr();
    Qwen3Asr(Qwen3Asr&&) noexcept;
    Qwen3Asr& operator=(Qwen3Asr&&) noexcept;

    Qwen3Asr(const Qwen3Asr&) = delete;
    Qwen3Asr& operator=(const Qwen3Asr&) = delete;

    // model_dir contains the five .ncnn.param/.ncnn.bin pairs. assets_dir
    // contains vocab.json and merges.txt from the original Qwen3-ASR model.
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
