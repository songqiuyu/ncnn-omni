#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ncnn_omni {

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

} // namespace ncnn_omni
