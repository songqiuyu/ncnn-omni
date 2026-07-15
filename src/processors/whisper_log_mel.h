#pragma once

#include "ncnn_omni/status.h"

#include <vector>

namespace ncnn_omni {

struct LogMelFeatures {
    int mel_bins = 0;
    int frames = 0;
    // Row-major [mel_bins, frames].
    std::vector<float> values;
};

class WhisperLogMel {
public:
    WhisperLogMel();

    // Matches WhisperFeatureExtractor for Qwen3-ASR: mono 16 kHz, 128 bins,
    // n_fft=400, hop=160, center=True/reflect padding, dither=0.
    Result<LogMelFeatures> compute(const std::vector<float>& samples) const;

private:
    std::vector<double> hann_;
    // Row-major [128, 201].
    std::vector<double> mel_filters_;
};

} // namespace ncnn_omni
