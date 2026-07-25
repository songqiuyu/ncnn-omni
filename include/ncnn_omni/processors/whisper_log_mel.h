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

// Reusable Whisper-compatible frontend. The initial implementation deliberately
// exposes the verified 16 kHz/128-bin contract instead of pretending that all
// STFT and Mel configurations have been validated.
class WhisperLogMel {
public:
    WhisperLogMel();

    // mono 16 kHz, 128 bins, n_fft=400, hop=160, center=True/reflect padding.
    Result<LogMelFeatures> compute(const std::vector<float>& samples) const;

    static constexpr int sample_rate() { return 16000; }
    static constexpr int hop_length() { return 160; }

private:
    std::vector<double> hann_;
    std::vector<double> dft_cos_;
    std::vector<double> dft_sin_;
    std::vector<double> mel_filters_;
};

} // namespace ncnn_omni
