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

// Canonicalize a waveform for Qwen3-ASR's processor/model boundary. The
// upstream feature extractor produces floor(N / 160) Mel frames but a
// ceil(N / 160) feature mask for a partial final hop, which makes the audio
// tower fail. Applying the same zero pad before both reference and ncnn paths
// preserves every original sample and makes that boundary self-consistent.
std::vector<float> prepare_qwen3_asr_frontend_samples(const std::vector<float>& samples);

class WhisperLogMel {
public:
    WhisperLogMel();

    // Matches WhisperFeatureExtractor for Qwen3-ASR: mono 16 kHz, 128 bins,
    // n_fft=400, hop=160, center=True/reflect padding, dither=0.
    Result<LogMelFeatures> compute(const std::vector<float>& samples) const;

private:
    std::vector<double> hann_;
    // Precomputed [201, 400] DFT coefficients avoid trigonometric work per frame.
    std::vector<double> dft_cos_;
    std::vector<double> dft_sin_;
    // Row-major [128, 201].
    std::vector<double> mel_filters_;
};

} // namespace ncnn_omni
