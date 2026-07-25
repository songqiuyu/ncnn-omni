#pragma once

#include "ncnn_omni/audio.h"
#include "ncnn_omni/processors/whisper_log_mel.h"

namespace ncnn_omni {

// Qwen3-ASR-specific policy around the reusable Whisper feature extractor.
// Keeping this outside the model runner makes the processor independently
// testable and keeps upstream compatibility behavior out of inference code.
class Qwen3AsrAudioProcessor {
public:
    Result<LogMelFeatures> process(const AudioBuffer& audio) const;
    static std::vector<float> prepare_samples(const std::vector<float>& samples);

private:
    WhisperLogMel frontend_;
};

} // namespace ncnn_omni
