#pragma once

#include "ncnn_omni/audio.h"
#include "ncnn_omni/status.h"

#include <string>

namespace ncnn_omni {

// Decode an uncompressed PCM16 or float32 RIFF/WAVE file. This function only
// decodes and normalizes samples; task-specific sample-rate/channel validation
// belongs to the processor or model adapter that consumes the AudioBuffer.
Result<AudioBuffer> load_pcm_wav(const std::string& path);

} // namespace ncnn_omni
