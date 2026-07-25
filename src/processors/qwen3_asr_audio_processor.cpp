#include "processors/qwen3_asr_audio_processor.h"

namespace ncnn_omni {

std::vector<float> Qwen3AsrAudioProcessor::prepare_samples(
    const std::vector<float>& samples)
{
    std::vector<float> result = samples;
    const size_t hop = static_cast<size_t>(WhisperLogMel::hop_length());
    result.resize((result.size() + hop - 1) / hop * hop, 0.f);
    return result;
}

Result<LogMelFeatures> Qwen3AsrAudioProcessor::process(const AudioBuffer& audio) const
{
    if (audio.sample_rate != WhisperLogMel::sample_rate() || audio.channels != 1)
        return Result<LogMelFeatures>("Qwen3-ASR requires mono 16 kHz PCM");
    return frontend_.compute(prepare_samples(audio.samples));
}

} // namespace ncnn_omni
