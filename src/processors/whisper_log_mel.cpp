#include "processors/whisper_log_mel.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>

namespace ncnn_omni {
namespace {

constexpr int kSampleRate = 16000;
constexpr int kNfft = 400;
constexpr int kHop = 160;
constexpr int kFreqBins = kNfft / 2 + 1;
constexpr int kMelBins = 128;
constexpr double kPi = 3.141592653589793238462643383279502884;

double hz_to_mel_slaney(double hz)
{
    constexpr double f_sp = 200.0 / 3.0;
    double mel = hz / f_sp;
    constexpr double min_log_hz = 1000.0;
    constexpr double min_log_mel = min_log_hz / f_sp;
    constexpr double logstep = 0.06875177742094912; // log(6.4) / 27
    if (hz >= min_log_hz) mel = min_log_mel + std::log(hz / min_log_hz) / logstep;
    return mel;
}

double mel_to_hz_slaney(double mel)
{
    constexpr double f_sp = 200.0 / 3.0;
    double hz = mel * f_sp;
    constexpr double min_log_hz = 1000.0;
    constexpr double min_log_mel = min_log_hz / f_sp;
    constexpr double logstep = 0.06875177742094912;
    if (mel >= min_log_mel) hz = min_log_hz * std::exp(logstep * (mel - min_log_mel));
    return hz;
}

std::vector<float> reflect_pad(const std::vector<float>& input, int amount)
{
    std::vector<float> out(input.size() + 2 * amount, 0.f);
    if (input.empty()) return out;
    if (input.size() == 1) {
        std::fill(out.begin(), out.end(), input.front());
        return out;
    }

    auto reflected_index = [size = static_cast<int>(input.size())](int index) {
        while (index < 0 || index >= size) {
            if (index < 0) index = -index;
            if (index >= size) index = 2 * size - 2 - index;
        }
        return index;
    };
    for (int i = -amount; i < static_cast<int>(input.size()) + amount; ++i)
        out[static_cast<size_t>(i + amount)] = input[static_cast<size_t>(reflected_index(i))];
    return out;
}

} // namespace

std::vector<float> prepare_qwen3_asr_frontend_samples(const std::vector<float>& samples)
{
    std::vector<float> result = samples;
    result.resize((result.size() + kHop - 1) / kHop * kHop, 0.f);
    return result;
}

WhisperLogMel::WhisperLogMel()
{
    hann_.resize(kNfft);
    for (int i = 0; i < kNfft; ++i)
        hann_[static_cast<size_t>(i)] = 0.5 - 0.5 * std::cos(2.0 * kPi * i / kNfft);

    dft_cos_.resize(kFreqBins * kNfft);
    dft_sin_.resize(kFreqBins * kNfft);
    for (int bin = 0; bin < kFreqBins; ++bin) {
        for (int n = 0; n < kNfft; ++n) {
            const double phase = -2.0 * kPi * bin * n / kNfft;
            const size_t index = static_cast<size_t>(bin * kNfft + n);
            dft_cos_[index] = std::cos(phase);
            dft_sin_[index] = std::sin(phase);
        }
    }

    std::vector<double> filter_hz(kMelBins + 2);
    const double mel_min = hz_to_mel_slaney(0.0);
    const double mel_max = hz_to_mel_slaney(kSampleRate / 2.0);
    for (int i = 0; i < kMelBins + 2; ++i) {
        const double mel = mel_min + (mel_max - mel_min) * i / (kMelBins + 1.0);
        filter_hz[static_cast<size_t>(i)] = mel_to_hz_slaney(mel);
    }

    mel_filters_.assign(kMelBins * kFreqBins, 0.0);
    for (int mel = 0; mel < kMelBins; ++mel) {
        const double left = filter_hz[static_cast<size_t>(mel)];
        const double center = filter_hz[static_cast<size_t>(mel + 1)];
        const double right = filter_hz[static_cast<size_t>(mel + 2)];
        const double enorm = 2.0 / (right - left);
        for (int bin = 0; bin < kFreqBins; ++bin) {
            const double hz = (kSampleRate / 2.0) * bin / (kFreqBins - 1.0);
            const double lower = (hz - left) / (center - left);
            const double upper = (right - hz) / (right - center);
            mel_filters_[static_cast<size_t>(mel * kFreqBins + bin)] =
                std::max(0.0, std::min(lower, upper)) * enorm;
        }
    }
}

Result<LogMelFeatures> WhisperLogMel::compute(const std::vector<float>& samples) const
{
    if (samples.empty()) return Result<LogMelFeatures>("audio contains no samples");

    const std::vector<float> padded = reflect_pad(samples, kNfft / 2);
    const int stft_frames = 1 + static_cast<int>((padded.size() - kNfft) / kHop);
    // Whisper explicitly removes the final centered STFT frame.
    const int frames = stft_frames - 1;
    if (frames <= 0) return Result<LogMelFeatures>("audio is too short for Whisper Log-Mel");

    std::vector<double> power(static_cast<size_t>(kFreqBins));
    LogMelFeatures output;
    output.mel_bins = kMelBins;
    output.frames = frames;
    output.values.resize(static_cast<size_t>(kMelBins * frames));
    float global_max = -std::numeric_limits<float>::infinity();

    // A direct 400-point real DFT is intentionally used in the correctness-first
    // implementation. It is deterministic and has no platform FFT dependency.
    for (int frame = 0; frame < frames; ++frame) {
        const int offset = frame * kHop;
        for (int bin = 0; bin < kFreqBins; ++bin) {
            double re = 0.0;
            double im = 0.0;
            const double* cos_values = dft_cos_.data() + bin * kNfft;
            const double* sin_values = dft_sin_.data() + bin * kNfft;
            for (int n = 0; n < kNfft; ++n) {
                const double sample = static_cast<double>(padded[static_cast<size_t>(offset + n)]) *
                                      hann_[static_cast<size_t>(n)];
                re += sample * cos_values[n];
                im += sample * sin_values[n];
            }
            power[static_cast<size_t>(bin)] = re * re + im * im;
        }

        for (int mel = 0; mel < kMelBins; ++mel) {
            double energy = 0.0;
            const double* filter = mel_filters_.data() + static_cast<size_t>(mel * kFreqBins);
            for (int bin = 0; bin < kFreqBins; ++bin)
                energy += filter[bin] * power[static_cast<size_t>(bin)];
            const float value = static_cast<float>(std::log10(std::max(1e-10, energy)));
            output.values[static_cast<size_t>(mel * frames + frame)] = value;
            global_max = std::max(global_max, value);
        }
    }

    const float floor = global_max - 8.f;
    for (float& value : output.values) value = (std::max(value, floor) + 4.f) / 4.f;
    return Result<LogMelFeatures>(std::move(output));
}

} // namespace ncnn_omni
