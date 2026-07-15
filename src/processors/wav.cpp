#include "ncnn_omni/qwen3_asr.h"

#include <cstring>
#include <fstream>
#include <limits>

namespace ncnn_omni {
namespace {

uint16_t read_u16(std::istream& stream)
{
    unsigned char bytes[2]{};
    stream.read(reinterpret_cast<char*>(bytes), 2);
    return static_cast<uint16_t>(bytes[0] | (bytes[1] << 8));
}

uint32_t read_u32(std::istream& stream)
{
    unsigned char bytes[4]{};
    stream.read(reinterpret_cast<char*>(bytes), 4);
    return static_cast<uint32_t>(bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24));
}

} // namespace

Result<AudioBuffer> load_pcm_wav(const std::string& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return Result<AudioBuffer>("cannot open WAV file: " + path);

    char tag[4]{};
    stream.read(tag, 4);
    if (std::memcmp(tag, "RIFF", 4) != 0) return Result<AudioBuffer>("WAV is not RIFF");
    (void)read_u32(stream);
    stream.read(tag, 4);
    if (std::memcmp(tag, "WAVE", 4) != 0) return Result<AudioBuffer>("RIFF file is not WAVE");

    uint16_t format = 0;
    uint16_t channels = 0;
    uint16_t bits = 0;
    uint32_t sample_rate = 0;
    std::vector<unsigned char> data;

    while (stream && (!format || data.empty())) {
        stream.read(tag, 4);
        if (stream.gcount() != 4) break;
        const uint32_t size = read_u32(stream);
        if (std::memcmp(tag, "fmt ", 4) == 0) {
            if (size < 16) return Result<AudioBuffer>("invalid WAV fmt chunk");
            format = read_u16(stream);
            channels = read_u16(stream);
            sample_rate = read_u32(stream);
            (void)read_u32(stream);
            (void)read_u16(stream);
            bits = read_u16(stream);
            stream.seekg(size - 16, std::ios::cur);
        } else if (std::memcmp(tag, "data", 4) == 0) {
            data.resize(size);
            stream.read(reinterpret_cast<char*>(data.data()), size);
        } else {
            stream.seekg(size, std::ios::cur);
        }
        if (size & 1U) stream.seekg(1, std::ios::cur);
    }

    if (!format || data.empty()) return Result<AudioBuffer>("WAV is missing fmt or data chunk");
    if (channels != 1) return Result<AudioBuffer>("first version requires mono WAV");
    if (sample_rate != 16000) return Result<AudioBuffer>("first version requires 16 kHz WAV");

    AudioBuffer audio;
    audio.sample_rate = static_cast<int>(sample_rate);
    audio.channels = channels;
    if (format == 1 && bits == 16) {
        audio.samples.resize(data.size() / 2);
        for (size_t i = 0; i < audio.samples.size(); ++i) {
            const uint16_t raw = static_cast<uint16_t>(data[i * 2] | (data[i * 2 + 1] << 8));
            const int16_t value = static_cast<int16_t>(raw);
            audio.samples[i] = value / 32768.f;
        }
    } else if (format == 3 && bits == 32) {
        audio.samples.resize(data.size() / 4);
        std::memcpy(audio.samples.data(), data.data(), data.size());
    } else {
        return Result<AudioBuffer>("first version supports PCM16 or float32 WAV only");
    }
    return Result<AudioBuffer>(std::move(audio));
}

} // namespace ncnn_omni
