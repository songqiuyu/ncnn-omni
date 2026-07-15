#include "ncnn_omni/qwen3_asr.h"

#include "processors/whisper_log_mel.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

template <typename T>
bool write_binary(const std::filesystem::path& path, const std::vector<T>& values)
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream) return false;
    stream.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
    return static_cast<bool>(stream);
}

void usage(const char* program)
{
    std::cerr << "Usage: " << program << " --audio FILE --output DIR\n";
}

} // namespace

int main(int argc, char** argv)
{
    std::string audio_path;
    std::string output_dir;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) {
            usage(argv[0]);
            return 2;
        }
        const std::string argument = argv[i];
        const std::string value = argv[++i];
        if (argument == "--audio") audio_path = value;
        else if (argument == "--output") output_dir = value;
        else {
            std::cerr << "Unknown argument: " << argument << '\n';
            usage(argv[0]);
            return 2;
        }
    }
    if (audio_path.empty() || output_dir.empty()) {
        usage(argv[0]);
        return 2;
    }

    auto audio = ncnn_omni::load_pcm_wav(audio_path);
    if (!audio) {
        std::cerr << "Audio error: " << audio.error() << '\n';
        return 1;
    }
    ncnn_omni::WhisperLogMel frontend;
    const std::vector<float> frontend_samples =
        ncnn_omni::prepare_qwen3_asr_frontend_samples(audio.value().samples);
    auto features = frontend.compute(frontend_samples);
    if (!features) {
        std::cerr << "Frontend error: " << features.error() << '\n';
        return 1;
    }

    const std::filesystem::path root(output_dir);
    std::error_code error;
    std::filesystem::create_directories(root, error);
    if (error) {
        std::cerr << "Cannot create output directory: " << error.message() << '\n';
        return 1;
    }
    if (!write_binary(root / "normalized_pcm.f32", audio.value().samples) ||
        !write_binary(root / "frontend_pcm.f32", frontend_samples) ||
        !write_binary(root / "input_features.f32", features.value().values)) {
        std::cerr << "Cannot write frontend tensor files\n";
        return 1;
    }

    std::ofstream metadata(root / "metadata.json");
    if (!metadata) {
        std::cerr << "Cannot write frontend metadata\n";
        return 1;
    }
    metadata << "{\n"
             << "  \"sample_rate\": " << audio.value().sample_rate << ",\n"
             << "  \"channels\": " << audio.value().channels << ",\n"
             << "  \"pcm_dtype\": \"float32\",\n"
             << "  \"pcm_samples\": " << audio.value().samples.size() << ",\n"
             << "  \"frontend_pcm_samples\": " << frontend_samples.size() << ",\n"
             << "  \"input_features_dtype\": \"float32\",\n"
             << "  \"input_features_shape\": [" << features.value().mel_bins
             << ", " << features.value().frames << "]\n"
             << "}\n";
    return metadata ? 0 : 1;
}
