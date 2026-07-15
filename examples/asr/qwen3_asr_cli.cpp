#include "ncnn_omni/qwen3_asr.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void usage(std::ostream& stream, const char* program)
{
    stream << "Usage: " << program
           << " --model DIR --assets DIR --audio FILE [options]\n\n"
              "Required:\n"
              "  --model DIR          Directory containing the five ncnn model pairs\n"
              "  --assets DIR         Directory containing vocab.json and merges.txt\n"
              "  --audio FILE         Mono 16 kHz PCM16/float32 WAV file\n\n"
              "Options:\n"
              "  --language NAME      Force a language, for example Chinese or English\n"
              "  --context TEXT       Add transcription context\n"
              "  --max-new-tokens N   Generation limit (default: 256)\n"
              "  --threads N          ncnn CPU threads (default: hardware concurrency)\n"
              "  -h, --help           Show this help\n";
}

} // namespace

int main(int argc, char** argv)
{
    std::string model;
    std::string assets;
    std::string audio_path;
    int threads = 0;
    ncnn_omni::AsrOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "-h" || argument == "--help") {
            usage(std::cout, argv[0]);
            return 0;
        }
        if (i + 1 >= argc) {
            usage(std::cerr, argv[0]);
            return 2;
        }
        const std::string value = argv[++i];
        if (argument == "--model") model = value;
        else if (argument == "--assets") assets = value;
        else if (argument == "--audio") audio_path = value;
        else if (argument == "--language") options.language = value;
        else if (argument == "--context") options.context = value;
        else if (argument == "--max-new-tokens") options.max_new_tokens = std::atoi(value.c_str());
        else if (argument == "--threads") threads = std::atoi(value.c_str());
        else {
            std::cerr << "Unknown argument: " << argument << '\n';
            usage(std::cerr, argv[0]);
            return 2;
        }
    }
    if (model.empty() || assets.empty() || audio_path.empty()) {
        usage(std::cerr, argv[0]);
        return 2;
    }

    auto audio = ncnn_omni::load_pcm_wav(audio_path);
    if (!audio) {
        std::cerr << "Audio error: " << audio.error() << '\n';
        return 1;
    }
    ncnn_omni::Qwen3Asr recognizer;
    auto loaded = recognizer.load(model, assets, threads);
    if (!loaded) {
        std::cerr << "Model error: " << loaded.error() << '\n';
        return 1;
    }
    auto result = recognizer.transcribe(audio.value(), options);
    if (!result) {
        std::cerr << "Inference error: " << result.error() << '\n';
        return 1;
    }

    std::cout << "language: " << result.value().language << '\n';
    std::cout << "text: " << result.value().text << '\n';
    std::cout << "raw: " << result.value().raw_text << '\n';
    std::cout << "tokens: " << result.value().token_ids.size() << '\n';
    std::cout << "token_ids:";
    for (int32_t id : result.value().token_ids) std::cout << ' ' << id;
    std::cout << '\n';
    std::cout << "timing_ms: frontend=" << result.value().frontend_ms
              << " audio_encoder=" << result.value().audio_encoder_ms
              << " prefill=" << result.value().prefill_ms
              << " decode=" << result.value().decode_ms << '\n';
    return 0;
}
