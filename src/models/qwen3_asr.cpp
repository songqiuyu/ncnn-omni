#include "ncnn_omni/qwen3_asr.h"

#include "generation/text_decoder.h"
#include "processors/qwen2_tokenizer.h"
#include "processors/qwen3_asr_audio_processor.h"
#include "runtime/ncnn/ncnn_module.h"

#include <net.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <mutex>
#include <thread>

namespace ncnn_omni {
namespace {

constexpr int kHidden = 1024;
constexpr int kAudioHidden = 896;
constexpr int kLayers = 28;
constexpr int kKvHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kRopeDim = 64;
constexpr int kAudioChunkFrames = 100;
constexpr int kAudioChunkTokens = 13;
constexpr int32_t kVocabSize = 151936;
constexpr int32_t kEndOfText = 151643;
constexpr int32_t kImStart = 151644;
constexpr int32_t kImEnd = 151645;
constexpr int32_t kAudioStart = 151669;
constexpr int32_t kAudioEnd = 151670;
constexpr int32_t kAudioPad = 151676;
constexpr int32_t kAsrText = 151704;
constexpr float kMaskValue = -10000.f;
constexpr double kRopeTheta = 1000000.0;

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point begin)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

std::string param_path(const std::string& root, const std::string& name)
{
    return (std::filesystem::path(root) / (name + ".ncnn.param")).string();
}

std::string bin_path(const std::string& root, const std::string& name)
{
    return (std::filesystem::path(root) / (name + ".ncnn.bin")).string();
}

Result<bool> load_module(NcnnModule& module,
                         const std::string& root,
                         const std::string& name,
                         int threads,
                         std::vector<NcnnPort> inputs,
                         std::vector<NcnnPort> outputs)
{
    NcnnModuleSpec spec;
    spec.id = name;
    spec.param_path = param_path(root, name);
    spec.bin_path = bin_path(root, name);
    spec.inputs = std::move(inputs);
    spec.outputs = std::move(outputs);
    spec.runtime.num_threads = threads;
    return module.load(spec);
}

ncnn::Mat make_float_2d(int rows, int columns)
{
    return ncnn::Mat(columns, rows, static_cast<size_t>(4u), 1);
}

TextDecoderConfig text_decoder_config()
{
    TextDecoderConfig config;
    config.hidden_size = kHidden;
    config.layer_count = kLayers;
    config.kv_head_count = kKvHeads;
    config.head_dimension = kHeadDim;
    config.rope_dimension = kRopeDim;
    config.vocabulary_size = kVocabSize;
    config.rope_theta = kRopeTheta;
    config.mask_value = kMaskValue;
    config.initial_cache_length = 1;
    config.mask_initial_cache = true;
    return config;
}

const std::array<float, kAudioChunkTokens * kAudioHidden>& audio_positions()
{
    static const auto positions = [] {
        std::array<float, kAudioChunkTokens * kAudioHidden> values{};
        const double increment = std::log(10000.0) / (kAudioHidden / 2 - 1.0);
        for (int row = 0; row < kAudioChunkTokens; ++row) {
            for (int dim = 0; dim < kAudioHidden; ++dim) {
                const double scale = std::exp(-increment * (dim % (kAudioHidden / 2)));
                const double angle = row * scale;
                values[static_cast<size_t>(row * kAudioHidden + dim)] = static_cast<float>(
                    dim < kAudioHidden / 2 ? std::sin(angle) : std::cos(angle));
            }
        }
        return values;
    }();
    return positions;
}

std::vector<NcnnPort> decoder_inputs()
{
    std::vector<NcnnPort> ports = {
        {"hidden", "in0"}, {"attention_mask", "in1"},
        {"rope_cos", "in2"}, {"rope_sin", "in3"}};
    ports.reserve(4 + kLayers * 2);
    for (int i = 0; i < kLayers * 2; ++i)
        ports.push_back({"cache_" + std::to_string(i), "in" + std::to_string(i + 4)});
    return ports;
}

std::vector<NcnnPort> decoder_outputs()
{
    std::vector<NcnnPort> ports = {{"hidden", "out0"}};
    ports.reserve(1 + kLayers * 2);
    for (int i = 0; i < kLayers * 2; ++i)
        ports.push_back({"cache_" + std::to_string(i), "out" + std::to_string(i + 1)});
    return ports;
}

void append_ids(std::vector<int32_t>& output, const std::vector<int32_t>& input)
{
    output.insert(output.end(), input.begin(), input.end());
}

Result<std::vector<int32_t>> build_prompt(Qwen2Tokenizer& tokenizer,
                                          int audio_tokens,
                                          const AsrOptions& options)
{
    std::vector<int32_t> ids;
    auto add_text = [&](const std::string& text) -> std::string {
        auto encoded = tokenizer.encode(text);
        if (!encoded) return encoded.error();
        append_ids(ids, encoded.value());
        return {};
    };

    ids.push_back(kImStart);
    if (std::string error = add_text("system\n" + options.context); !error.empty())
        return Result<std::vector<int32_t>>(error);
    ids.push_back(kImEnd);
    if (std::string error = add_text("\n"); !error.empty()) return Result<std::vector<int32_t>>(error);
    ids.push_back(kImStart);
    if (std::string error = add_text("user\n"); !error.empty()) return Result<std::vector<int32_t>>(error);
    ids.push_back(kAudioStart);
    ids.insert(ids.end(), static_cast<size_t>(audio_tokens), kAudioPad);
    ids.push_back(kAudioEnd);
    ids.push_back(kImEnd);
    if (std::string error = add_text("\n"); !error.empty()) return Result<std::vector<int32_t>>(error);
    ids.push_back(kImStart);
    if (std::string error = add_text("assistant\n"); !error.empty()) return Result<std::vector<int32_t>>(error);
    if (!options.language.empty()) {
        if (std::string error = add_text("language " + options.language); !error.empty())
            return Result<std::vector<int32_t>>(error);
        ids.push_back(kAsrText); // AddedToken, intentionally visible when decoding.
    }
    return Result<std::vector<int32_t>>(std::move(ids));
}

void parse_output(const std::string& raw,
                  const std::string& forced_language,
                  std::string& language,
                  std::string& text)
{
    if (!forced_language.empty()) {
        language = forced_language;
        text = raw;
        return;
    }
    const std::string tag = "<asr_text>";
    const size_t split = raw.find(tag);
    if (split == std::string::npos) {
        text = raw;
        return;
    }
    const std::string meta = raw.substr(0, split);
    text = raw.substr(split + tag.size());
    const std::string prefix = "language ";
    const size_t begin = meta.find(prefix);
    if (begin != std::string::npos) {
        size_t end = meta.find_first_of("\r\n", begin + prefix.size());
        language = meta.substr(begin + prefix.size(), end - (begin + prefix.size()));
    }
    if (language == "None" && text.empty()) language.clear();
}

} // namespace

class Qwen3Asr::Impl {
public:
    Result<bool> load(const std::string& model_dir,
                      const std::string& assets_dir,
                      int requested_threads)
    {
        std::lock_guard<std::mutex> lock(mutex);
        loaded = false;
        unload_modules();
        static const std::vector<std::string> modules = {
            "audio_conv", "audio_transformer", "embed_token", "decoder", "lm_head"};
        for (const std::string& module : modules) {
            if (!std::filesystem::is_regular_file(param_path(model_dir, module)) ||
                !std::filesystem::is_regular_file(bin_path(model_dir, module)))
                return Result<bool>("missing ncnn module files for: " + module);
        }
        const std::string vocab = (std::filesystem::path(assets_dir) / "vocab.json").string();
        const std::string merges = (std::filesystem::path(assets_dir) / "merges.txt").string();
        auto token_status = tokenizer.load(vocab, merges);
        if (!token_status) return token_status;

        const int threads = requested_threads > 0 ? requested_threads :
                            std::max(1u, std::thread::hardware_concurrency());
        auto status = load_module(audio_conv, model_dir, "audio_conv", threads,
                                  {{"features", "in0"}}, {{"encoded", "out0"}});
        if (!status) return load_failure(status.error());
        status = load_module(audio_transformer, model_dir, "audio_transformer", threads,
                             {{"features", "in0"}}, {{"encoded", "out0"}});
        if (!status) return load_failure(status.error());
        status = load_module(embedding, model_dir, "embed_token", threads,
                             {{"token_ids", "in0"}}, {{"embeddings", "out0"}});
        if (!status) return load_failure(status.error());
        status = load_module(decoder, model_dir, "decoder", threads,
                             decoder_inputs(), decoder_outputs());
        if (!status) return load_failure(status.error());
        status = load_module(lm_head, model_dir, "lm_head", threads,
                             {{"hidden", "in0"}}, {{"logits", "out0"}});
        if (!status) return load_failure(status.error());
        loaded = true;
        return Result<bool>(true);
    }

    Result<ncnn::Mat> encode_audio(const LogMelFeatures& mel)
    {
        const int full_chunks = mel.frames / kAudioChunkFrames;
        const int tail_frames = mel.frames % kAudioChunkFrames;
        const int conv_tokens = full_chunks * kAudioChunkTokens +
                                (tail_frames == 0 ? 0 : (tail_frames + 7) / 8);
        ncnn::Mat transformer_input = make_float_2d(conv_tokens, kAudioHidden);
        int output_row = 0;
        const auto& positions = audio_positions();
        for (int start = 0; start < mel.frames; start += kAudioChunkFrames) {
            const int width = std::min(kAudioChunkFrames, mel.frames - start);
            // Upstream pad_sequence pads a tail chunk to 100 whenever the
            // audio also contains at least one full chunk. Explicit input
            // zeros are not equivalent to a smaller convolution tensor:
            // convolution bias and GELU make the padded activations nonzero
            // before the following strided layers.
            const int padded_width = mel.frames > kAudioChunkFrames ? kAudioChunkFrames : width;
            ncnn::Mat conv_input(padded_width, 128, 1, static_cast<size_t>(4u), 1);
            conv_input.fill(0.f);
            for (int bin = 0; bin < 128; ++bin) {
                float* row = conv_input.channel(0).row(bin);
                const float* source = mel.values.data() + static_cast<size_t>(bin * mel.frames + start);
                std::copy(source, source + width, row);
            }
            auto output = audio_conv.run({{"features", conv_input}}, {"encoded"});
            if (!output) return Result<ncnn::Mat>(output.error());
            const ncnn::Mat& chunk = output.value()[0];
            const int expected = (width + 7) / 8;
            const int padded_output = (padded_width + 7) / 8;
            if (chunk.dims != 2 || chunk.w != kAudioHidden || chunk.h != padded_output)
                return Result<ncnn::Mat>("unexpected Audio Conv output shape");
            for (int row_index = 0; row_index < expected; ++row_index) {
                const float* source = chunk.row(row_index);
                const float* position = positions.data() + row_index * kAudioHidden;
                float* target = transformer_input.row(output_row++);
                for (int dim = 0; dim < kAudioHidden; ++dim)
                    target[dim] = source[dim] + position[dim];
            }
        }
        if (output_row != conv_tokens)
            return Result<ncnn::Mat>("Audio Conv token count mismatch");
        // The reference CPU/SDPA path passes cu_seqlens but no block mask to
        // SDPA, so all audio tokens attend globally. Splitting at 104 changes
        // model semantics even though it matches the intended FlashAttention
        // varlen scheduling. CPU FP32 parity therefore requires one dense call.
        auto output = audio_transformer.run({{"features", transformer_input}}, {"encoded"});
        if (!output) return Result<ncnn::Mat>(output.error());
        ncnn::Mat audio = std::move(output.value()[0]);
        if (audio.dims != 2 || audio.w != kHidden || audio.h != conv_tokens)
            return Result<ncnn::Mat>("unexpected Audio Transformer output shape");
        return Result<ncnn::Mat>(std::move(audio));
    }

    Result<AsrResult> transcribe(const AudioBuffer& audio, const AsrOptions& options)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!loaded) return Result<AsrResult>("Qwen3Asr is not loaded");
        if (options.max_new_tokens <= 0) return Result<AsrResult>("max_new_tokens must be positive");

        AsrResult result;
        auto begin = Clock::now();
        auto mel = audio_processor.process(audio);
        if (!mel) return Result<AsrResult>(mel.error());
        result.frontend_ms = elapsed_ms(begin);

        begin = Clock::now();
        auto audio_embeddings = encode_audio(mel.value());
        if (!audio_embeddings) return Result<AsrResult>(audio_embeddings.error());
        result.audio_encoder_ms = elapsed_ms(begin);

        auto prompt = build_prompt(tokenizer, audio_embeddings.value().h, options);
        if (!prompt) return Result<AsrResult>(prompt.error());

        TextDecoder text_decoder(embedding, decoder, lm_head, text_decoder_config());
        auto prompt_embeddings = text_decoder.embed_tokens(prompt.value());
        if (!prompt_embeddings) return Result<AsrResult>(prompt_embeddings.error());
        int audio_row = 0;
        for (size_t row = 0; row < prompt.value().size(); ++row) {
            if (prompt.value()[row] != kAudioPad) continue;
            if (audio_row >= audio_embeddings.value().h)
                return Result<AsrResult>("audio placeholder count is smaller than audio embeddings");
            std::copy(audio_embeddings.value().row(audio_row),
                      audio_embeddings.value().row(audio_row) + kHidden,
                      prompt_embeddings.value().row(static_cast<int>(row)));
            ++audio_row;
        }
        if (audio_row != audio_embeddings.value().h)
            return Result<AsrResult>("audio placeholder count does not match audio embeddings");
        audio_embeddings.value().release();

        TextGenerationOptions generation_options;
        generation_options.max_new_tokens = options.max_new_tokens;
        generation_options.stop_token_ids = {kEndOfText, kImEnd};
        auto generated = text_decoder.generate(prompt_embeddings.value(),
                                               static_cast<int>(prompt.value().size()),
                                               generation_options);
        if (!generated) return Result<AsrResult>(generated.error());
        result.token_ids = std::move(generated.value().token_ids);
        result.prefill_ms = generated.value().prefill_ms;
        result.decode_ms = generated.value().decode_ms;

        auto raw = tokenizer.decode(result.token_ids, true);
        if (!raw) return Result<AsrResult>(raw.error());
        result.raw_text = raw.value();
        parse_output(result.raw_text, options.language, result.language, result.text);
        return Result<AsrResult>(std::move(result));
    }

private:
    Result<bool> load_failure(const std::string& error)
    {
        unload_modules();
        return Result<bool>(error);
    }

    void unload_modules()
    {
        audio_conv.unload();
        audio_transformer.unload();
        embedding.unload();
        decoder.unload();
        lm_head.unload();
    }

    std::mutex mutex;
    bool loaded = false;
    Qwen3AsrAudioProcessor audio_processor;
    Qwen2Tokenizer tokenizer;
    NcnnModule audio_conv;
    NcnnModule audio_transformer;
    NcnnModule embedding;
    NcnnModule decoder;
    NcnnModule lm_head;
};

Qwen3Asr::Qwen3Asr() : impl_(std::make_unique<Impl>()) {}
Qwen3Asr::~Qwen3Asr() = default;
Qwen3Asr::Qwen3Asr(Qwen3Asr&&) noexcept = default;
Qwen3Asr& Qwen3Asr::operator=(Qwen3Asr&&) noexcept = default;

Result<bool> Qwen3Asr::load(const std::string& model_dir,
                            const std::string& assets_dir,
                            int num_threads)
{
    return impl_->load(model_dir, assets_dir, num_threads);
}

Result<AsrResult> Qwen3Asr::transcribe(const AudioBuffer& audio, const AsrOptions& options)
{
    return impl_->transcribe(audio, options);
}

} // namespace ncnn_omni
