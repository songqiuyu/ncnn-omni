#include "ncnn_omni/qwen3_asr.h"

#include "processors/qwen2_tokenizer.h"
#include "processors/whisper_log_mel.h"
#include "runtime/ncnn/ncnn_module.h"

#include <net.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
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
                         int threads)
{
    return module.load(param_path(root, name), bin_path(root, name), threads);
}

ncnn::Mat make_float_2d(int rows, int columns)
{
    return ncnn::Mat(columns, rows, static_cast<size_t>(4u), 1);
}

ncnn::Mat make_ids(const std::vector<int32_t>& ids)
{
    ncnn::Mat result(static_cast<int>(ids.size()), static_cast<size_t>(4u), 1);
    std::copy(ids.begin(), ids.end(), static_cast<int32_t*>(result.data));
    return result;
}

Result<ncnn::Mat> embed(NcnnModule& module, const std::vector<int32_t>& ids)
{
    auto outputs = module.run({{"in0", make_ids(ids)}}, {"out0"});
    if (!outputs) return Result<ncnn::Mat>(outputs.error());
    ncnn::Mat value = std::move(outputs.value()[0]);
    if (value.dims != 2 || value.w != kHidden || value.h != static_cast<int>(ids.size()))
        return Result<ncnn::Mat>("unexpected token embedding shape");
    return Result<ncnn::Mat>(std::move(value));
}

const std::array<double, kRopeDim>& rope_frequencies()
{
    static const auto frequencies = [] {
        std::array<double, kRopeDim> values{};
        for (int i = 0; i < kRopeDim; ++i)
            values[static_cast<size_t>(i)] =
                1.0 / std::pow(kRopeTheta, (2.0 * i) / kHeadDim);
        return values;
    }();
    return frequencies;
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

const std::vector<std::string>& decoder_input_names()
{
    static const auto names = [] {
        std::vector<std::string> values;
        values.reserve(4 + kLayers * 2);
        for (int i = 0; i < 4 + kLayers * 2; ++i)
            values.push_back("in" + std::to_string(i));
        return values;
    }();
    return names;
}

const std::vector<std::string>& decoder_output_names()
{
    static const auto names = [] {
        std::vector<std::string> values;
        values.reserve(1 + kLayers * 2);
        for (int i = 0; i < 1 + kLayers * 2; ++i)
            values.push_back("out" + std::to_string(i));
        return values;
    }();
    return names;
}

ncnn::Mat rope(int sequence, int position, bool cosine)
{
    const auto& frequencies = rope_frequencies();
    ncnn::Mat result = make_float_2d(sequence, kRopeDim);
    for (int row = 0; row < sequence; ++row) {
        float* values = result.row(row);
        for (int i = 0; i < kRopeDim; ++i) {
            const double angle = static_cast<double>(position + row) *
                                 frequencies[static_cast<size_t>(i)];
            values[i] = static_cast<float>(cosine ? std::cos(angle) : std::sin(angle));
        }
    }
    return result;
}

ncnn::Mat causal_mask(int sequence, int physical_past)
{
    ncnn::Mat result = make_float_2d(sequence, physical_past + sequence);
    for (int row = 0; row < sequence; ++row) {
        float* values = result.row(row);
        for (int column = 0; column < physical_past + sequence; ++column) {
            if (column == 0) values[column] = kMaskValue;
            else if (column >= physical_past && column > physical_past + row) values[column] = kMaskValue;
            else values[column] = 0.f;
        }
    }
    return result;
}

std::vector<ncnn::Mat> sentinel_caches()
{
    std::vector<ncnn::Mat> caches;
    caches.reserve(kLayers * 2);
    for (int i = 0; i < kLayers * 2; ++i) {
        ncnn::Mat cache(kHeadDim, 1, kKvHeads, static_cast<size_t>(4u), 1);
        cache.fill(0.f);
        caches.emplace_back(std::move(cache));
    }
    return caches;
}

struct DecoderOutput {
    ncnn::Mat hidden;
    std::vector<ncnn::Mat> caches;
};

Result<DecoderOutput> run_decoder(NcnnModule& decoder,
                                  const ncnn::Mat& hidden,
                                  int sequence,
                                  int logical_position,
                                  const std::vector<ncnn::Mat>& caches)
{
    if (static_cast<int>(caches.size()) != kLayers * 2)
        return Result<DecoderOutput>("decoder requires 56 KV tensors");
    const int physical_past = caches[0].h;
    const auto& input_names = decoder_input_names();
    std::vector<std::pair<std::string, ncnn::Mat>> inputs;
    inputs.reserve(60);
    inputs.emplace_back(input_names[0], hidden);
    inputs.emplace_back(input_names[1], causal_mask(sequence, physical_past));
    inputs.emplace_back(input_names[2], rope(sequence, logical_position, true));
    inputs.emplace_back(input_names[3], rope(sequence, logical_position, false));
    for (int i = 0; i < kLayers * 2; ++i)
        inputs.emplace_back(input_names[static_cast<size_t>(i + 4)],
                            caches[static_cast<size_t>(i)]);

    auto outputs = decoder.run(inputs, decoder_output_names());
    if (!outputs) return Result<DecoderOutput>(outputs.error());
    if (outputs.value()[0].dims != 2 || outputs.value()[0].w != kHidden ||
        outputs.value()[0].h != sequence)
        return Result<DecoderOutput>("unexpected decoder hidden shape");

    DecoderOutput result;
    result.hidden = std::move(outputs.value()[0]);
    result.caches.reserve(56);
    for (size_t i = 1; i < outputs.value().size(); ++i)
        result.caches.emplace_back(std::move(outputs.value()[i]));
    return Result<DecoderOutput>(std::move(result));
}

Result<int32_t> next_token(NcnnModule& lm_head, const ncnn::Mat& hidden, int row)
{
    ncnn::Mat last(kHidden, static_cast<size_t>(4u), 1);
    const float* source = hidden.row(row);
    std::copy(source, source + kHidden, static_cast<float*>(last.data));
    auto output = lm_head.run({{"in0", last}}, {"out0"});
    if (!output) return Result<int32_t>(output.error());
    const ncnn::Mat& logits = output.value()[0];
    if (logits.total() != kVocabSize) return Result<int32_t>("unexpected LM Head output shape");
    const float* values = static_cast<const float*>(logits.data);
    int32_t best = 0;
    for (int32_t i = 1; i < kVocabSize; ++i) {
        if (values[i] > values[best]) best = i;
    }
    return Result<int32_t>(best);
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
        model_root = model_dir;
        threads = requested_threads > 0 ? requested_threads :
                  std::max(1u, std::thread::hardware_concurrency());
        loaded = true;
        return Result<bool>(true);
    }

    Result<ncnn::Mat> encode_audio(const LogMelFeatures& mel)
    {
        NcnnModule audio_conv;
        auto load = load_module(audio_conv, model_root, "audio_conv", threads);
        if (!load) return Result<ncnn::Mat>(load.error());

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
            auto output = audio_conv.run({{"in0", conv_input}}, {"out0"});
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
        audio_conv.clear();

        NcnnModule transformer;
        load = load_module(transformer, model_root, "audio_transformer", threads);
        if (!load) return Result<ncnn::Mat>(load.error());
        // The reference CPU/SDPA path passes cu_seqlens but no block mask to
        // SDPA, so all audio tokens attend globally. Splitting at 104 changes
        // model semantics even though it matches the intended FlashAttention
        // varlen scheduling. CPU FP32 parity therefore requires one dense call.
        auto output = transformer.run({{"in0", transformer_input}}, {"out0"});
        if (!output) return Result<ncnn::Mat>(output.error());
        ncnn::Mat audio = std::move(output.value()[0]);
        if (audio.dims != 2 || audio.w != kHidden || audio.h != conv_tokens)
            return Result<ncnn::Mat>("unexpected Audio Transformer output shape");
        return Result<ncnn::Mat>(std::move(audio));
    }

    Result<AsrResult> transcribe(const AudioBuffer& audio, const AsrOptions& options)
    {
        if (!loaded) return Result<AsrResult>("Qwen3Asr is not loaded");
        if (audio.sample_rate != 16000 || audio.channels != 1)
            return Result<AsrResult>("Qwen3-ASR first version requires mono 16 kHz PCM");
        if (options.max_new_tokens <= 0) return Result<AsrResult>("max_new_tokens must be positive");

        AsrResult result;
        auto begin = Clock::now();
        // Canonicalize the PCM before both reference and ncnn feature paths.
        // See prepare_qwen3_asr_frontend_samples for the upstream partial-hop
        // mask/feature inconsistency that this compatibility step resolves.
        auto frontend_samples = prepare_qwen3_asr_frontend_samples(audio.samples);
        auto mel = frontend.compute(frontend_samples);
        if (!mel) return Result<AsrResult>(mel.error());
        result.frontend_ms = elapsed_ms(begin);

        begin = Clock::now();
        auto audio_embeddings = encode_audio(mel.value());
        if (!audio_embeddings) return Result<AsrResult>(audio_embeddings.error());
        result.audio_encoder_ms = elapsed_ms(begin);

        auto prompt = build_prompt(tokenizer, audio_embeddings.value().h, options);
        if (!prompt) return Result<AsrResult>(prompt.error());

        NcnnModule embedding;
        auto load = load_module(embedding, model_root, "embed_token", threads);
        if (!load) return Result<AsrResult>(load.error());
        auto prompt_embeddings = embed(embedding, prompt.value());
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

        NcnnModule decoder;
        load = load_module(decoder, model_root, "decoder", threads);
        if (!load) return Result<AsrResult>(load.error());
        NcnnModule lm_head;
        load = load_module(lm_head, model_root, "lm_head", threads);
        if (!load) return Result<AsrResult>(load.error());

        begin = Clock::now();
        auto decoded = run_decoder(decoder, prompt_embeddings.value(),
                                   static_cast<int>(prompt.value().size()), 0, sentinel_caches());
        if (!decoded) return Result<AsrResult>(decoded.error());
        prompt_embeddings.value().release();
        auto token = next_token(lm_head, decoded.value().hidden,
                                static_cast<int>(prompt.value().size()) - 1);
        if (!token) return Result<AsrResult>(token.error());
        result.prefill_ms = elapsed_ms(begin);

        begin = Clock::now();
        int logical_position = static_cast<int>(prompt.value().size());
        for (int step = 0; step < options.max_new_tokens; ++step) {
            const int32_t id = token.value();
            if (id == kEndOfText || id == kImEnd) break;
            result.token_ids.push_back(id);

            auto token_embedding = embed(embedding, {id});
            if (!token_embedding) return Result<AsrResult>(token_embedding.error());
            auto next = run_decoder(decoder, token_embedding.value(), 1,
                                    logical_position, decoded.value().caches);
            if (!next) return Result<AsrResult>(next.error());
            decoded = std::move(next);
            ++logical_position;
            token = next_token(lm_head, decoded.value().hidden, 0);
            if (!token) return Result<AsrResult>(token.error());
        }
        result.decode_ms = elapsed_ms(begin);

        auto raw = tokenizer.decode(result.token_ids, true);
        if (!raw) return Result<AsrResult>(raw.error());
        result.raw_text = raw.value();
        parse_output(result.raw_text, options.language, result.language, result.text);
        return Result<AsrResult>(std::move(result));
    }

    std::string model_root;
    int threads = 1;
    bool loaded = false;
    WhisperLogMel frontend;
    Qwen2Tokenizer tokenizer;
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
