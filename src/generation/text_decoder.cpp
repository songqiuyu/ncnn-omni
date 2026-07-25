#include "generation/text_decoder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <utility>

namespace ncnn_omni {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point begin)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
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

std::string cache_name(int index)
{
    return "cache_" + std::to_string(index);
}

bool contains(const std::vector<int32_t>& values, int32_t value)
{
    return std::find(values.begin(), values.end(), value) != values.end();
}

Result<bool> validate_config(const TextDecoderConfig& config)
{
    if (config.hidden_size <= 0 || config.layer_count <= 0 ||
        config.kv_head_count <= 0 || config.head_dimension <= 0 ||
        config.rope_dimension <= 0 || config.vocabulary_size <= 0)
        return Result<bool>("text decoder dimensions must be positive");
    if (config.rope_dimension > config.head_dimension)
        return Result<bool>("RoPE dimension exceeds attention head dimension");
    if (config.initial_cache_length <= 0)
        return Result<bool>("initial cache length must be positive");
    return Result<bool>(true);
}

struct DecoderStep {
    ncnn::Mat hidden;
    std::vector<ncnn::Mat> caches;
};

} // namespace

TextDecoder::TextDecoder(NcnnModule& embedding,
                         NcnnModule& decoder,
                         NcnnModule& lm_head,
                         TextDecoderConfig config)
    : embedding_(embedding), decoder_(decoder), lm_head_(lm_head), config_(std::move(config))
{
    rope_frequencies_.resize(static_cast<size_t>(std::max(0, config_.rope_dimension)));
    for (int i = 0; i < config_.rope_dimension; ++i) {
        rope_frequencies_[static_cast<size_t>(i)] =
            1.0 / std::pow(config_.rope_theta,
                           (2.0 * i) / config_.head_dimension);
    }
}

Result<ncnn::Mat> TextDecoder::embed_tokens(const std::vector<int32_t>& ids) const
{
    auto valid = validate_config(config_);
    if (!valid) return Result<ncnn::Mat>(valid.error());
    if (ids.empty()) return Result<ncnn::Mat>("cannot embed an empty token sequence");
    auto outputs = embedding_.run({{"token_ids", make_ids(ids)}}, {"embeddings"});
    if (!outputs) return Result<ncnn::Mat>(outputs.error());
    ncnn::Mat value = std::move(outputs.value()[0]);
    if (value.dims != 2 || value.w != config_.hidden_size ||
        value.h != static_cast<int>(ids.size()))
        return Result<ncnn::Mat>("unexpected token embedding shape");
    return Result<ncnn::Mat>(std::move(value));
}

Result<TextGenerationResult> TextDecoder::generate(
    const ncnn::Mat& prompt_embeddings,
    int prompt_length,
    const TextGenerationOptions& options) const
{
    auto valid = validate_config(config_);
    if (!valid) return Result<TextGenerationResult>(valid.error());
    if (options.max_new_tokens <= 0)
        return Result<TextGenerationResult>("max_new_tokens must be positive");
    if (prompt_length <= 0 || prompt_embeddings.dims != 2 ||
        prompt_embeddings.w != config_.hidden_size || prompt_embeddings.h != prompt_length)
        return Result<TextGenerationResult>("invalid prompt embedding shape");

    auto make_rope = [&](int sequence, int position, bool cosine) {
        ncnn::Mat result = make_float_2d(sequence, config_.rope_dimension);
        for (int row = 0; row < sequence; ++row) {
            float* values = result.row(row);
            for (int i = 0; i < config_.rope_dimension; ++i) {
                const double angle = static_cast<double>(position + row) *
                                     rope_frequencies_[static_cast<size_t>(i)];
                values[i] = static_cast<float>(cosine ? std::cos(angle) : std::sin(angle));
            }
        }
        return result;
    };

    auto make_mask = [&](int sequence, int physical_past) {
        ncnn::Mat result = make_float_2d(sequence, physical_past + sequence);
        for (int row = 0; row < sequence; ++row) {
            float* values = result.row(row);
            for (int column = 0; column < physical_past + sequence; ++column) {
                const bool initial_mask = config_.mask_initial_cache &&
                                          column < config_.initial_cache_length;
                const bool future_mask = column >= physical_past &&
                                         column > physical_past + row;
                values[column] = initial_mask || future_mask ? config_.mask_value : 0.f;
            }
        }
        return result;
    };

    auto run_decoder = [&](const ncnn::Mat& hidden,
                           int sequence,
                           int logical_position,
                           const std::vector<ncnn::Mat>& caches) -> Result<DecoderStep> {
        if (static_cast<int>(caches.size()) != config_.layer_count * 2)
            return Result<DecoderStep>("decoder KV tensor count mismatch");
        const int physical_past = caches.empty() ? 0 : caches[0].h;
        NcnnTensorMap inputs;
        inputs.reserve(static_cast<size_t>(4 + config_.layer_count * 2));
        inputs.emplace_back("hidden", hidden);
        inputs.emplace_back("attention_mask", make_mask(sequence, physical_past));
        inputs.emplace_back("rope_cos", make_rope(sequence, logical_position, true));
        inputs.emplace_back("rope_sin", make_rope(sequence, logical_position, false));
        std::vector<std::string> output_names{"hidden"};
        output_names.reserve(static_cast<size_t>(1 + config_.layer_count * 2));
        for (int i = 0; i < config_.layer_count * 2; ++i) {
            inputs.emplace_back(cache_name(i), caches[static_cast<size_t>(i)]);
            output_names.push_back(cache_name(i));
        }

        auto outputs = decoder_.run(inputs, output_names);
        if (!outputs) return Result<DecoderStep>(outputs.error());
        if (outputs.value()[0].dims != 2 || outputs.value()[0].w != config_.hidden_size ||
            outputs.value()[0].h != sequence)
            return Result<DecoderStep>("unexpected decoder hidden shape");
        DecoderStep result;
        result.hidden = std::move(outputs.value()[0]);
        result.caches.reserve(static_cast<size_t>(config_.layer_count * 2));
        for (size_t i = 1; i < outputs.value().size(); ++i)
            result.caches.emplace_back(std::move(outputs.value()[i]));
        return Result<DecoderStep>(std::move(result));
    };

    auto next_token = [&](const ncnn::Mat& hidden, int row) -> Result<int32_t> {
        ncnn::Mat last(config_.hidden_size, static_cast<size_t>(4u), 1);
        const float* source = hidden.row(row);
        std::copy(source, source + config_.hidden_size, static_cast<float*>(last.data));
        auto output = lm_head_.run({{"hidden", last}}, {"logits"});
        if (!output) return Result<int32_t>(output.error());
        const ncnn::Mat& logits = output.value()[0];
        if (logits.total() != static_cast<size_t>(config_.vocabulary_size))
            return Result<int32_t>("unexpected LM head output shape");
        const float* values = static_cast<const float*>(logits.data);
        int32_t best = 0;
        for (int32_t i = 1; i < config_.vocabulary_size; ++i) {
            if (values[i] > values[best]) best = i;
        }
        return Result<int32_t>(best);
    };

    std::vector<ncnn::Mat> caches;
    caches.reserve(static_cast<size_t>(config_.layer_count * 2));
    for (int i = 0; i < config_.layer_count * 2; ++i) {
        ncnn::Mat cache(config_.head_dimension,
                        config_.initial_cache_length,
                        config_.kv_head_count,
                        static_cast<size_t>(4u), 1);
        cache.fill(0.f);
        caches.emplace_back(std::move(cache));
    }

    TextGenerationResult result;
    auto begin = Clock::now();
    auto decoded = run_decoder(prompt_embeddings, prompt_length, 0, caches);
    if (!decoded) return Result<TextGenerationResult>(decoded.error());
    auto token = next_token(decoded.value().hidden, prompt_length - 1);
    if (!token) return Result<TextGenerationResult>(token.error());
    result.prefill_ms = elapsed_ms(begin);

    begin = Clock::now();
    int logical_position = prompt_length;
    for (int step = 0; step < options.max_new_tokens; ++step) {
        const int32_t id = token.value();
        if (contains(options.stop_token_ids, id)) break;
        result.token_ids.push_back(id);
        if (step + 1 == options.max_new_tokens) break;

        auto token_embedding = embed_tokens({id});
        if (!token_embedding) return Result<TextGenerationResult>(token_embedding.error());
        auto next = run_decoder(token_embedding.value(), 1, logical_position,
                                decoded.value().caches);
        if (!next) return Result<TextGenerationResult>(next.error());
        decoded = std::move(next);
        ++logical_position;
        token = next_token(decoded.value().hidden, 0);
        if (!token) return Result<TextGenerationResult>(token.error());
    }
    result.decode_ms = elapsed_ms(begin);
    return Result<TextGenerationResult>(std::move(result));
}

} // namespace ncnn_omni
