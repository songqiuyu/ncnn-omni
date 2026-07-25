#pragma once

#include "ncnn_omni/status.h"
#include "runtime/ncnn/ncnn_module.h"

#include <net.h>

#include <cstdint>
#include <vector>

namespace ncnn_omni {

// Runtime description of the currently supported decoder graph contract. Model
// adapters translate model/package metadata into this structure; the generation
// loop itself does not depend on ASR, audio, prompt templates, or tokenizers.
struct TextDecoderConfig {
    int hidden_size = 0;
    int layer_count = 0;
    int kv_head_count = 0;
    int head_dimension = 0;
    int rope_dimension = 0;
    int32_t vocabulary_size = 0;
    double rope_theta = 10000.0;
    float mask_value = -10000.f;
    int initial_cache_length = 0;
    bool mask_initial_cache = false;
};

struct TextGenerationOptions {
    int max_new_tokens = 256;
    std::vector<int32_t> stop_token_ids;
};

struct TextGenerationResult {
    std::vector<int32_t> token_ids;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
};

class TextDecoder {
public:
    TextDecoder(NcnnModule& embedding,
                NcnnModule& decoder,
                NcnnModule& lm_head,
                TextDecoderConfig config);

    Result<ncnn::Mat> embed_tokens(const std::vector<int32_t>& ids) const;
    Result<TextGenerationResult> generate(
        const ncnn::Mat& prompt_embeddings,
        int prompt_length,
        const TextGenerationOptions& options) const;

private:
    NcnnModule& embedding_;
    NcnnModule& decoder_;
    NcnnModule& lm_head_;
    TextDecoderConfig config_;
    std::vector<double> rope_frequencies_;
};

} // namespace ncnn_omni
