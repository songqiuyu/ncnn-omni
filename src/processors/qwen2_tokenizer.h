#pragma once

#include "ncnn_omni/status.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ncnn_omni {

class Qwen2Tokenizer {
public:
    Result<bool> load(const std::string& vocab_json, const std::string& merges_txt);
    Result<std::vector<int32_t>> encode(const std::string& text);
    Result<std::string> decode(const std::vector<int32_t>& ids,
                               bool skip_special_tokens = true) const;

private:
    std::vector<std::string> pretokenize(const std::string& text) const;
    std::vector<std::string> bpe(const std::string& token);

    std::unordered_map<std::string, int32_t> encoder_;
    std::unordered_map<int32_t, std::string> decoder_;
    std::unordered_map<std::string, int> merge_rank_;
    std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;
    std::unordered_map<unsigned char, std::string> byte_encoder_;
    std::unordered_map<std::string, unsigned char> byte_decoder_;
};

} // namespace ncnn_omni
