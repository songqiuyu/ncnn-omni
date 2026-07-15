#include "processors/qwen2_tokenizer.h"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

void test_qwen2_tokenizer()
{
    const char* root = std::getenv("QWEN3_ASR_ASSETS");
    if (!root) return; // The test remains hermetic when model assets are unavailable.

    ncnn_omni::Qwen2Tokenizer tokenizer;
    auto loaded = tokenizer.load((std::filesystem::path(root) / "vocab.json").string(),
                                 (std::filesystem::path(root) / "merges.txt").string());
    if (!loaded) throw std::runtime_error(loaded.error());

    struct Case { const char* text; std::vector<int32_t> ids; };
    const Case cases[] = {
        {"system\n", {8948, 198}},
        {"user\n", {872, 198}},
        {"assistant\n", {77091, 198}},
        {"Hello world", {9707, 1879}},
        {"这是上下文。", {100346, 102285, 16744, 1773}},
        {"language Chinese", {11528, 8453}},
        {"Qwen3-ASR: test 123!", {48, 16948, 18, 12, 1911, 49, 25, 1273, 220, 16, 17, 18, 0}},
    };
    for (const auto& item : cases) {
        auto encoded = tokenizer.encode(item.text);
        if (!encoded) throw std::runtime_error(encoded.error());
        if (encoded.value() != item.ids)
            throw std::runtime_error(std::string("Qwen tokenizer mismatch for: ") + item.text);
    }
}
