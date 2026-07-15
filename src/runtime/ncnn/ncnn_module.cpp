#include "runtime/ncnn/ncnn_module.h"

#include <algorithm>

namespace ncnn_omni {

Result<bool> NcnnModule::load(const std::string& param_path,
                              const std::string& bin_path,
                              int num_threads)
{
    clear();
    net_.opt.use_vulkan_compute = false;
    net_.opt.use_fp16_arithmetic = false;
    net_.opt.use_fp16_packed = false;
    net_.opt.use_fp16_storage = false;
    net_.opt.use_bf16_storage = false;
    net_.opt.num_threads = std::max(1, num_threads);
    if (net_.load_param(param_path.c_str()) != 0)
        return Result<bool>("failed to load ncnn param: " + param_path);
    if (net_.load_model(bin_path.c_str()) != 0)
        return Result<bool>("failed to load ncnn weights: " + bin_path);
    loaded_ = true;
    return Result<bool>(true);
}

void NcnnModule::clear()
{
    net_.clear();
    loaded_ = false;
}

Result<std::vector<ncnn::Mat>> NcnnModule::run(
    const std::vector<std::pair<std::string, ncnn::Mat>>& inputs,
    const std::vector<std::string>& outputs) const
{
    if (!loaded_) return Result<std::vector<ncnn::Mat>>("ncnn module is not loaded");
    ncnn::Extractor extractor = net_.create_extractor();
    extractor.set_light_mode(true);
    for (const auto& input : inputs) {
        if (extractor.input(input.first.c_str(), input.second) != 0)
            return Result<std::vector<ncnn::Mat>>("failed to bind ncnn input: " + input.first);
    }

    std::vector<ncnn::Mat> values;
    values.reserve(outputs.size());
    for (const std::string& name : outputs) {
        ncnn::Mat value;
        if (extractor.extract(name.c_str(), value) != 0)
            return Result<std::vector<ncnn::Mat>>("failed to extract ncnn output: " + name);
        values.emplace_back(std::move(value));
    }
    return Result<std::vector<ncnn::Mat>>(std::move(values));
}

} // namespace ncnn_omni
