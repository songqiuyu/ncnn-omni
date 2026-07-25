#include "runtime/ncnn/ncnn_module.h"

#include <algorithm>
#include <unordered_set>

namespace ncnn_omni {
namespace {

Result<bool> validate_ports(const std::vector<NcnnPort>& ports,
                            const std::string& kind)
{
    std::unordered_set<std::string> logical_names;
    std::unordered_set<std::string> blob_names;
    for (const NcnnPort& port : ports) {
        if (port.logical_name.empty() || port.blob_name.empty())
            return Result<bool>(kind + " port names must not be empty");
        if (!logical_names.insert(port.logical_name).second)
            return Result<bool>("duplicate logical " + kind + " port: " + port.logical_name);
        if (!blob_names.insert(port.blob_name).second)
            return Result<bool>("duplicate ncnn " + kind + " blob: " + port.blob_name);
    }
    return Result<bool>(true);
}

const NcnnPort* find_port(const std::vector<NcnnPort>& ports,
                          const std::string& logical_name)
{
    const auto found = std::find_if(ports.begin(), ports.end(), [&](const NcnnPort& port) {
        return port.logical_name == logical_name;
    });
    return found == ports.end() ? nullptr : &*found;
}

} // namespace

Result<bool> NcnnModule::load(const NcnnModuleSpec& spec)
{
    if (spec.param_path.empty() || spec.bin_path.empty())
        return Result<bool>("ncnn module paths must not be empty");
    auto valid = validate_ports(spec.inputs, "input");
    if (!valid) return valid;
    valid = validate_ports(spec.outputs, "output");
    if (!valid) return valid;

    unload();
    spec_ = spec;
    net_.opt.use_vulkan_compute = spec.runtime.device == NcnnDevice::vulkan;
    net_.opt.use_fp16_arithmetic = spec.runtime.use_fp16_arithmetic;
    net_.opt.use_fp16_packed = spec.runtime.use_fp16_packed;
    net_.opt.use_fp16_storage = spec.runtime.use_fp16_storage;
    net_.opt.use_bf16_storage = spec.runtime.use_bf16_storage;
    net_.opt.num_threads = std::max(1, spec.runtime.num_threads);
    if (net_.load_param(spec.param_path.c_str()) != 0) {
        unload();
        return Result<bool>("failed to load ncnn param: " + spec.param_path);
    }
    if (net_.load_model(spec.bin_path.c_str()) != 0) {
        unload();
        return Result<bool>("failed to load ncnn weights: " + spec.bin_path);
    }
    loaded_ = true;
    return Result<bool>(true);
}

Result<bool> NcnnModule::load(const std::string& param_path,
                              const std::string& bin_path,
                              int num_threads)
{
    NcnnModuleSpec spec;
    spec.param_path = param_path;
    spec.bin_path = bin_path;
    spec.runtime.num_threads = num_threads;
    return load(spec);
}

void NcnnModule::unload()
{
    net_.clear();
    spec_ = {};
    loaded_ = false;
}

Result<std::vector<ncnn::Mat>> NcnnModule::run(
    const NcnnTensorMap& inputs,
    const std::vector<std::string>& outputs) const
{
    if (!loaded_) return Result<std::vector<ncnn::Mat>>("ncnn module is not loaded");
    ncnn::Extractor extractor = net_.create_extractor();
    extractor.set_light_mode(true);
    for (const auto& input : inputs) {
        const NcnnPort* port = spec_.inputs.empty() ? nullptr : find_port(spec_.inputs, input.first);
        if (!spec_.inputs.empty() && !port)
            return Result<std::vector<ncnn::Mat>>("unknown logical ncnn input: " + input.first);
        const std::string& blob_name = port ? port->blob_name : input.first;
        if (extractor.input(blob_name.c_str(), input.second) != 0)
            return Result<std::vector<ncnn::Mat>>("failed to bind ncnn input: " + input.first);
    }

    std::vector<ncnn::Mat> values;
    values.reserve(outputs.size());
    for (const std::string& name : outputs) {
        const NcnnPort* port = spec_.outputs.empty() ? nullptr : find_port(spec_.outputs, name);
        if (!spec_.outputs.empty() && !port)
            return Result<std::vector<ncnn::Mat>>("unknown logical ncnn output: " + name);
        const std::string& blob_name = port ? port->blob_name : name;
        ncnn::Mat value;
        if (extractor.extract(blob_name.c_str(), value) != 0)
            return Result<std::vector<ncnn::Mat>>("failed to extract ncnn output: " + name);
        values.emplace_back(std::move(value));
    }
    return Result<std::vector<ncnn::Mat>>(std::move(values));
}

} // namespace ncnn_omni
