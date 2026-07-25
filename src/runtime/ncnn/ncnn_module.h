#pragma once

#include "ncnn_omni/status.h"

#include <net.h>

#include <string>
#include <utility>
#include <vector>

namespace ncnn_omni {

enum class NcnnDevice {
    cpu,
    vulkan,
};

struct NcnnRuntimeOptions {
    int num_threads = 1;
    NcnnDevice device = NcnnDevice::cpu;
    bool use_fp16_arithmetic = false;
    bool use_fp16_packed = false;
    bool use_fp16_storage = false;
    bool use_bf16_storage = false;
};

struct NcnnPort {
    std::string logical_name;
    std::string blob_name;
};

struct NcnnModuleSpec {
    std::string id;
    std::string param_path;
    std::string bin_path;
    std::vector<NcnnPort> inputs;
    std::vector<NcnnPort> outputs;
    NcnnRuntimeOptions runtime;
};

using NcnnTensorMap = std::vector<std::pair<std::string, ncnn::Mat>>;

class NcnnModule {
public:
    Result<bool> load(const NcnnModuleSpec& spec);

    // Compatibility overload for simple modules whose logical and ncnn blob
    // names are identical.
    Result<bool> load(const std::string& param_path,
                      const std::string& bin_path,
                      int num_threads);
    void unload();
    void clear() { unload(); }

    bool is_loaded() const { return loaded_; }
    const NcnnModuleSpec& spec() const { return spec_; }

    // Input/output names are logical port names from the module specification.
    // Each invocation creates a fresh Extractor, so mutable invocation state is
    // never stored in the loaded module.
    Result<std::vector<ncnn::Mat>> run(
        const NcnnTensorMap& inputs,
        const std::vector<std::string>& outputs) const;

private:
    ncnn::Net net_;
    NcnnModuleSpec spec_;
    bool loaded_ = false;
};

} // namespace ncnn_omni
