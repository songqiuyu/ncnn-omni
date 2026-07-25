#include "runtime/ncnn/ncnn_module.h"

#include <stdexcept>

void test_ncnn_module()
{
    ncnn_omni::Result<bool> literal_error("failure");
    if (literal_error || literal_error.error() != "failure")
        throw std::runtime_error("Result<bool> treated a string error as success");

    ncnn_omni::NcnnModule module;
    auto run_before_load = module.run({}, {});
    if (run_before_load)
        throw std::runtime_error("unloaded ncnn module unexpectedly ran");

    ncnn_omni::NcnnModuleSpec empty_paths;
    auto empty = module.load(empty_paths);
    if (empty || empty.error().empty())
        throw std::runtime_error("empty ncnn module paths were not rejected");

    ncnn_omni::NcnnModuleSpec duplicate_ports;
    duplicate_ports.param_path = "missing.param";
    duplicate_ports.bin_path = "missing.bin";
    duplicate_ports.inputs = {{"features", "in0"}, {"features", "in1"}};
    auto duplicate = module.load(duplicate_ports);
    if (duplicate || duplicate.error().find("duplicate logical input") == std::string::npos)
        throw std::runtime_error("duplicate logical ncnn ports were not rejected");

    ncnn_omni::NcnnModuleSpec missing_files;
    missing_files.param_path = "missing.param";
    missing_files.bin_path = "missing.bin";
    missing_files.inputs = {{"features", "in0"}};
    missing_files.outputs = {{"encoded", "out0"}};
    auto missing = module.load(missing_files);
    if (missing || module.is_loaded())
        throw std::runtime_error("failed ncnn load left the module loaded");
}
