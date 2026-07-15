#pragma once

#include "ncnn_omni/status.h"

#include <net.h>

#include <string>
#include <utility>
#include <vector>

namespace ncnn_omni {

class NcnnModule {
public:
    Result<bool> load(const std::string& param_path,
                      const std::string& bin_path,
                      int num_threads);
    void clear();

    Result<std::vector<ncnn::Mat>> run(
        const std::vector<std::pair<std::string, ncnn::Mat>>& inputs,
        const std::vector<std::string>& outputs) const;

private:
    ncnn::Net net_;
    bool loaded_ = false;
};

} // namespace ncnn_omni
