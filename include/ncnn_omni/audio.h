#pragma once

#include <vector>

namespace ncnn_omni {

// Model-independent, interleaved floating-point PCM.
struct AudioBuffer {
    std::vector<float> samples;
    int sample_rate = 0;
    int channels = 0;
};

} // namespace ncnn_omni
