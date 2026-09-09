#pragma once

#include "mtmd.h"
#include <cstddef>
#include <memory>
#include <vector>

struct ggml_context;

// The tensor context and its loaded data must outlive this processor.
class MTMD_API mtmd_ear {
public:
    explicit mtmd_ear(ggml_context * weights);
    ~mtmd_ear();
    // F32 frames [n_frames, 512] -> [n_frames + 1, 5120], with the tone row last.
    // One call contains 1..1024 frames. Calls on one instance must be serialized.
    std::vector<float> process(const float * frames, size_t n_frames);
private:
    struct impl;
    std::unique_ptr<impl> data;
};
