#pragma once

#include <array>
#include <memory>
#include <string>

struct ggml_context;

// Trained residual added to the talker text rows. One instance per stream.
class mtmd_side {
public:
    explicit mtmd_side(const std::string & path, bool use_gpu = false);
    // The caller keeps the F32 weight context alive for this instance.
    explicit mtmd_side(ggml_context * weights, bool use_gpu = false);
    ~mtmd_side();
    std::array<float, 2048> process(const std::array<float, 5120> & hidden);
private:
    struct impl;
    std::unique_ptr<impl> data;
};

// A trained linear head on a pooled brain state, followed by gated speaker task vectors.
// Weights: expression.{mu,sd,weight,bias,directions}; directions already include each axis's cap.
class mtmd_expression {
public:
    explicit mtmd_expression(ggml_context * weights, bool use_gpu = false);
    ~mtmd_expression();
    struct result {
        std::array<float, 4> probabilities;
        std::array<float, 2048> offset;
    };
    result process(const std::array<float, 5120> & pooled);
    std::array<float, 2048> direction(size_t emotion) const;
private:
    struct impl;
    std::unique_ptr<impl> data;
};
