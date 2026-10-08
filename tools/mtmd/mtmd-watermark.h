#pragma once

#include <cstdint>
#include <memory>
#include <vector>

struct ggml_context;

// AudioSeal's streaming 16-bit watermark generator (https://github.com/facebookresearch/audioseal).
// Copyright (c) Meta Platforms, Inc. and affiliates. MIT license.
// Causal SEANet encoder and decoder with LSTMs, 16 kHz, one 320-sample frame per step.
// The weights are shared; each stream owns its state vector.
class mtmd_watermark {
public:
    static constexpr int frame = 320;
    mtmd_watermark(ggml_context * weights, bool use_gpu);
    ~mtmd_watermark();
    // Sum of the message embeddings for a 16-bit payload, most significant bit first.
    std::vector<float> message(uint16_t payload) const;
    // Zeroed convolution and LSTM history for a new stream.
    std::vector<float> initial_state() const;
    // Watermarks one frame: out = in + watermark. Advances the stream state. Thread safe.
    void process(std::vector<float> & state, const std::vector<float> & message, const float * in, float * out);
private:
    struct impl;
    std::unique_ptr<impl> data;
};
