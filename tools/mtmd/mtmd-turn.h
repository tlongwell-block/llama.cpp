#pragma once

#include <array>
#include <memory>
#include <vector>

struct ggml_context;

// Streaming CPC/ALiBi turn projection, shared by VAP and backchannel checkpoints.
// Each step consumes 100 ms of aligned user and actually played system audio at 16 kHz.
class mtmd_turn {
public:
    enum class mode { vap, backchannel, duplex, detection };
    struct result {
        std::array<float, 2> next_speaker{}; // user, system (VAP)
        float backchannel = 0;
    };
    mtmd_turn(ggml_context * weights, mode kind, bool use_gpu);
    ~mtmd_turn();
    void reset();
    result process(const std::array<float, 1600> & user, const std::array<float, 1600> & system);
    // One 80 ms pair of continuous Mimi features, microphone first.
    float detect(const std::array<float, 512> & user, const std::array<float, 512> & system);
    // Optional frozen-reference diagnostics, in channel/frame/embedding order.
    std::vector<float> encoded() const;
    std::vector<float> hidden() const;
private:
    struct impl;
    std::unique_ptr<impl> data;
};

struct mtmd_context_params;

// MaAI human-backchannel detector. One shared Mimi encoder, two streaming states.
class mtmd_backchannel {
public:
    mtmd_backchannel(ggml_context * weights, const char * path, const mtmd_context_params & params);
    ~mtmd_backchannel();
    void reset();
    // Returns -1 for the startup frame, otherwise the microphone BC probability.
    float process(const std::array<float, 1280> & user, const std::array<float, 1280> & system);
    std::vector<float> encoded() const;
private:
    struct impl;
    std::unique_ptr<impl> data;
};
