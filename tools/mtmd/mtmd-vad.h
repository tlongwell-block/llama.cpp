#pragma once

#include <array>
#include <memory>
#include <string>

// Fixed Silero v5 16 kHz stream. Weights are immutable; state belongs to one stream.
class mtmd_vad {
public:
    explicit mtmd_vad(const std::string & path);
    ~mtmd_vad();
    mtmd_vad(const mtmd_vad &) = delete;
    mtmd_vad & operator=(const mtmd_vad &) = delete;
    void reset();
    float process(const std::array<float, 512> & samples);

private:
    struct impl;
    std::unique_ptr<impl> data;
};
