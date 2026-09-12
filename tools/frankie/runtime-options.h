#pragma once

#include <fstream>
#include <charconv>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

struct frankie_options {
    bool use_gpu = true;
    bool text_encoder_gpu = true;
    int threads = 4;
    uint32_t batch_size = 0; // automatic: 128 CPU, 512 GPU
    uint32_t ubatch_size = 0; // automatic: 128 CPU, 512 GPU
    uint32_t mtp_tokens = 0; // opt-in, requires an appended MTP head in the brain component
    uint32_t context_tokens = 131072;
    uint32_t max_utterance_seconds = 90;
    uint32_t max_output_audio_seconds = 300;
    uint32_t max_output_tokens = 4096;
    uint32_t speech_context_words = 100;
    std::string cache_type = "q4_0";
    std::string thinking = "none";
    float side_scale = 0.0f;
    float presence_penalty = 0.0f;
    std::string voice, voice_text, voice_codes;
    std::string expression;
    std::string vap_model, bc_model;
    std::string ear_model, talker_model, mouth_model;
};

inline int frankie_thinking_budget(const std::string & effort) {
    if (effort == "none" || effort == "off") { return 0; }
    if (effort == "minimal") { return 128; }
    if (effort == "low") { return 512; }
    if (effort == "medium") { return 2048; }
    if (effort == "high") { return 8192; }
    if (effort == "xhigh") { return 16384; }
    if (effort == "max") { return 32768; }
    throw std::runtime_error("thinking must be none, minimal, low, medium, high, xhigh or max");
}

inline std::vector<char> frankie_read_file(const std::string & path, size_t limit) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0 || file.tellg() > static_cast<std::streamoff>(limit)) {
        throw std::runtime_error("missing, empty or oversized file: " + path);
    }
    std::vector<char> bytes(static_cast<size_t>(file.tellg()));
    file.seekg(0);
    if (!file.read(bytes.data(), bytes.size())) { throw std::runtime_error("cannot read: " + path); }
    return bytes;
}

inline uint32_t frankie_unsigned(const std::string & value) {
    uint32_t result;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size()) {
        throw std::runtime_error("expected an unsigned integer: " + value);
    }
    return result;
}

inline float frankie_float(const std::string & value) {
    size_t used = 0;
    const float result = std::stof(value, &used);
    if (used != value.size()) { throw std::runtime_error("invalid numeric option: " + value); }
    return result;
}
