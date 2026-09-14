#pragma once

#include <fstream>
#include <charconv>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

struct frankie_options {
    static constexpr uint32_t default_output_tokens = 32768;
    bool use_gpu = true;
    bool text_encoder_gpu = true;
    int threads = 4;
    uint32_t batch_size = 0; // automatic: 128 CPU, 512 GPU
    uint32_t ubatch_size = 0; // automatic: 128 CPU, 512 GPU
    uint32_t mtp_tokens = 0; // opt-in, requires an appended MTP head in the brain component
    uint32_t http_slots = 0; // experimental shared-brain HTTP slots
    uint32_t http_context_tokens = 0; // automatic: divide the total across voice and HTTP slots
    uint32_t context_tokens = 131072; // total shared KV pool
    uint32_t max_utterance_seconds = 90;
    uint32_t max_output_audio_seconds = 300;
    uint32_t max_output_tokens = default_output_tokens;
    uint32_t speech_context_words = 100;
    std::string cache_type = "q4_0";
    std::string thinking = "none";
    std::string http_thinking = "none";
    float side_scale = 0.0f;
    float presence_penalty = 0.0f;
    std::string voice, voice_text, voice_codes;
    std::string expression;
    std::string vap_model, bc_model;
    std::string ear_model, talker_model, mouth_model;

    uint32_t voice_context_tokens() const {
        return context_tokens - http_slots * http_context_tokens;
    }

    void resolve_context() {
        if (http_slots > 8) { throw std::runtime_error("--http-slots must be 0..8"); }
        if (context_tokens < 4096 || context_tokens > 262144u * (1 + http_slots)) {
            throw std::runtime_error("--ctx-size is the total KV pool: at least 4096 and at most 262144 per slot");
        }
        if (http_slots && !http_context_tokens) { http_context_tokens = context_tokens / (1 + http_slots); }
        if (http_slots && (http_context_tokens < 128 || http_context_tokens > 262144)) {
            throw std::runtime_error("--http-ctx-size must be 128..262144 per HTTP slot, or 0 for an equal split");
        }
        const uint64_t reserved = uint64_t(http_slots) * http_context_tokens;
        if (reserved + 4096 > context_tokens || context_tokens - reserved > 262144) {
            throw std::runtime_error("--ctx-size minus --http-slots * --http-ctx-size must leave 4096..262144 tokens for voice");
        }
        max_output_tokens = std::min(max_output_tokens, voice_context_tokens());
    }
};

inline size_t frankie_http_output_budget(size_t prompt, size_t requested, size_t context) {
    if (prompt >= context) {
        throw std::runtime_error("prompt fills or exceeds the per-request context (prompt_tokens=" +
            std::to_string(prompt) + ", context_tokens=" + std::to_string(context) + ")");
    }
    return std::min(requested, context - prompt);
}

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
