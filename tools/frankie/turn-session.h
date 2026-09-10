#pragma once

#include "component.h"
#include "runtime-options.h"
#include "mtmd-turn.h"

#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <thread>

// The acoustic models have their own bounded worker: neither the socket reader nor
// brain/mouth generation waits for a prediction. Readings older than 500 ms fail open.
class frankie_turn_session {
    using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    struct frame { std::array<float, 1600> user{}, system{}; uint64_t end = 0; bool reset = false; };
    std::unique_ptr<mtmd_turn> vap, bc;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable changed;
    bool stopping = false, failed = false;
    std::deque<frame> queue;
    frame pending;
    size_t used = 0;
    uint64_t samples = 0;
public:
    struct reading { float next_system = 0, backchannel = 0; uint64_t end = 0; bool valid = false; };
private:
    reading latest;
    static std::unique_ptr<mtmd_turn> load(component & source, const std::string & path,
                                         const std::string & name, mtmd_turn::mode mode, bool gpu) {
        const auto asset = "assets." + name + ".gguf";
        std::vector<char> bytes;
        if (!path.empty()) { bytes = frankie_read_file(path, 64 * 1024 * 1024); }
        else if (source.has_asset(asset)) { bytes = source.asset(asset, 64 * 1024 * 1024); }
        else { return nullptr; }
        if (bytes.size() > 64 * 1024 * 1024) { throw std::runtime_error("turn asset too large"); }
        ggml_context * raw = nullptr;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_from_buffer(bytes.data(), bytes.size(), {false, &raw}), gguf_free);
        context_ptr weights(raw, ggml_free);
        if (!metadata || !weights) { throw std::runtime_error("turn model load failed"); }
        return std::make_unique<mtmd_turn>(raw, mode, gpu);
    }
public:
    frankie_turn_session(component & source, const frankie_options & options) {
        vap = load(source, options.vap_model, "vap", mtmd_turn::mode::vap, options.use_gpu);
        bc = load(source, options.bc_model, "bc", mtmd_turn::mode::backchannel, options.use_gpu);
        if (!vap && !bc) { return; }
        worker = std::thread([this] {
            for (;;) {
                frame current;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    changed.wait(lock, [&] { return stopping || !queue.empty(); });
                    if (stopping) { return; }
                    current = std::move(queue.front()); queue.pop_front();
                }
                try {
                    if (current.reset) { if (vap) { vap->reset(); } if (bc) { bc->reset(); } }
                    reading r;
                    if (vap) { r.next_system = vap->process(current.user, current.system).next_speaker[1]; }
                    if (bc) { r.backchannel = bc->process(current.user, current.system).backchannel; }
                    r.end = current.end; r.valid = true;
                    std::lock_guard<std::mutex> lock(mutex);
                    latest = r;
                } catch (const std::exception & e) {
                    std::lock_guard<std::mutex> lock(mutex);
                    failed = true; latest.valid = false; queue.clear();
                    std::cerr << "turn prediction disabled: " << e.what() << "\n";
                    return;
                }
            }
        });
    }
    ~frankie_turn_session() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; changed.notify_all(); }
        if (worker.joinable()) { worker.join(); }
    }
    bool has_vap() const { return bool(vap); }
    bool has_bc() const { return bool(bc); }
    void append(const std::array<float, 512> & user, const std::array<float, 512> & system) {
        if (!worker.joinable()) { return; }
        std::lock_guard<std::mutex> lock(mutex);
        if (failed) { return; }
        for (size_t i = 0; i < user.size(); ++i) {
            pending.user[used] = user[i]; pending.system[used++] = system[i]; ++samples;
            if (used == 1600) {
                if (queue.size() >= 30) {
                    latest.valid = false; queue.clear(); pending.reset = true;
                    std::cerr << "turn prediction catching up: reset after three seconds of backlog\n";
                }
                pending.end = samples;
                queue.push_back(pending); pending.reset = false; used = 0; changed.notify_one();
            }
        }
    }
    reading current(uint64_t input_ms) {
        std::lock_guard<std::mutex> lock(mutex);
        auto r = latest;
        r.valid = r.valid && !failed && input_ms >= r.end / 16 && input_ms - r.end / 16 <= 500;
        return r;
    }
    bool release(uint64_t input_ms, int silence_ms) {
        const auto r = current(input_ms);
        return !vap || silence_ms >= 1500 || !r.valid || r.next_system >= 0.4f;
    }
};
