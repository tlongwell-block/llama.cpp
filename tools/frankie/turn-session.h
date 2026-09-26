#pragma once

#include "component.h"
#include "runtime-options.h"
#include "mtmd-turn.h"
#include "mtmd.h"

#include <condition_variable>
#include <deque>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>

// Bounded acoustic workers keep prediction off the socket reader and brain/mouth
// paths. VAP expires after 500 ms; human-backchannel readings expire after 240 ms.
class frankie_turn_session {
    using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    struct frame { std::array<float, 1600> user{}, system{}; uint64_t end = 0, epoch = 0; };
    std::unique_ptr<mtmd_turn> turn, vap, bc;
    std::unique_ptr<mtmd_backchannel> detector;
public:
    struct reading {
        float next_system = 0, backchannel = 0, human_backchannel = 0;
        uint64_t end = 0, detector_end = 0;
        bool valid = false, detector_valid = false;
    };
private:
    // Each acoustic model group has its own bounded stream. GPU VAP work must
    // never delay CPU human-backchannel decisions behind a long brain prefill.
    class stream {
        std::thread worker;
        std::mutex mutex;
        std::condition_variable changed;
        std::deque<frame> queue;
        frame pending;
        reading latest;
        size_t step, used = 0;
        uint64_t samples = 0, epoch = 0;
        bool stopping = false, failed = false;
    public:
        stream(size_t step, std::function<reading(const frame &)> predict, std::function<void()> reset) : step(step) {
            worker = std::thread([this, predict, reset] {
                uint64_t active_epoch = 0;
                for (;;) {
                    frame current;
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        changed.wait(lock, [&] { return stopping || !queue.empty(); });
                        if (stopping) { return; }
                        current = std::move(queue.front()); queue.pop_front();
                    }
                    try {
                        if (current.epoch != active_epoch) { reset(); active_epoch = current.epoch; }
                        const auto result = predict(current);
                        std::lock_guard<std::mutex> lock(mutex);
                        if (current.epoch == epoch) { latest = result; }
                    } catch (const std::exception & e) {
                        std::lock_guard<std::mutex> lock(mutex);
                        failed = true; latest = {}; queue.clear();
                        std::cerr << "acoustic prediction disabled: " << e.what() << "\n";
                        return;
                    }
                }
            });
        }
        ~stream() {
            { std::lock_guard<std::mutex> lock(mutex); stopping = true; changed.notify_one(); }
            worker.join();
        }
        void append(const std::array<float, 512> & user, const std::array<float, 512> & system) {
            std::lock_guard<std::mutex> lock(mutex);
            if (failed) { return; }
            for (size_t i = 0; i < user.size(); ++i) {
                pending.user[used] = user[i]; pending.system[used++] = system[i]; ++samples;
                if (used != step) { continue; }
                if (queue.size() >= 30) {
                    latest = {}; queue.clear(); ++epoch;
                    std::cerr << "acoustic stream reset after backlog\n";
                }
                pending.end = samples; pending.epoch = epoch;
                queue.push_back(pending); used = 0; changed.notify_one();
            }
        }
        reading current() {
            std::lock_guard<std::mutex> lock(mutex);
            return latest;
        }
    };
    std::unique_ptr<stream> turn_stream, detector_stream;
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
    static std::unique_ptr<mtmd_backchannel> load_detector(component & source, const frankie_options & options) {
        constexpr size_t limit = 256 * 1024 * 1024;
        std::vector<char> bytes;
        if (!options.bc_detector.empty()) { bytes = frankie_read_file(options.bc_detector, limit); }
        else if (source.has_asset("assets.bc_det.gguf")) { bytes = source.asset("assets.bc_det.gguf", limit); }
        else { return nullptr; }
        ggml_context * raw = nullptr;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_from_buffer(bytes.data(), bytes.size(), {false, &raw}), gguf_free);
        context_ptr weights(raw, ggml_free);
        if (!metadata || !weights) { throw std::runtime_error("BC-Det model load failed"); }
        const auto key = gguf_find_key(metadata.get(), "turn.mode");
        if (key < 0 || gguf_get_kv_type(metadata.get(), key) != GGUF_TYPE_STRING ||
            std::string(gguf_get_val_str(metadata.get(), key)) != "detection") {
            throw std::runtime_error("expected a MaAI BC-Det asset");
        }
        auto params = mtmd_context_params_default();
        params.use_gpu = options.bc_detector_gpu;
        params.n_threads = std::min(options.threads, 4);
        params.model_reader_user_data = &bytes;
        params.model_reader_size = bytes.size();
        params.model_reader = [](void * user, void * output, uint64_t offset, size_t length) -> size_t {
            const auto & data = *static_cast<const std::vector<char> *>(user);
            if (offset > data.size() || length > data.size() - offset) { return 0; }
            std::memcpy(output, data.data() + offset, length);
            return length;
        };
        return std::make_unique<mtmd_backchannel>(raw, "embedded-bc-det", params);
    }

public:
    frankie_turn_session(component & source, const frankie_options & options) {
        if (options.vap_model.empty() || options.bc_model.empty()) {
            turn = load(source, "", "turn", mtmd_turn::mode::duplex, options.use_gpu);
        }
        if (!turn || !options.vap_model.empty()) { vap = load(source, options.vap_model, "vap", mtmd_turn::mode::vap, options.use_gpu); }
        if (!turn || !options.bc_model.empty()) { bc = load(source, options.bc_model, "bc", mtmd_turn::mode::backchannel, options.use_gpu); }
        detector = load_detector(source, options);
        if (turn || vap || bc) {
            turn_stream = std::make_unique<stream>(1600, [this](const frame & current) {
                reading r;
                if (turn) {
                    const auto prediction = turn->process(current.user, current.system);
                    r.next_system = prediction.next_speaker[1]; r.backchannel = prediction.backchannel;
                }
                if (vap) { r.next_system = vap->process(current.user, current.system).next_speaker[1]; }
                if (bc) { r.backchannel = bc->process(current.user, current.system).backchannel; }
                r.end = current.end; r.valid = true;
                return r;
            }, [this] {
                for (auto * model : {turn.get(), vap.get(), bc.get()}) { if (model) { model->reset(); } }
            });
        }
        if (detector) {
            detector_stream = std::make_unique<stream>(1280, [this](const frame & current) {
                std::array<float, 1280> user{}, system{};
                std::copy_n(current.user.begin(), 1280, user.begin());
                std::copy_n(current.system.begin(), 1280, system.begin());
                reading r;
                r.human_backchannel = detector->process(user, system);
                r.detector_valid = r.human_backchannel >= 0;
                r.detector_end = current.end;
                return r;
            }, [this] { detector->reset(); });
        }
    }
    bool has_vap() const { return bool(turn) || bool(vap); }
    bool has_bc() const { return bool(turn) || bool(bc); }
    bool has_detector() const { return bool(detector); }
    void append(const std::array<float, 512> & user, const std::array<float, 512> & system) {
        if (turn_stream) { turn_stream->append(user, system); }
        if (detector_stream) { detector_stream->append(user, system); }
    }
    reading current(uint64_t input_ms) {
        auto r = turn_stream ? turn_stream->current() : reading{};
        if (detector_stream) {
            const auto d = detector_stream->current();
            r.human_backchannel = d.human_backchannel;
            r.detector_end = d.detector_end;
            r.detector_valid = d.detector_valid;
        }
        r.valid = r.valid && input_ms >= r.end / 16 && input_ms - r.end / 16 <= 500;
        r.detector_valid = r.detector_valid && input_ms >= r.detector_end / 16 && input_ms - r.detector_end / 16 <= 240;
        return r;
    }
    bool release(uint64_t input_ms, int silence_ms) {
        const auto r = current(input_ms);
        return !has_vap() || silence_ms >= 1500 || !r.valid || r.next_system >= 0.4f;
    }
};
