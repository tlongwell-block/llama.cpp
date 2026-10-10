#pragma once
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// Runs a PCM stage on its own thread, in order, so the mouth computes its next frame while the
// stage handles this one. The watermark costs about a quarter of a frame on the CPU; inline it
// pushed the mouth past real time. push() blocks once `limit` samples wait, so the stage's own
// pacing still holds the mouth back. A stage failure is rethrown by the next push() or finish().
class frankie_pcm_worker {
    std::function<void(const float *, size_t)> stage;
    size_t limit, queued = 0;
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::vector<float>> queue;
    std::exception_ptr failure;
    bool closing = false, stopping = false;
    std::thread worker;

    void run() {
        for (;;) {
            std::vector<float> chunk;
            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [&] { return stopping || closing || !queue.empty(); });
                if (stopping || queue.empty()) { return; }
                chunk = std::move(queue.front());
                queue.pop_front();
            }
            try {
                stage(chunk.data(), chunk.size());
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex);
                failure = std::current_exception();
                queue.clear();
                queued = 0;
                changed.notify_all();
                return;
            }
            std::lock_guard<std::mutex> lock(mutex);
            queued -= chunk.size();
            changed.notify_all();
        }
    }

  public:
    frankie_pcm_worker(std::function<void(const float *, size_t)> stage, size_t limit) :
        stage(std::move(stage)), limit(limit), worker([this] { run(); }) {}

    // Unwinding: whatever still waits is dropped.
    ~frankie_pcm_worker() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        changed.notify_all();
        if (worker.joinable()) { worker.join(); }
    }

    void push(const float * pcm, size_t n) {
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock, [&] { return failure || queued < limit; });
        if (failure) { std::rethrow_exception(failure); }
        queue.emplace_back(pcm, pcm + n);
        queued += n;
        changed.notify_all();
    }

    // Normal completion: every pushed sample has been through the stage when this returns.
    void finish() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            closing = true;
        }
        changed.notify_all();
        if (worker.joinable()) { worker.join(); }
        if (failure) { std::rethrow_exception(failure); }
    }
};
