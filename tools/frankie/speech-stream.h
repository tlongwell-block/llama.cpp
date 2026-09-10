#pragma once
#include "mouth-session.h"
#include "speech-boundary.h"

#include <condition_variable>
#include <chrono>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>

// Two pending phrases bound the hidden rows retained ahead of playback.
class speech_stream {
    brain_session & brain;
    mouth_session & mouth;
    mouth_session::audio_callback on_audio;
    brain_session::stage_callback on_stage;
    std::function<void(const std::string &)> on_phrase;
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<brain_session::response> queue;
    bool finished = false;
    bool speaking = false;
    size_t audio_samples = 0;
    std::chrono::steady_clock::time_point audio_clock;
    std::exception_ptr failure;
    std::thread worker;
    std::string committed;

    void consume() {
        try {
            for (;;) {
                brain_session::response phrase;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    changed.wait(lock, [&] { return finished || !queue.empty() || brain.cancelled.load(); });
                    if (brain.cancelled.load() || (finished && queue.empty())) {
                        return;
                    }
                    phrase = std::move(queue.front());
                    queue.pop_front();
                    speaking = true;
                    changed.notify_all();
                }
                mouth.speak(phrase, [&](const float * pcm, size_t samples) {
                    if (on_audio) { on_audio(pcm, samples); }
                    std::lock_guard<std::mutex> lock(mutex);
                    if (!audio_samples) { audio_clock = std::chrono::steady_clock::now(); }
                    audio_samples += samples;
                    changed.notify_all();
                }, on_stage);
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    speaking = false;
                    changed.notify_all();
                }
                if (on_phrase) { on_phrase(phrase.message.content); }
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            failure = std::current_exception();
            changed.notify_all();
        }
    }

  public:
    speech_stream(brain_session & b, mouth_session & m, mouth_session::audio_callback audio, brain_session::stage_callback stage = {},
                  std::function<void(const std::string &)> phrase = {}) :
        brain(b), mouth(m), on_audio(std::move(audio)), on_stage(std::move(stage)), on_phrase(std::move(phrase)), worker([this] { consume(); }) {}

    ~speech_stream() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            finished = true;
            queue.clear();
        }
        changed.notify_all();
        if (worker.joinable()) {
            mouth.cancelled = true;
            worker.join();
        }
    }

    void feed(const brain_session::response & response, bool last) {
        {
            std::unique_lock<std::mutex> lock(mutex);
            // Give the mouth two frames of lead before another brain decode shares the device.
            while (speaking && !brain.cancelled.load() && !failure) {
                const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - audio_clock).count();
                if (audio_samples && int64_t(audio_samples / 24) - elapsed >= 160) { break; }
                changed.wait_for(lock, std::chrono::milliseconds(5));
            }
            if (failure) {
                std::rethrow_exception(failure);
            }
        }
        const auto & text = response.message.content;
        if (text.compare(0, committed.size(), committed) != 0) {
            throw std::runtime_error("streaming parser retracted spoken text");
        }
        const size_t end = frankie_speech_boundary(text, committed.size(), last || !response.message.tool_calls.empty());
        if (end <= committed.size()) {
            return;
        }
        auto phrase = response;
        const size_t lead = response.content_offset == std::string::npos ? response.raw.text.rfind(text) : response.content_offset;
        if (lead == std::string::npos || response.raw.text.compare(lead, text.size(), text) != 0) {
            throw std::runtime_error("parsed speech is not a contiguous generated span");
        }
        phrase.content_offset = lead + committed.size();
        phrase.message.content = text.substr(committed.size(), end - committed.size());
        phrase.message.tool_calls.clear();
        committed = text.substr(0, end);
        if (phrase.message.content.find_first_not_of(" \t\r\n") == std::string::npos) {
            return;
        }
        std::unique_lock<std::mutex> lock(mutex);
        while (queue.size() >= 2 && !brain.cancelled.load() && !failure) {
            changed.wait_for(lock, std::chrono::milliseconds(20));
        }
        if (failure) {
            std::rethrow_exception(failure);
        }
        if (brain.cancelled.load()) {
            throw std::runtime_error("cancelled");
        }
        queue.push_back(std::move(phrase));
        changed.notify_all();
    }

    void finish() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            finished = true;
        }
        changed.notify_all();
        worker.join();
        if (failure) {
            std::rethrow_exception(failure);
        }
    }
};
