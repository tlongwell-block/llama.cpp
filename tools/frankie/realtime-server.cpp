#include "base64.hpp"
#include "brain-session.h"
#include "common.h"
#include "cpp-httplib/httplib.h"
#include "mouth-session.h"
#include "speech-stream.h"
#include "mtmd-vad.h"
#include "nlohmann/json.hpp"
#define MA_NO_DEVICE_IO
#define MA_NO_ENCODING
#define MA_NO_DECODING
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio/miniaudio.h"

#include <cstring>
#include <condition_variable>
#include <optional>
#include <iostream>
#include <mutex>
#include <set>
#include <thread>
using json = nlohmann::ordered_json;

// Local manual-turn prototype. Realtime owns items; model objects only execute requests.
struct realtime_session {
    brain_session &             brain;
    mouth_session &             mouth;
    httplib::ws::WebSocket &    socket;
    brain_session::request      request;
    json                        config;
    std::vector<char>           audio;
    std::map<std::string, bool> pending_calls;
    std::thread                 worker;
    std::thread                 input_worker;
    std::atomic<bool>           input_busy{false};
    std::string                 input_failure;
    size_t                      precommit_samples = 0;
    std::mutex                  state;
    bool                        busy = false, connected = true;
    size_t                      serial = 0;
    std::string                 response_id;
    component vad_source;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> vad_weights{ nullptr, ggml_free };
    std::unique_ptr<mtmd_vad> vad;
    std::vector<char> vad_frame;
    std::map<std::string, std::vector<float>> unencoded;
    std::map<std::string, size_t> audio_messages;
    std::map<std::string, size_t> emitted_samples;
    std::set<std::string> truncated;
    struct heard_phrase { size_t end; std::string text; };
    std::map<std::string, std::vector<heard_phrase>> phrase_history;
    std::map<std::string, size_t> heard_samples;
    std::string heard_text(const std::string & id) const {
        std::string text;
        const auto phrases = phrase_history.find(id);
        const auto heard = heard_samples.find(id);
        if (phrases != phrase_history.end() && heard != heard_samples.end()) {
            for (const auto & phrase : phrases->second) {
                if (phrase.end > heard->second) { break; }
                text += phrase.text;
            }
        }
        return text + " [interrupted by the user]";
    }
    bool speaking = false;
    bool server_vad = false, auto_response = false, auto_interrupt = false;
    int silence_ms = 240, silence_frames = 0, speech_frames = 0;
    float threshold = 0.5f;
    uint64_t input_samples = 0;
    std::string input_id;
    struct speculative_turn {
        std::string input;
        bool ready = false, committed = false, authorized = false, abandoned = false;
        int resumed_frames = 0;
        std::vector<char> capture;
        size_t user_index = 0;
        std::string output;
        bool tool_released = false, merge_requested = false;
        std::chrono::steady_clock::time_point released;
        std::optional<brain_session::saved_state> saved;
    };
    std::shared_ptr<speculative_turn> tentative, released_turn;
    std::optional<brain_session::saved_state> merge_restore;

    void restore_merged_input() {
        if (merge_restore) {
            brain.restore_state(std::move(*merge_restore));
            merge_restore.reset();
        }
    }

    void merge_false_start(const std::string & id, size_t samples) {
        const auto turn = released_turn;
        if (!turn || turn->output != id || turn->tool_released || turn->merge_requested ||
            samples >= 24000 * 700 / 1000 ||
            std::chrono::steady_clock::now() - turn->released >= std::chrono::milliseconds(700) ||
            !speaking || input_busy.load() || !unencoded.empty() || turn->user_index >= request.chat.messages.size()) { return; }
        if (audio.size() + turn->capture.size() > 24000 * 2 * 19) { return; }
        // A published call cannot be undone. Never merge across any call/result pair.
        for (size_t i = turn->user_index; i < request.chat.messages.size(); ++i) {
            if (!request.chat.messages[i].tool_calls.empty() || request.chat.messages[i].role == "tool") { return; }
        }
        turn->merge_requested = true;
        brain.cancelled = true;
        mouth.cancelled = true;
        request.chat.messages.resize(turn->user_index);
        request.audio_rows.erase("[FRANKIE_AUDIO_" + turn->input + "]");
        send({{"type", "conversation.item.deleted"}, {"item_id", turn->input}});
        audio_messages.erase(id);
        audio.insert(audio.begin(), turn->capture.begin(), turn->capture.end());
        precommit_samples = 0;
        if (turn->saved) {
            merge_restore = std::move(turn->saved);
            turn->saved.reset();
        }
        std::cerr << id << " stage=false_start_merged heard_samples=" << samples << "\n";
        changed.notify_all();
    }

    std::condition_variable changed;

    void commit_tentative() {
        if (!tentative || tentative->abandoned || tentative->committed || !tentative->ready ||
            silence_frames * 32 < silence_ms) { return; }
        speaking = false;
        send({{"type", "input_audio_buffer.speech_stopped"}, {"item_id", input_id},
              {"audio_end_ms", input_samples / 24}});
        std::cerr << input_id << " stage=speech_stopped wall_us=" << std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count() << "\n";
        commit_audio();
        tentative->committed = true;
        if (auto_response) { authorize_tentative(); }
    }

    void authorize_tentative() {
        if (!tentative || !tentative->committed || tentative->authorized) {
            throw std::runtime_error("speculative response is not awaiting authorization");
        }
        tentative->authorized = true;
        send({{"type", "response.created"},
              {"response", {{"id", response_id}, {"status", "in_progress"}, {"output", json::array()}}}});
        changed.notify_all();
    }

    void await_authorization(const std::shared_ptr<speculative_turn> & turn) {
        if (!turn) { return; }
        std::unique_lock<std::mutex> lock(state);
        turn->ready = true;
        commit_tentative();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
        while (!turn->authorized && !turn->abandoned && connected && !brain.cancelled.load()) {
            if (std::chrono::steady_clock::now() >= deadline) { throw std::runtime_error("speculative authorization deadline"); }
            changed.wait_for(lock, std::chrono::milliseconds(20));
        }
        if (turn->abandoned || !connected || brain.cancelled.load()) { throw std::runtime_error("cancelled"); }
    }


    std::string next_id(const char * prefix) { return std::string(prefix) + std::to_string(++serial); }

    void send(json event) {
        event["event_id"] = next_id("event_");
        if (connected && !socket.send(event.dump())) {
            connected       = false;
            brain.cancelled = true;
            mouth.cancelled = true;
        }
    }

    void fail(const std::string & message,
              const std::string & client_event_id,
              const char *        code = "unsupported_or_invalid_request") {
        json detail = {
            { "type",    "invalid_request_error" },
            { "code",    code                    },
            { "message", message                 }
        };
        if (!client_event_id.empty()) {
            detail["event_id"] = client_event_id;
        }
        send({
            { "type",  "error" },
            { "error", detail  }
        });
    }

    realtime_session(brain_session & b, mouth_session & m, httplib::ws::WebSocket & s, const std::string & package) :
        brain(b), mouth(m), socket(s), vad_source(package, "vad") {
        ggml_context * raw = nullptr;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_from_callback(component::callback, &vad_source, 1024 * 1024, vad_source.size(), { false, &raw }), gguf_free);
        vad_weights.reset(raw);
        if (!metadata || !vad_weights) {
            throw std::runtime_error("VAD component load failed");
        }
        vad = std::make_unique<mtmd_vad>(raw, true);
        brain.reset();
        request.chat.enable_thinking  = false;
        request.chat.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        common_chat_msg system;
        system.role    = "system";
        system.content = "You are Frankie. Respond briefly.";
        request.chat.messages.push_back(system);
        config = {
            { "id",                "session_local"                                                             },
            { "type",              "realtime"                                                                  },
            { "model",             "frankie"                                                                   },
            { "instructions",      system.content                                                              },
            { "output_modalities", json::array({ "audio" })                                                    },
            { "tools",             json::array()                                                               },
            { "audio",
             { { "input",
                  { { "format", { { "type", "audio/pcm" }, { "rate", 24000 } } }, { "turn_detection", nullptr } } },
                { "output",
                  { { "format", { { "type", "audio/pcm" }, { "rate", 24000 } } }, { "voice", "frankie" } } } } }
        };
        send({
            { "type",    "session.created" },
            { "session", config            }
        });
    }

    ~realtime_session() {
        {
            std::lock_guard<std::mutex> lock(state);
            connected       = false;
            brain.cancelled = true;
            mouth.cancelled = true;
        }
        if (worker.joinable()) {
            worker.join();
        }
        if (input_worker.joinable()) { input_worker.join(); }
    }

    void settle_input() {
        if (input_worker.joinable()) { input_worker.join(); }
        if (!input_failure.empty()) {
            auto failure = std::move(input_failure);
            input_failure.clear();
            throw std::runtime_error("input precommit failed: " + failure);
        }
    }

    void precommit_audio() {
        // Keep the reader/VAD independent of model work. At most one snapshot is in flight.
        if (busy || input_busy.load() || !speaking || !unencoded.empty()) { return; }
        for (const auto & call : pending_calls) { if (!call.second) { return; } }
        const size_t samples = audio.size() / 2;
        const size_t lag = 25, batch = 2;
        if (samples / 1920 < lag + batch || samples < precommit_samples + batch * 1920) { return; }
        settle_input();
        auto captured = audio;
        auto snapshot = request;
        const auto marker = "[FRANKIE_AUDIO_" + input_id + "]";
        common_chat_msg user;
        user.role = "user";
        user.content = marker;
        snapshot.chat.messages.push_back(user);
        precommit_samples = samples;
        brain.cancelled = false;
        input_busy = true;
        input_worker = std::thread([this, captured = std::move(captured), snapshot = std::move(snapshot), marker, samples]() mutable {
            try {
                restore_merged_input();
                std::vector<float> pcm(captured.size() / 3 + 64);
                auto n = ma_convert_frames(pcm.data(), pcm.size(), ma_format_f32, 1, 16000,
                                           captured.data(), captured.size() / 2, ma_format_s16, 1, 24000);
                if (!n || n > 16000 * 20) { throw std::runtime_error("precommit resampling failed"); }
                pcm.resize(n);
                auto rows = brain.encode_audio(pcm);
                const size_t count = std::min(samples / 1920 - 25, rows.size() / 5120 - 1);
                brain.precommit(snapshot, marker, rows, count);
                std::cerr << marker << " stage=input_precommit_done capture_samples=" << samples
                          << " rows=" << count << " wall_us=" << std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now().time_since_epoch()).count() << "\n";
            } catch (const std::exception & e) { input_failure = e.what(); }
            input_busy = false;
        });
    }

    void update(const json & value, const std::string & event_id) {
        if (!value.is_object() || value.value("type", "") != "realtime") {
            throw std::runtime_error("session.type must be realtime");
        }
        auto merged = config;
        merged.merge_patch(value);
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (it.key() != "type" && it.key() != "model" && it.key() != "instructions" && it.key() != "tools" &&
                it.key() != "output_modalities" && it.key() != "audio") {
                throw std::runtime_error("unsupported session field: " + it.key());
            }
        }
        if (merged.value("model", "") != "frankie") {
            throw std::runtime_error("unknown model");
        }
        auto & a = merged.at("audio");
        if (!a.is_object()) {
            throw std::runtime_error("audio must be an object");
        }
        for (auto it = a.begin(); it != a.end(); ++it) {
            if (it.key() != "input" && it.key() != "output") {
                throw std::runtime_error("unsupported audio field");
            }
        }
        auto detection = a.at("input").value("turn_detection", json());
        if (!detection.is_null()) {
            if (!detection.is_object() || detection.value("type", "") != "server_vad") {
                throw std::runtime_error("only server_vad turn detection supported");
            }
            for (auto it = detection.begin(); it != detection.end(); ++it) {
                if (it.key() != "type" && it.key() != "threshold" && it.key() != "silence_duration_ms" &&
                    it.key() != "prefix_padding_ms" && it.key() != "create_response" && it.key() != "interrupt_response") {
                    throw std::runtime_error("unsupported VAD setting");
                }
            }
            const float t = detection.value("threshold", 0.5f);
            const int silence = detection.value("silence_duration_ms", 240);
            if (!std::isfinite(t) || t <= 0 || t >= 1 || silence < 160 || silence > 2000 ||
                detection.value("prefix_padding_ms", 320) != 320) {
                throw std::runtime_error("VAD settings bounds (prefix padding must be 320 ms)");
            }
            detection["threshold"] = t;
            detection["silence_duration_ms"] = silence;
            detection["prefix_padding_ms"] = 320;
            detection["create_response"] = detection.value("create_response", true);
            detection["interrupt_response"] = detection.value("interrupt_response", true);
            a["input"]["turn_detection"] = detection;
        }
        for (const char * direction : { "input", "output" }) {
            auto & f = a.at(direction).at("format");
            if (!f.is_object() || f.size() != 2 || f.value("type", "") != "audio/pcm" || f.value("rate", 0) != 24000) {
                throw std::runtime_error("PCM24k required");
            }
            for (auto it = a[direction].begin(); it != a[direction].end(); ++it) {
                if (it.key() != "format" && !(std::string(direction) == "input" && it.key() == "turn_detection") &&
                    !(std::string(direction) == "output" && it.key() == "voice")) {
                    throw std::runtime_error("unsupported audio option");
                }
            }
        }
        if (a["output"].value("voice", "") != "frankie") {
            throw std::runtime_error("unknown voice");
        }
        if (merged["output_modalities"] != json::array({ "audio" }) &&
            merged["output_modalities"] != json::array({ "text" })) {
            throw std::runtime_error("choose audio or text output");
        }
        auto instructions = merged.at("instructions").get<std::string>();
        if (instructions.size() > 16384 || instructions.find("[FRANKIE_") != std::string::npos) {
            throw std::runtime_error("instructions bounds");
        }
        std::vector<common_chat_tool> tools;
        if (!merged["tools"].is_array() || merged["tools"].size() > 32) {
            throw std::runtime_error("tool bounds");
        }
        for (auto & t : merged["tools"]) {
            if (t.value("type", "") != "function") {
                throw std::runtime_error("function tools only");
            }
            auto name = t.at("name").get<std::string>();
            if (name.empty() || name.size() > 256) {
                throw std::runtime_error("tool name bounds");
            }
            tools.push_back({ name, t.value("description", ""), t.at("parameters").dump() });
        }
        if (!audio.empty() || speaking) {
            throw std::runtime_error("cannot reconfigure during capture");
        }
        server_vad = !detection.is_null();
        if (server_vad) {
            threshold = detection["threshold"];
            silence_ms = detection["silence_duration_ms"];
            auto_response = detection["create_response"];
            auto_interrupt = detection["interrupt_response"];
        }
        config                           = std::move(merged);
        request.chat.tools               = std::move(tools);
        request.chat.messages[0].content = instructions;
        if (request.chat.messages.size() == 1) {
            if (worker.joinable()) {
                worker.join();
            }
            busy = true;
            brain.cancelled = false;
            worker = std::thread([this, input = request.chat, event_id]() {
                const auto start = std::chrono::steady_clock::now();
                std::string failure;
                try {
                    brain.warm_prefix(input);
                } catch (const std::exception & e) {
                    failure = e.what();
                }
                std::lock_guard<std::mutex> guard(state);
                busy = false;
                if (!connected) {
                    return;
                }
                if (!failure.empty()) {
                    fail(failure, event_id, "prefix_warm_failed");
                    return;
                }
                const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start).count();
                std::cerr << "session prefix_warm_ms=" << ms << "\n";
                send({{"type", "session.updated"}, {"session", config}});
            });
        } else {
            send({{"type", "session.updated"}, {"session", config}});
        }
    }

    void trim_history() {
        for (const auto & call : pending_calls) { if (!call.second) { return; } }
        std::map<std::string, size_t> pending_rows;
        // Parakeet emits one row per 80 ms plus the learned end row.
        for (const auto & input : unencoded) { pending_rows[input.first] = (input.second.size() + 1279) / 1280 + 2; }
        for (;;) {
            const size_t tokens = brain.prompt_tokens(request, pending_rows);
            size_t users = 0, cut = 0;
            for (size_t i = 1; i < request.chat.messages.size(); ++i) {
                if (request.chat.messages[i].role == "user") { if (++users == 2) { cut = i; } }
            }
            if (tokens + 512 <= brain_session::context_tokens &&
                request.chat.messages.size() < brain_session::max_messages - 2 &&
                request.audio_rows.size() + unencoded.size() < brain_session::max_audio_segments - 1) { return; }
            if (cut == 0) {
                // Preserve the active tool call/result pair, but cap oversized output visibly.
                auto largest = request.chat.messages.end();
                for (auto it = request.chat.messages.begin(); it != request.chat.messages.end(); ++it) {
                    if (it->role == "tool" && it->content.size() > 256 &&
                        (largest == request.chat.messages.end() || it->content.size() > largest->content.size())) { largest = it; }
                }
                if (largest == request.chat.messages.end()) { throw std::runtime_error("current turn exceeds context budget"); }
                size_t keep = largest->content.size() / 2;
                while (keep > 0 && (static_cast<unsigned char>(largest->content[keep]) & 0xc0) == 0x80) { --keep; }
                largest->content.resize(keep);
                largest->content += "\n[Tool output truncated to fit conversation context]";
                std::cerr << "history_tool_output_truncated bytes=" << keep << "\n";
                continue;
            }
            erase_history_prefix(cut);
        }
    }

    void erase_history_prefix(size_t cut) {
        for (size_t i = 1; i < cut; ++i) {
            const auto & message = request.chat.messages[i];
            for (auto it = request.audio_rows.begin(); it != request.audio_rows.end();) {
                if (message.content.find(it->first) != std::string::npos) { it = request.audio_rows.erase(it); } else { ++it; }
            }
            for (auto it = unencoded.begin(); it != unencoded.end();) {
                if (message.content.find(it->first) != std::string::npos) { it = unencoded.erase(it); } else { ++it; }
            }
            for (auto it = request.images.begin(); it != request.images.end();) {
                if (message.content.find(it->first) != std::string::npos) { it = request.images.erase(it); } else { ++it; }
            }
            for (const auto & call : message.tool_calls) { pending_calls.erase(call.id); }
        }
        for (auto it = audio_messages.begin(); it != audio_messages.end();) {
            if (it->second < cut) { emitted_samples.erase(it->first); truncated.erase(it->first); phrase_history.erase(it->first); heard_samples.erase(it->first); it = audio_messages.erase(it); }
            else { it->second -= cut - 1; ++it; }
        }
        request.chat.messages.erase(request.chat.messages.begin() + 1, request.chat.messages.begin() + cut);
        std::cerr << "history_rollover removed_messages=" << cut - 1 << " \n";
    }

    void commit_audio() {
        if (audio.size() < 4800 || request.audio_rows.size() + unencoded.size() >= brain_session::max_audio_segments || request.chat.messages.size() >= brain_session::max_messages) {
            throw std::runtime_error("audio/history bounds");
        }
        std::vector<float> pcm(audio.size() / 3 + 64);
        auto frames = ma_convert_frames(pcm.data(), pcm.size(), ma_format_f32, 1, 16000, audio.data(), audio.size() / 2,
                                        ma_format_s16, 1, 24000);
        if (frames == 0 || frames > 16000 * 20) {
            throw std::runtime_error("resampling failed");
        }
        pcm.resize(frames);
        auto        id     = input_id.empty() ? next_id("item_") : input_id;
        input_id.clear();
        std::string marker = "[FRANKIE_AUDIO_" + id + "]";
        if (unencoded.size() + request.audio_rows.size() >= brain_session::max_audio_segments) {
            throw std::runtime_error("audio history full");
        }
        unencoded.emplace(marker, std::move(pcm));
        common_chat_msg message;
        message.role    = "user";
        message.content = marker;
        request.chat.messages.push_back(message);
        audio.clear();
        send({
            { "type",             "input_audio_buffer.committed" },
            { "item_id",          id                             },
            { "previous_item_id", nullptr                        }
        });
        send({
            { "type",             "conversation.item.created"                                              },
            { "previous_item_id", nullptr                                                                  },
            { "item",
             { { "id", id },
                { "type", "message" },
                { "role", "user" },
                { "status", "completed" },
                { "content", json::array({ { { "type", "input_audio" }, { "transcript", nullptr } } }) } } }
        });
    }

    void append_audio(const json & event) {
        auto encoded = event.at("audio").get<std::string>();
        if (encoded.size() > 64000) {
            throw std::runtime_error("audio chunk too large");
        }
        auto pcm = base64::decode(encoded);
        if (pcm.empty() || pcm.size() % 2 || pcm.size() > 48000 || audio.size() + pcm.size() > 24000 * 2 * 20) {
            throw std::runtime_error("audio buffer bounds");
        }
        if (!server_vad) {
            audio.insert(audio.end(), pcm.begin(), pcm.end());
            return;
        }
        for (size_t offset = 0; offset < pcm.size();) {
            const size_t count = std::min(size_t(1536) - vad_frame.size(), pcm.size() - offset);
            vad_frame.insert(vad_frame.end(), pcm.begin() + offset, pcm.begin() + offset + count);
            offset += count;
            if (vad_frame.size() != 1536) {
                continue;
            }
            std::array<float, 512> samples{};
            if (ma_convert_frames(samples.data(), 512, ma_format_f32, 1, 16000, vad_frame.data(), 768,
                                  ma_format_s16, 1, 24000) != 512) {
                throw std::runtime_error("VAD resampling failed");
            }
            audio.insert(audio.end(), vad_frame.begin(), vad_frame.end());
            vad_frame.clear();
            input_samples += 768;
            const bool speech = vad->process(samples) >= threshold;
            speech_frames = speech ? speech_frames + 1 : 0;
            if (!speaking && speech_frames >= 3) {
                speaking = true;
                silence_frames = 0;
                input_id = next_id("item_");
                precommit_samples = 0;
                send({{"type", "input_audio_buffer.speech_started"}, {"item_id", input_id},
                      {"audio_start_ms", (input_samples - audio.size() / 2) / 24}});
                if (busy && auto_interrupt) {
                    brain.cancelled = true;
                    mouth.cancelled = true;
                }
            }
            if (speaking) {
                silence_frames = speech ? 0 : silence_frames + 1;
                if (tentative && !tentative->committed) {
                    tentative->resumed_frames = speech ? tentative->resumed_frames + 1 : 0;
                    if (tentative->resumed_frames >= 2 && !tentative->abandoned) {
                        tentative->abandoned = true;
                        brain.cancelled = true;
                        mouth.cancelled = true;
                        changed.notify_all();
                        std::cerr << input_id << " stage=speculation_aborted\n";
                    }
                    commit_tentative();
                    if (audio.size() >= 24000 * 2 * 19) {
                        throw std::runtime_error("speculative utterance duration limit");
                    }
                } else if (silence_frames * 32 >= 80 && !busy && unencoded.empty() &&
                           std::all_of(pending_calls.begin(), pending_calls.end(), [](const auto & c) { return c.second; })) {
                    respond(json::object(), true);
                } else if (silence_frames * 32 >= silence_ms || audio.size() >= 24000 * 2 * 19) {
                    speaking = false;
                    std::cerr << input_id << " stage=speech_stopped wall_us=" << std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count() << "\n";
                    send({{"type", "input_audio_buffer.speech_stopped"}, {"item_id", input_id},
                          {"audio_end_ms", input_samples / 24}});
                    commit_audio();
                    if (auto_response && !busy) {
                        respond(json::object());
                    }
                }
            } else if (audio.size() > 15360) {
                audio.erase(audio.begin(), audio.end() - 15360);
            }
        }
        precommit_audio();
    }

    void truncate_audio(const json & event) {
        const auto id = event.at("item_id").get<std::string>();
        const int64_t end = event.at("audio_end_ms").get<int64_t>();
        auto emitted = emitted_samples.find(id);
        if (event.at("content_index") != 0 || end < 0 || emitted == emitted_samples.end() || uint64_t(end) > emitted->second / 24) {
            throw std::runtime_error("invalid audio truncation");
        }
        const size_t samples = size_t(end) * 24;
        if (heard_samples.count(id) && samples > heard_samples.at(id)) {
            throw std::runtime_error("cannot extend truncated audio");
        }
        heard_samples[id] = samples;
        truncated.insert(id);
        merge_false_start(id, samples);
        auto found = audio_messages.find(id);
        if (found != audio_messages.end()) {
            request.chat.messages[found->second].content = heard_text(id);
        }
        send({{"type", "conversation.item.truncated"}, {"item_id", id}, {"content_index", 0}, {"audio_end_ms", end}});
    }

    void create_item(json item) {
        if (request.chat.messages.size() >= brain_session::max_messages) {
            throw std::runtime_error("history full");
        }
        common_chat_msg message;
        auto            type = item.at("type").get<std::string>();
        if (type == "function_call_output") {
            auto id    = item.at("call_id").get<std::string>();
            auto found = pending_calls.find(id);
            if (found == pending_calls.end() || found->second) {
                throw std::runtime_error("unknown or duplicate tool result");
            }
            message.role         = "tool";
            message.tool_call_id = id;
            message.content      = item.at("output").get<std::string>();
            for (auto & m : request.chat.messages) {
                for (auto & c : m.tool_calls) {
                    if (c.id == id) {
                        message.tool_name = c.name;
                    }
                }
            }
            if (message.content.size() > 256 * 1024) {
                throw std::runtime_error("tool result bounds");
            }
            // Tool output is text, never a reference to internal media rows.
            for (size_t offset = 0; (offset = message.content.find("[FRANKIE_", offset)) != std::string::npos;) {
                message.content.replace(offset, 9, "[FRANKIE-");
                offset += 9;
            }
            found->second = true;
        } else if (type == "message" && item.value("role", "") == "user") {
            message.role = "user";
            std::map<std::string, brain_session::image_ptr> images;
            if (!item.at("content").is_array() || item["content"].empty() || item["content"].size() > 16) {
                throw std::runtime_error("content bounds");
            }
            for (auto & part : item.at("content")) {
                const auto kind = part.value("type", "");
                if (kind == "input_text") {
                    auto text = part.at("text").get<std::string>();
                    if (text.find("[FRANKIE_") != std::string::npos) {
                        throw std::runtime_error("reserved media marker");
                    }
                    message.content += text;
                } else if (kind == "input_image") {
                    if (request.images.size() + images.size() >= 4) {
                        throw std::runtime_error("image history full");
                    }
                    auto url    = part.at("image_url").get<std::string>();
                    auto comma  = url.find(',');
                    auto header = url.substr(0, comma);
                    if (comma == std::string::npos ||
                        (header != "data:image/png;base64" && header != "data:image/jpeg;base64")) {
                        throw std::runtime_error("inline PNG or JPEG required");
                    }
                    auto bytes  = base64::decode(url.substr(comma + 1));
                    auto image  = brain.decode_image({ bytes.begin(), bytes.end() });
                    auto marker = "[FRANKIE_IMAGE_" + next_id("image_") + "]";
                    images.emplace(marker, std::move(image));
                    message.content += marker;
                } else {
                    throw std::runtime_error("unsupported content; use input_audio_buffer for audio");
                }
                if (message.content.size() > 16384) {
                    throw std::runtime_error("text bounds");
                }
            }
            request.images.insert(images.begin(), images.end());
        } else {
            throw std::runtime_error("unsupported conversation item");
        }
        item["id"]     = next_id("item_");
        item["status"] = "completed";
        request.chat.messages.push_back(message);
        send({
            { "type",             "conversation.item.created" },
            { "previous_item_id", nullptr                     },
            { "item",             item                        }
        });
    }

    void respond(const json & event, bool speculate = false) {
        if (event.contains("response") && !event["response"].empty()) {
            throw std::runtime_error("response overrides not implemented in prototype");
        }
        for (auto & call : pending_calls) {
            if (!call.second) {
                throw std::runtime_error("tool result pending");
            }
        }
        if (worker.joinable()) {
            worker.join();
        }
        response_id             = next_id("resp_");
        const auto id           = response_id;
        auto       snapshot     = request;
        auto       captured     = unencoded;
        std::shared_ptr<speculative_turn> turn;
        if (speculate) {
            turn = std::make_shared<speculative_turn>();
            turn->input = input_id;
            turn->user_index = request.chat.messages.size();
            const auto marker = "[FRANKIE_AUDIO_" + input_id + "]";
            common_chat_msg user;
            user.role = "user";
            user.content = marker;
            snapshot.chat.messages.push_back(user);
            const size_t speech_bytes = audio.size() - std::min(audio.size(), size_t(silence_frames) * 1536);
            turn->capture.assign(audio.begin(), audio.begin() + speech_bytes);
            std::vector<float> pcm(speech_bytes / 3 + 64);
            const auto n = ma_convert_frames(pcm.data(), pcm.size(), ma_format_f32, 1, 16000,
                                             audio.data(), speech_bytes / 2, ma_format_s16, 1, 24000);
            if (!n || n > 16000 * 20) { throw std::runtime_error("speculative input bounds"); }
            pcm.resize(n);
            captured.emplace(marker, std::move(pcm));
            std::cerr << input_id << " stage=speculation_started silence_ms=" << silence_frames * 32 << "\n";
        }
        busy = true;
        brain.cancelled = false;
        mouth.cancelled = false;
        if (turn) { tentative = turn; }
        bool       audio_output = config["output_modalities"][0] == "audio";
        if (!turn) { send({
            { "type",     "response.created"                                                         },
            { "response", { { "id", id }, { "status", "in_progress" }, { "output", json::array() } } }
        }); }
        worker = std::thread([this, id, snapshot = std::move(snapshot), captured = std::move(captured), audio_output, turn]() mutable {
            std::optional<brain_session::saved_state> saved;
            std::map<std::string, std::string> transcripts;
            bool input_published = false;
            auto publish_input = [&] {
                if (input_published) { return; }
                for (const auto & input : captured) {
                    request.audio_rows[input.first] = snapshot.audio_rows.at(input.first);
                    unencoded.erase(input.first);
                    const std::string prefix = "[FRANKIE_AUDIO_";
                    const auto user_id = input.first.substr(prefix.size(), input.first.size() - prefix.size() - 1);
                    send({{"type", "conversation.item.input_audio_transcription.completed"}, {"item_id", user_id},
                          {"content_index", 0}, {"transcript", transcripts.at(input.first)}});
                }
                input_published = true;
            };
            std::string item_id;
            bool history_saved = false;
            json        output = json::array();
            std::string status = "completed", reason;
            const auto worker_start = std::chrono::steady_clock::now();
            auto stage = [&](const char * name) {
                const auto ms = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - worker_start).count() / 1000.0;
                std::lock_guard<std::mutex> guard(state);
                std::cerr << id << " stage=" << name << " worker_ms=" << ms << "\n";
            };
            try {
                // Finish the input pass without holding the socket reader's state lock.
                settle_input();
                restore_merged_input();
                if (turn) { saved = brain.suspend_state(); }
                if (!turn) {
                    std::lock_guard<std::mutex> guard(state);
                    // New capture may commit while the prior input pass settles.
                    // Budget only the response's captured prefix, then retain later user items.
                    std::vector<common_chat_msg> later(request.chat.messages.begin() + snapshot.chat.messages.size(), request.chat.messages.end());
                    request.chat.messages.resize(snapshot.chat.messages.size());
                    try {
                        trim_history();
                        snapshot = request;
                    } catch (...) {
                        request.chat.messages.insert(request.chat.messages.end(), later.begin(), later.end());
                        throw;
                    }
                    request.chat.messages.insert(request.chat.messages.end(), later.begin(), later.end());
                }
                stage("ear_start");
                for (const auto & input : captured) {
                    std::string transcript;
                    auto rows = brain.encode_audio(input.second, &transcript);
                    brain.finish_audio(snapshot, input.first, rows);
                    snapshot.audio_rows.emplace(input.first, std::move(rows));
                    transcripts.emplace(input.first, std::move(transcript));
                }
                if (turn && brain.prompt_tokens(snapshot, {}) + 512 > brain_session::context_tokens) {
                    throw std::runtime_error("speculative context budget exceeded");
                }
                stage("ear_done");
                if (!turn) {
                    std::lock_guard<std::mutex> guard(state);
                    publish_input();
                }
                {
                    std::lock_guard<std::mutex> lock(state);
                    item_id = next_id("item_");
                }
                json fields = {{"response_id", id}, {"item_id", item_id}, {"output_index", 0}, {"content_index", 0}};
                bool audio_started = false;
                const auto inference_start = std::chrono::steady_clock::now();
                auto elapsed_ms = [&] { return std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - inference_start).count(); };
                auto emit_audio = [&](const float * samples, size_t count) {
                    await_authorization(turn);
                    std::vector<int16_t> pcm(count);
                    auto frames = ma_convert_frames(pcm.data(), pcm.size(), ma_format_s16, 1, 24000,
                                                    samples, count, ma_format_f32, 1, 24000);
                    if (frames != count || pcm.empty()) {
                        throw std::runtime_error("output PCM conversion failed");
                    }
                    std::lock_guard<std::mutex> guard(state);
                    if (!connected || brain.cancelled.load()) {
                        throw std::runtime_error("cancelled");
                    }
                    publish_input();
                    if (!audio_started) {
                        send({{"type", "response.output_item.added"}, {"response_id", id}, {"output_index", 0},
                              {"item", {{"id", item_id}, {"type", "message"}, {"role", "assistant"},
                                        {"status", "in_progress"}, {"content", json::array()}}}});
                        auto added = fields;
                        added["type"] = "response.content_part.added";
                        added["part"] = {{"type", "output_audio"}, {"transcript", ""}};
                        send(added);
                        std::cerr << id << " first_pcm_ms=" << elapsed_ms() << "\n";
                        audio_started = true;
                        if (turn) {
                            turn->released = std::chrono::steady_clock::now();
                            turn->output = item_id;
                            released_turn = turn;
                        }
                        emitted_samples.emplace(item_id, 0);
                    }
                    if (emitted_samples[item_id] + count > 24000 * 30) {
                        throw brain_session::output_limit("max_output_audio");
                    }
                    auto delta = fields;
                    delta["type"] = "response.output_audio.delta";
                    delta["delta"] = base64::encode(reinterpret_cast<const char *>(pcm.data()), pcm.size() * sizeof(int16_t));
                    emitted_samples[item_id] += count;
                    send(delta);
                };
                const bool pipelined = audio_output;
                std::unique_ptr<speech_stream> speech;
                if (pipelined) {
                    speech = std::make_unique<speech_stream>(brain, mouth, emit_audio, stage, [&](const std::string & text) {
                        std::lock_guard<std::mutex> guard(state);
                        auto & phrases = phrase_history[item_id];
                        if (phrases.size() >= 512) { throw std::runtime_error("phrase history limit"); }
                        phrases.push_back({emitted_samples.at(item_id), text});
                    });
                }
                auto result = brain.generate(snapshot, speech ? brain_session::stream_callback([&](const auto & part, bool last) {
                    speech->feed(part, last);
                }) : brain_session::stream_callback{}, stage);
                {
                    std::lock_guard<std::mutex> guard(state);
                    std::cerr << id << " brain_done_ms=" << elapsed_ms() << "\n";
                }
                if (speech) {
                    speech->finish();
                }
                await_authorization(turn);
                std::unique_lock<std::mutex> lock(state);
                if (!connected || brain.cancelled.load()) {
                    throw std::runtime_error("cancelled");
                }
                publish_input();
                if (audio_started || result.message.tool_calls.empty()) {
                    json part = audio_output ? json{{"type", "output_audio"}, {"transcript", result.message.content}} :
                                               json{{"type", "output_text"}, {"text", result.message.content}};
                    output.push_back({{"id", item_id}, {"type", "message"}, {"role", "assistant"},
                                      {"status", "completed"}, {"content", json::array({part})}});
                    if (!audio_output) {
                        auto initial = output.back();
                        initial["status"] = "in_progress";
                        initial["content"] = json::array();
                        send({{"type", "response.output_item.added"}, {"response_id", id}, {"output_index", 0}, {"item", initial}});
                        auto added = fields;
                        added["type"] = "response.content_part.added";
                        added["part"] = {{"type", "output_text"}, {"text", ""}};
                        send(added);
                    } else {
                        auto done = fields;
                        done["type"] = "response.output_audio.done";
                        send(done);
                    }
                    auto text = fields;
                    text["type"] = audio_output ? "response.output_audio_transcript.delta" : "response.output_text.delta";
                    text["delta"] = result.message.content;
                    send(text);
                    auto done = fields;
                    done["type"] = audio_output ? "response.output_audio_transcript.done" : "response.output_text.done";
                    done[audio_output ? "transcript" : "text"] = result.message.content;
                    send(done);
                    auto content = fields;
                    content["type"] = "response.content_part.done";
                    content["part"] = part;
                    send(content);
                    if (audio_output) {
                        audio_messages[item_id] = request.chat.messages.size();
                        if (truncated.count(item_id)) {
                            result.message.content = heard_text(item_id);
                        }
                    }
                }
                if (!result.message.tool_calls.empty()) {
                    if (turn) { turn->tool_released = true; }
                    if (released_turn) { released_turn->tool_released = true; }
                    auto & call = result.message.tool_calls[0];
                    call.id = next_id("call_");
                    pending_calls.emplace(call.id, false);
                    const auto tool_item = output.empty() ? item_id : next_id("item_");
                    const size_t index = output.size();
                    output.push_back({{"id", tool_item}, {"type", "function_call"}, {"status", "completed"},
                                      {"call_id", call.id}, {"name", call.name}, {"arguments", call.arguments}});
                    auto initial = output.back();
                    initial["status"] = "in_progress";
                    initial["arguments"] = "";
                    send({{"type", "response.output_item.added"}, {"response_id", id}, {"output_index", index}, {"item", initial}});
                    send({{"type", "response.function_call_arguments.delta"}, {"response_id", id}, {"item_id", tool_item},
                          {"output_index", index}, {"delta", call.arguments}});
                    send({{"type", "response.function_call_arguments.done"}, {"response_id", id}, {"item_id", tool_item},
                          {"output_index", index}, {"call_id", call.id}, {"name", call.name}, {"arguments", call.arguments}});
                }
                request.chat.messages.push_back(result.message);
                history_saved = true;
                for (size_t i = 0; i < output.size(); ++i) {
                    send({{"type", "response.output_item.done"}, {"response_id", id}, {"output_index", i}, {"item", output[i]}});
                }
            } catch (const brain_session::output_limit & e) {
                status = brain.cancelled.load() ? "cancelled" : "incomplete";
                reason = e.what();
                // Discard unpublished calls and decoded-but-unsaid recurrent state.
                brain.reset();
                try {
                    await_authorization(turn);
                    std::lock_guard<std::mutex> guard(state);
                    publish_input();
                } catch (const std::exception & failure) {
                    status = brain.cancelled.load() ? "cancelled" : "failed";
                    reason = failure.what();
                }
            } catch (const std::exception & e) {
                status = brain.cancelled.load() ? "cancelled" : "failed";
                reason = e.what();
            }
            std::lock_guard<std::mutex> lock(state);
            const bool restore = turn && (!turn->authorized || turn->merge_requested);
            if (restore && saved) {
                try { brain.restore_state(std::move(*saved)); }
                catch (const std::exception & e) { status = "failed"; reason = e.what(); }
            }
            if (turn && !turn->authorized) {
                if (status == "failed") { fail(reason, "", "speculation_failed"); }
                std::cerr << turn->input << " stage=speculation_restored\n";
                tentative.reset();
                busy = false;
                return;
            }
            if (turn) {
                tentative.reset();
                if (!restore && saved && !turn->tool_released) { turn->saved = std::move(saved); }
            }
            if (!(turn && turn->merge_requested) && !history_saved && emitted_samples.count(item_id)) {
                common_chat_msg interrupted;
                interrupted.role = "assistant";
                interrupted.content = status == "incomplete" ? "[Response stopped at its output limit; no tool call was executed.]" : heard_text(item_id);
                audio_messages[item_id] = request.chat.messages.size();
                request.chat.messages.push_back(interrupted);
            }
            json                        response = {
                { "id",     id                                             },
                { "status", status                                         },
                { "output", status == "completed" ? output : json::array() }
            };
            if (status == "incomplete") {
                response["status_details"] = {{"type", "incomplete"}, {"reason", reason}};
                std::cerr << id << " stage=output_limited reason=" << reason << "\n";
            }
            if (status == "failed") {
                std::cerr << id << " inference failed: " << reason << "\n";
                response["status_details"] = {
                    { "type",  "failed"                                                 },
                    { "error", { { "code", "inference_error" }, { "message", reason } } }
                };
            }
            send({
                { "type",     "response.done" },
                { "response", response        }
            });
            busy = false;
        });
    }

    void run() {
        std::string bytes;
        while (socket.read(bytes) == httplib::ws::Text) {
            std::lock_guard<std::mutex> lock(state);
            std::string                 client_event_id;
            try {
                if (bytes.size() > 1024 * 1024) {
                    throw std::runtime_error("event too large");
                }
                auto event = json::parse(bytes);
                if (event.contains("event_id")) {
                    client_event_id = event.at("event_id").get<std::string>();
                    if (client_event_id.size() > 256) {
                        client_event_id.clear();
                        throw std::runtime_error("event_id too long");
                    }
                }
                auto type = event.at("type").get<std::string>();
                if (type == "response.cancel") {
                    if (!busy && event.value("response_id", "") != response_id) {
                        throw std::runtime_error("no active response");
                    }
                    if (event.contains("response_id") && event["response_id"] != response_id) {
                        throw std::runtime_error("response id mismatch");
                    }
                    // A delayed cancel for a finished response must not abort new input prefill.
                    if (busy) {
                        brain.cancelled = true;
                        mouth.cancelled = true;
                    }
                    continue;
                }
                if (type == "input_audio_buffer.append") {
                    append_audio(event);
                    continue;
                }
                if (type == "conversation.item.truncate") {
                    truncate_audio(event);
                    continue;
                }
                if (type == "input_audio_buffer.commit" && tentative) {
                    throw std::runtime_error("cannot manually commit a speculative capture");
                }
                if (type == "response.create" && tentative && tentative->committed && !tentative->authorized) {
                    if (event.contains("response") && !event["response"].empty()) {
                        throw std::runtime_error("response overrides not implemented in prototype");
                    }
                    authorize_tentative();
                    continue;
                }
                if (type == "input_audio_buffer.clear" && tentative && !tentative->committed) {
                    tentative->abandoned = true;
                    brain.cancelled = true;
                    mouth.cancelled = true;
                    audio.clear();
                    vad_frame.clear();
                    vad->reset();
                    speaking = false;
                    speech_frames = 0;
                    silence_frames = 0;
                    precommit_samples = 0;
                    input_id.clear();
                    changed.notify_all();
                    send({{"type", "input_audio_buffer.cleared"}});
                    continue;
                }
                if (busy && type != "input_audio_buffer.commit") {
                    fail("this prototype cannot mutate input during a response", client_event_id,
                         "prototype_response_active");
                    continue;
                }
                if (type != "response.create" && type != "input_audio_buffer.commit") { settle_input(); }
                if (type == "session.update") {
                    update(event.at("session"), client_event_id);
                } else if (type == "input_audio_buffer.clear") {
                    audio.clear();
                    vad_frame.clear();
                    vad->reset();
                    speaking = false;
                    speech_frames = 0;
                    silence_frames = 0;
                    precommit_samples = 0;
                    input_id.clear();
                    send({
                        { "type", "input_audio_buffer.cleared" }
                    });
                } else if (type == "input_audio_buffer.commit") {
                    commit_audio();
                } else if (type == "conversation.item.create") {
                    create_item(event.at("item"));
                } else if (type == "response.create") {
                    respond(event);
                } else {
                    throw std::runtime_error("unsupported event: " + type);
                }
            } catch (const std::exception & e) {
                fail(e.what(), client_event_id);
            }
        }
    }
};

int main(int argc, char ** argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error("package port required");
        }
        const char * token = std::getenv("FRANKIE_REALTIME_TOKEN");
        if (!token || std::strlen(token) < 16) {
            throw std::runtime_error("FRANKIE_REALTIME_TOKEN must contain at least 16 bytes");
        }
        std::string auth = "Bearer " + std::string(token);
        common_init();
        brain_session     brain(argv[1]);
        mouth_session     mouth(argv[1]);
        httplib::Server   server;
        std::atomic<bool> occupied{ false };
        server.set_payload_max_length(1024 * 1024);
        server.set_read_timeout(30);
        server.set_write_timeout(10);
        server.Get("/health", [](const auto &, auto & response) { response.set_content("ready", "text/plain"); });
        server.WebSocket("/v1/realtime", [&](const auto &, auto & socket) {
            if (occupied.exchange(true)) {
                socket.close();
                return;
            }
            try {
                realtime_session session(brain, mouth, socket, argv[1]);
                session.run();
            } catch (const std::exception & e) {
                std::cerr << "session closed: " << e.what() << "\n";
            }
            occupied = false;
        });
        server.set_pre_routing_handler([&](const auto & request, auto & response) {
            if (request.path == "/v1/realtime") {
                if (request.get_header_value("Authorization") != auth) {
                    response.status = 401;
                    return httplib::Server::HandlerResponse::Handled;
                }
                if (occupied.load()) {
                    response.status = 503;
                    return httplib::Server::HandlerResponse::Handled;
                }
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
        std::cerr << "Realtime prototype ready on loopback\n";
        if (!server.listen("127.0.0.1", std::stoi(argv[2]))) {
            throw std::runtime_error("listen failed");
        }
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
