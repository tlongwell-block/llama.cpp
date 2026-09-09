#include "base64.hpp"
#include "brain-session.h"
#include "common.h"
#include "cpp-httplib/httplib.h"
#include "mouth-session.h"
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
#include <iostream>
#include <mutex>
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
    std::mutex                  state;
    bool                        busy = false, connected = true;
    size_t                      serial = 0;
    std::string                 response_id;

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

    realtime_session(brain_session & b, mouth_session & m, httplib::ws::WebSocket & s) : brain(b), mouth(m), socket(s) {
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
    }

    void update(const json & value) {
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
        if (a.at("input").contains("turn_detection") && !a["input"]["turn_detection"].is_null()) {
            throw std::runtime_error("this prototype supports manual turns only");
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
        config                           = std::move(merged);
        request.chat.tools               = std::move(tools);
        request.chat.messages[0].content = instructions;
        send({
            { "type",    "session.updated" },
            { "session", config            }
        });
    }

    void commit_audio() {
        if (audio.size() < 4800 || request.audio_rows.size() >= 8 || request.chat.messages.size() >= 30) {
            throw std::runtime_error("audio/history bounds");
        }
        std::vector<float> pcm(audio.size() / 3 + 64);
        auto frames = ma_convert_frames(pcm.data(), pcm.size(), ma_format_f32, 1, 16000, audio.data(), audio.size() / 2,
                                        ma_format_s16, 1, 24000);
        if (frames == 0 || frames > 16000 * 20) {
            throw std::runtime_error("resampling failed");
        }
        pcm.resize(frames);
        auto        rows   = brain.encode_audio(pcm);
        auto        id     = next_id("item_");
        std::string marker = "[FRANKIE_AUDIO_" + id + "]";
        request.audio_rows.emplace(marker, std::move(rows));
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

    void create_item(json item) {
        if (request.chat.messages.size() >= 30) {
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
            if (message.content.size() > 16384 || message.content.find("[FRANKIE_") != std::string::npos) {
                throw std::runtime_error("tool result bounds");
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

    void respond(const json & event) {
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
        busy                    = true;
        brain.cancelled         = false;
        mouth.cancelled         = false;
        response_id             = next_id("resp_");
        const auto id           = response_id;
        auto       snapshot     = request;
        bool       audio_output = config["output_modalities"][0] == "audio";
        send({
            { "type",     "response.created"                                                         },
            { "response", { { "id", id }, { "status", "in_progress" }, { "output", json::array() } } }
        });
        worker = std::thread([this, id, snapshot = std::move(snapshot), audio_output]() {
            json        output = json::array();
            std::string status = "completed", reason;
            try {
                auto               result = brain.generate(snapshot);
                std::vector<float> spoken_pcm;
                if (result.message.tool_calls.empty() && audio_output) {
                    spoken_pcm = mouth.speak(result);
                }
                std::lock_guard<std::mutex> lock(state);
                if (!connected || brain.cancelled.load()) {
                    throw std::runtime_error("cancelled");
                }
                auto item_id = next_id("item_");
                if (!result.message.tool_calls.empty()) {
                    auto & call = result.message.tool_calls[0];
                    call.id     = next_id("call_");
                    pending_calls.emplace(call.id, false);
                    output.push_back({
                        { "id",        item_id         },
                        { "type",      "function_call" },
                        { "status",    "completed"     },
                        { "call_id",   call.id         },
                        { "name",      call.name       },
                        { "arguments", call.arguments  }
                    });
                    auto initial         = output[0];
                    initial["status"]    = "in_progress";
                    initial["arguments"] = "";
                    send({
                        { "type",         "response.output_item.added" },
                        { "response_id",  id                           },
                        { "output_index", 0                            },
                        { "item",         initial                      }
                    });
                    send({
                        { "type",         "response.function_call_arguments.delta" },
                        { "response_id",  id                                       },
                        { "item_id",      item_id                                  },
                        { "output_index", 0                                        },
                        { "delta",        call.arguments                           }
                    });
                    send({
                        { "type",         "response.function_call_arguments.done" },
                        { "response_id",  id                                      },
                        { "item_id",      item_id                                 },
                        { "output_index", 0                                       },
                        { "call_id",      call.id                                 },
                        { "name",         call.name                               },
                        { "arguments",    call.arguments                          }
                    });
                } else {
                    json part = audio_output ?
                                    json{
                                        { "type",       "output_audio"         },
                                        { "transcript", result.message.content }
                    } :
                                    json{ { "type", "output_text" }, { "text", result.message.content } };
                    output.push_back({
                        { "id",      item_id               },
                        { "type",    "message"             },
                        { "role",    "assistant"           },
                        { "status",  "completed"           },
                        { "content", json::array({ part }) }
                    });
                    auto initial       = output[0];
                    initial["status"]  = "in_progress";
                    initial["content"] = json::array();
                    send({
                        { "type",         "response.output_item.added" },
                        { "response_id",  id                           },
                        { "output_index", 0                            },
                        { "item",         initial                      }
                    });
                    json fields = {
                        { "response_id",   id      },
                        { "item_id",       item_id },
                        { "output_index",  0       },
                        { "content_index", 0       }
                    };
                    auto added    = fields;
                    added["type"] = "response.content_part.added";
                    added["part"] = audio_output ?
                                        json{
                                            { "type",       "output_audio" },
                                            { "transcript", ""             }
                    } :
                                        json{ { "type", "output_text" }, { "text", "" } };
                    send(added);
                    auto text = fields;
                    text["type"] =
                        audio_output ? "response.output_audio_transcript.delta" : "response.output_text.delta";
                    text["delta"] = result.message.content;
                    send(text);
                    if (audio_output) {
                        std::vector<int16_t> pcm(spoken_pcm.size());
                        auto                 frames = ma_convert_frames(pcm.data(), pcm.size(), ma_format_s16, 1, 24000,
                                                                        spoken_pcm.data(), spoken_pcm.size(), ma_format_f32, 1, 24000);
                        if (frames != pcm.size() || pcm.empty()) {
                            throw std::runtime_error("output PCM conversion failed");
                        }
                        size_t       bytes = pcm.size() * sizeof(int16_t);
                        const auto * data  = reinterpret_cast<const char *>(pcm.data());
                        for (size_t offset = 0; offset < bytes; offset += 24000) {
                            auto delta     = fields;
                            delta["type"]  = "response.output_audio.delta";
                            delta["delta"] = base64::encode(data + offset, std::min(size_t(24000), bytes - offset));
                            send(delta);
                        }
                        auto done    = fields;
                        done["type"] = "response.output_audio.done";
                        send(done);
                    }
                    auto done    = fields;
                    done["type"] = audio_output ? "response.output_audio_transcript.done" : "response.output_text.done";
                    done[audio_output ? "transcript" : "text"] = result.message.content;
                    send(done);
                    auto content    = fields;
                    content["type"] = "response.content_part.done";
                    content["part"] = part;
                    send(content);
                }
                request.chat.messages.push_back(result.message);
                send({
                    { "type",         "response.output_item.done" },
                    { "response_id",  id                          },
                    { "output_index", 0                           },
                    { "item",         output[0]                   }
                });
            } catch (const std::exception & e) {
                status = brain.cancelled.load() ? "cancelled" : "failed";
                reason = e.what();
            }
            std::lock_guard<std::mutex> lock(state);
            json                        response = {
                { "id",     id                                             },
                { "status", status                                         },
                { "output", status == "completed" ? output : json::array() }
            };
            if (status == "failed") {
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
                    if (!busy) {
                        throw std::runtime_error("no active response");
                    }
                    if (event.contains("response_id") && event["response_id"] != response_id) {
                        throw std::runtime_error("response id mismatch");
                    }
                    brain.cancelled = true;
                    mouth.cancelled = true;
                    continue;
                }
                if (busy) {
                    fail("this prototype cannot mutate input during a response", client_event_id,
                         "prototype_response_active");
                    continue;
                }
                if (type == "session.update") {
                    update(event.at("session"));
                } else if (type == "input_audio_buffer.append") {
                    auto encoded = event.at("audio").get<std::string>();
                    if (encoded.size() > 64000) {
                        throw std::runtime_error("audio chunk too large");
                    }
                    auto pcm = base64::decode(encoded);
                    if (pcm.empty() || pcm.size() % 2 || pcm.size() > 48000 ||
                        audio.size() + pcm.size() > 24000 * 2 * 20) {
                        throw std::runtime_error("audio buffer bounds");
                    }
                    audio.insert(audio.end(), pcm.begin(), pcm.end());
                } else if (type == "input_audio_buffer.clear") {
                    audio.clear();
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
                realtime_session session(brain, mouth, socket);
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
