#include "brain-session.h"
#include "completions.h"
#include "mouth-session.h"
#include "realtime-server.h"
#include "common.h"
#include "cpp-httplib/httplib.h"
#include <cmath>
#include <cstring>
#include <iostream>

int main(int argc, char ** argv) {
    try {
        if (argc < 3 || std::strcmp(argv[1], "--help") == 0) {
            std::cerr << "Usage: llama-frankie-realtime PACKAGE PORT [--device cpu|gpu] [--threads N]\n"
                         "  [--text-encoder-device cpu|gpu]\n"
                         "  [--ctx-size N] [--cache-type q4_0|q8_0|f16] [--batch-size N] [--ubatch-size N]\n"
                         "  [--thinking none|minimal|low|medium|high|xhigh|max] [--speech-context-words N]\n"
                         "  [--http-thinking none|minimal|low|medium|high|xhigh|max]\n"
                         "  [--mtp-tokens 0..4] [--host ADDRESS] [--http-slots 0..8] [--http-ctx-size N]\n"
                         "  [--max-utterance-seconds N] [--max-output-audio-seconds N] [--max-output-tokens N|inf]\n"
                         "  [--voice WAV] [--voice-text-file TXT --voice-codes I32]\n"
                         "  [--side-scale N] [--presence-penalty N] [--expression GGUF]\n"
                         "  [--vap-model GGUF] [--bc-model GGUF]\n"
                         "  [--ear-model GGUF] [--talker-model GGUF] [--mouth-model GGUF]\n"
                         "--ctx-size is the total KV pool (default 131072). HTTP slots default to an equal share.\n"
                         "--http-ctx-size is per HTTP slot; voice uses the remainder of the total pool.\n"
                         "WAV alone uses the native reference encoder. Breeze transcribes it with the packaged ear.\n"
                         "Breeze accepts optional --voice-text-file; Qwen ICL requires matching text and frame-major codes.\n";
            return argc == 2 && std::strcmp(argv[1], "--help") == 0 ? 0 : 1;
        }
        frankie_options options;
        std::string host = "127.0.0.1";
        for (int i = 3; i < argc; ++i) {
            const std::string key = argv[i];
            if (++i == argc) { throw std::runtime_error("missing value for " + key); }
            const std::string value = argv[i];
            if (key == "--device") {
                if (value != "cpu" && value != "gpu") { throw std::runtime_error("device must be cpu or gpu (Metal/CUDA)"); }
                options.use_gpu = value == "gpu";
            } else if (key == "--text-encoder-device") {
                if (value != "cpu" && value != "gpu") { throw std::runtime_error("text encoder device must be cpu or gpu"); }
                options.text_encoder_gpu = value == "gpu";
            } else if (key == "--threads") { options.threads = int(frankie_unsigned(value)); }
            else if (key == "--host") { host = value; }
            else if (key == "--voice") { options.voice = value; }
            else if (key == "--voice-text-file") { options.voice_text = value; }
            else if (key == "--voice-codes") { options.voice_codes = value; }
            else if (key == "--expression") { options.expression = value; }
            else if (key == "--vap-model") { options.vap_model = value; }
            else if (key == "--bc-model") { options.bc_model = value; }
            else if (key == "--batch-size") {
                options.batch_size = frankie_unsigned(value);
                if (!options.batch_size) { throw std::runtime_error("batch size must be positive"); }
            }
            else if (key == "--ubatch-size") {
                options.ubatch_size = frankie_unsigned(value);
                if (!options.ubatch_size) { throw std::runtime_error("ubatch size must be positive"); }
            }
            else if (key == "--ctx-size") { options.context_tokens = frankie_unsigned(value); }
            else if (key == "--max-utterance-seconds") { options.max_utterance_seconds = frankie_unsigned(value); }
            else if (key == "--max-output-audio-seconds") { options.max_output_audio_seconds = frankie_unsigned(value); }
            else if (key == "--max-output-tokens") { options.max_output_tokens = value == "inf" ? 0 : frankie_unsigned(value); }
            else if (key == "--cache-type") { options.cache_type = value; }
            else if (key == "--thinking") { options.thinking = value; }
            else if (key == "--http-thinking") { options.http_thinking = value; }
            else if (key == "--speech-context-words") { options.speech_context_words = frankie_unsigned(value); }
            else if (key == "--mtp-tokens") { options.mtp_tokens = frankie_unsigned(value); }
            else if (key == "--http-slots") { options.http_slots = frankie_unsigned(value); }
            else if (key == "--http-ctx-size") { options.http_context_tokens = frankie_unsigned(value); }
            else if (key == "--ear-model") { options.ear_model = value; }
            else if (key == "--talker-model") { options.talker_model = value; }
            else if (key == "--mouth-model") { options.mouth_model = value; }
            else if (key == "--side-scale") { options.side_scale = frankie_float(value); }
            else if (key == "--presence-penalty") { options.presence_penalty = frankie_float(value); }
            else { throw std::runtime_error("unknown option: " + key); }
        }
        const auto port = frankie_unsigned(argv[2]);
        if (!port || port > 65535) { throw std::runtime_error("port must be 1..65535"); }
        if (host.empty()) { throw std::runtime_error("host must not be empty"); }
        options.resolve_context();
        if (options.mtp_tokens > 4 || options.batch_size > 8192 || options.ubatch_size > (options.batch_size ? options.batch_size : (options.use_gpu ? 512u : 128u)) ||
            options.max_utterance_seconds < 2 || options.max_utterance_seconds > 120 ||
            !options.max_output_audio_seconds || options.max_output_audio_seconds > 3600 ||
            options.threads < 1 || options.threads > 256 ||
            (options.cache_type != "q4_0" && options.cache_type != "q8_0" && options.cache_type != "f16") || !std::isfinite(options.side_scale) ||
            options.side_scale < 0 || options.side_scale > 2 || !std::isfinite(options.presence_penalty) ||
            options.presence_penalty < 0 || options.presence_penalty > 2 ||
            (!options.voice_codes.empty() && options.voice_text.empty()) ||
            (options.voice.empty() && (!options.voice_text.empty() || !options.voice_codes.empty()))) {
            throw std::runtime_error("invalid runtime options; voice text/codes require --voice, and codes require text");
        }
        frankie_thinking_budget(options.thinking);
        frankie_thinking_budget(options.http_thinking);
        const char * token = std::getenv("FRANKIE_REALTIME_TOKEN");
        if (!token || std::strlen(token) < 16) {
            throw std::runtime_error("FRANKIE_REALTIME_TOKEN must contain at least 16 bytes");
        }
        std::string auth = "Bearer " + std::string(token);
        common_init();
        brain_session     brain(argv[1], options);
        brain.report_memory();
        mouth_session     mouth(argv[1], options, [&](const std::vector<float> & pcm24) {
            return frankie_reference_transcript(brain, pcm24);
        });
        mouth.warmup();
        mouth.report_memory();
        httplib::Server   server;
        frankie_completions completions(brain, server, options.http_slots);
        frankie_realtime_routes realtime(server, brain, mouth, argv[1], options);
        server.set_payload_max_length(18 * 1024 * 1024);
        server.set_read_timeout(30);
        server.set_write_timeout(10);
        server.Get("/health", [](const auto &, auto & response) { response.set_content("ready", "text/plain"); });
        server.set_pre_routing_handler([&](const auto & request, auto & response) {
            if (request.path.rfind("/v1/", 0) == 0 && request.get_header_value("Authorization") != auth) {
                response.status = 401;
                return httplib::Server::HandlerResponse::Handled;
            }
            if (request.path == "/v1/realtime") {
                if (request.get_header_value("Authorization") != auth) {
                    response.status = 401;
                    return httplib::Server::HandlerResponse::Handled;
                }
                if (realtime.occupied()) {
                    response.status = 503;
                    return httplib::Server::HandlerResponse::Handled;
                }
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
        std::cerr << "Frankie ready on " << host << ":" << port << "\n";
        if (!server.listen(host, port)) {
            throw std::runtime_error("listen failed");
        }
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
