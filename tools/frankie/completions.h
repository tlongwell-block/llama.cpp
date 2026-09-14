#pragma once
#include "brain-session.h"
#include "brain-sampling.h"
#include "sampling.h"
#include "image-input.h"
#include "mtmd-helper-batch.h"
#include "cpp-httplib/httplib.h"
#include "nlohmann/json.hpp"

#include <condition_variable>
#include <cmath>
#include <deque>
#include <limits>
#include <thread>

// Experimental HTTP lane: sequence zero belongs to voice. One context and
// one set of weights serve all slots, using the stock sampler and chat parser.
class frankie_completions {
    using json = nlohmann::ordered_json;
    struct job {
        json input;
        bool chat = false, ready = false, done = false;
        std::atomic<bool> cancelled{false};
        std::string id, raw, sent, reasoning, error;
        const int64_t created = std::time(nullptr);
        llama_tokens prompt, draft;
        size_t offset = 0, generated = 0, maximum = frankie_options::default_output_tokens;
        size_t cached = 0, cache_at = 0;
        size_t drafted = 0, accepted = 0;
        llama_pos pos = 0;
        std::map<std::string, brain_session::image_ptr> images;
        std::vector<mtmd::input_chunks_ptr> chunks;
        std::map<size_t, const mtmd_input_chunk *> media;
        std::vector<float> image_rows;
        std::unique_ptr<decode_embd_batch> image_batch;
        size_t image_offset = 0;
        llama_pos image_end = 0;
        llama_seq_id seq = 0;
        llama_token pending = LLAMA_TOKEN_NULL;
        common_chat_parser_params parser;
        std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler{nullptr, common_sampler_free};
        std::mutex mutex;
        std::condition_variable changed;
        std::deque<std::string> events;
        json result;

        void send(const json & choice) {
            std::lock_guard<std::mutex> lock(mutex);
            if (events.size() >= 128) { cancelled = true; return; }
            events.push_back("data: " + json({{"id", id}, {"object", chat ? "chat.completion.chunk" : "text_completion"},
                {"created", created}, {"model", "Frankie"}, {"choices", json::array({choice})}}).dump() + "\n\n");
            changed.notify_all();
        }
    };
    brain_session & brain;
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<std::shared_ptr<job>> slots;
    struct prompt_cache {
        llama_tokens prefix;
        brain_session::branch_state state;
    };
    std::vector<prompt_cache> caches;
    std::atomic<bool> stopping{false}, work{false};
    std::atomic<uint64_t> serial{0};
    size_t prefill_cursor = 0;
    std::thread worker;

    void prepare(job & j) {
        const auto * vocab = llama_model_get_vocab(brain.model.get());
        auto sampling = frankie_brain_sampling(false);
        if (j.chat) {
            common_chat_templates_inputs input;
            input.messages = common_chat_msgs_parse_oaicompat(common_json::parse(j.input.at("messages").dump()));
            input.tools = common_chat_tools_parse_oaicompat(common_json::parse(j.input.value("tools", json::array()).dump()));
            const auto effort = j.input.value("reasoning_effort", json());
            const int budget = frankie_thinking_budget(effort.is_null() ? brain.options.http_thinking : effort.get<std::string>());
            input.enable_thinking = j.input.value("enable_thinking", budget > 0);
            if (!effort.is_null() && j.input.contains("enable_thinking") && input.enable_thinking != (budget > 0)) {
                throw std::invalid_argument("enable_thinking conflicts with reasoning_effort");
            }
            input.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            auto formatted = common_chat_templates_apply(brain.templates.get(), input);
            sampling = frankie_brain_sampling(input.enable_thinking);
            if (input.enable_thinking) { brain.configure_thinking(sampling, formatted, budget); }
            size_t offset = 0;
            while (offset < formatted.prompt.size()) {
                const auto marker = formatted.prompt.find("[FRANKIE_", offset);
                auto tokens = common_tokenize(vocab, formatted.prompt.substr(offset, marker - offset), false, true);
                j.prompt.insert(j.prompt.end(), tokens.begin(), tokens.end());
                if (marker == std::string::npos) { break; }
                const auto end = formatted.prompt.find(']', marker);
                const auto found = j.images.find(formatted.prompt.substr(marker, end - marker + 1));
                if (end == std::string::npos || found == j.images.end()) { throw std::invalid_argument("unknown media marker"); }
                mtmd_input_part part{};
                part.bitmap = found->second.get();
                const mtmd_input_part * parts[] = { &part };
                mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
                if (mtmd_tokenize_from_parts(brain.vision.get(), chunks.get(), parts, 1, false)) {
                    throw std::invalid_argument("image tokenization failed");
                }
                for (size_t k = 0; k < mtmd_input_chunks_size(chunks.get()); ++k) {
                    const auto * chunk = mtmd_input_chunks_get(chunks.get(), k);
                    if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                        size_t count = 0;
                        const auto * tokens = mtmd_input_chunk_get_tokens_text(chunk, &count);
                        j.prompt.insert(j.prompt.end(), tokens, tokens + count);
                    } else {
                        j.media.emplace(j.prompt.size(), chunk);
                        j.prompt.resize(j.prompt.size() + mtmd_input_chunk_get_n_tokens(chunk), LLAMA_TOKEN_NULL);
                    }
                }
                j.chunks.push_back(std::move(chunks));
                offset = end + 1;
            }
            j.images.clear();
            j.parser = common_chat_parser_params(formatted);
            j.parser.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            if (!formatted.parser.empty()) { j.parser.parser.load(formatted.parser); }
            sampling.generation_prompt = formatted.generation_prompt;
            j.send({{"index", 0}, {"delta", {{"role", "assistant"}}}, {"finish_reason", nullptr}});
        } else {
            j.prompt = common_tokenize(vocab, j.input.at("prompt").get<std::string>(), false, true);
        }
        if (j.prompt.empty() || j.prompt.back() == LLAMA_TOKEN_NULL) {
            throw std::runtime_error("prompt must contain text after any media");
        }
        j.maximum = frankie_http_output_budget(j.prompt.size(), j.maximum, brain.options.http_context_tokens);
        const auto number = [&](const char * key, float fallback, float low, float high) {
            const auto value = j.input.value(key, json());
            if (value.is_null()) { return fallback; }
            if (!value.is_number()) { throw std::invalid_argument(std::string(key) + " must be a number"); }
            const float result = value.get<float>();
            if (!std::isfinite(result) || result < low || result > high) {
                throw std::invalid_argument(std::string(key) + " is out of range");
            }
            return result;
        };
        sampling.temp = number("temperature", sampling.temp, 0.0f, 2.0f);
        sampling.top_p = number("top_p", sampling.top_p, 0.0f, 1.0f);
        sampling.min_p = number("min_p", sampling.min_p, 0.0f, 1.0f);
        sampling.penalty_present = number("presence_penalty", sampling.penalty_present, -2.0f, 2.0f);
        sampling.penalty_freq = number("frequency_penalty", sampling.penalty_freq, -2.0f, 2.0f);
        sampling.penalty_last_n = j.maximum;
        sampling.penalty_repeat = number("repeat_penalty", sampling.penalty_repeat,
            std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
        const float repetition = number("repetition_penalty", sampling.penalty_repeat,
            std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
        if (!j.input.value("repeat_penalty", json()).is_null() && repetition != sampling.penalty_repeat) {
            throw std::invalid_argument("repeat_penalty conflicts with repetition_penalty");
        }
        sampling.penalty_repeat = repetition;
        if (!j.input.value("top_k", json()).is_null()) {
            const auto & value = j.input.at("top_k");
            if (!value.is_number_integer() || value < INT32_MIN || value > INT32_MAX) {
                throw std::invalid_argument("top_k must be a 32-bit integer");
            }
            sampling.top_k = value.get<int>();
        }
        if (!j.input.value("seed", json()).is_null()) { sampling.seed = j.input.at("seed").get<uint32_t>(); }
        j.sampler.reset(common_sampler_init(brain.model.get(), sampling));
        if (!j.sampler) { throw std::runtime_error("sampler initialization failed"); }
        if (j.input.value("cache_prompt", true) && j.media.empty()) {
            std::lock_guard<std::mutex> lock(mutex);
            size_t best = j.seq - 1, matched = 0;
            for (size_t i = 0; i < caches.size(); ++i) {
                const auto & prefix = caches[i].prefix;
                if ((!slots[i] || slots[i].get() == &j) && prefix.size() > matched && prefix.size() < j.prompt.size() &&
                    std::equal(prefix.begin(), prefix.end(), j.prompt.begin())) {
                    best = i;
                    matched = prefix.size();
                }
            }
            if (best != size_t(j.seq - 1)) {
                std::swap(slots[best], slots[j.seq - 1]);
                j.seq = best + 1;
            }
            if (matched && clear(j, true)) {
                j.offset = j.cached = matched;
                j.pos = caches[best].state.pos;
            }
            // Keep the mutable assistant tail out of the checkpoint without changing normal prefill batches.
            const size_t chunk = std::min<size_t>(128, llama_n_batch(brain.ctx.get()) - slots.size() * (1 + brain.options.mtp_tokens));
            j.cache_at = std::max(j.cached, j.prompt.size() > 128 ? (j.prompt.size() - 128) / chunk * chunk : 0);
        }
        if (!j.cached) { clear(j); }
        j.ready = true;
    }

    void finish(job & j, const std::string & reason) {
        if (brain.speculative) {
            std::cerr << "http_mtp id=" << j.id << " generated=" << j.generated
                      << " drafted=" << j.drafted << " accepted=" << j.accepted << "\n";
        }
        common_chat_msg message;
        message.role = "assistant";
        message.content = j.raw;
        if (j.chat) {
            message = common_chat_parse(j.raw, false, j.parser);
            message.role = "assistant";
            std::vector<std::string> ids;
            size_t serial = 0;
            message.set_tool_call_ids(ids, [&] {
                return "call_" + std::to_string(j.created) + "_" + j.id + "_" + std::to_string(serial++);
            });
        }
        auto msg = json::parse(common_chat_msgs_to_json_oaicompat({message}).dump())[0];
        const auto finish_reason = message.tool_calls.empty() ? reason : "tool_calls";
        if (j.chat && !message.tool_calls.empty()) {
            auto calls = msg["tool_calls"];
            for (size_t i = 0; i < calls.size(); ++i) { calls[i]["index"] = i; }
            j.send({{"index", 0}, {"delta", {{"tool_calls", calls}}}, {"finish_reason", nullptr}});
        }
        j.send(j.chat ? json{{"index", 0}, {"delta", json::object()}, {"finish_reason", finish_reason}}
                     : json{{"index", 0}, {"text", ""}, {"finish_reason", finish_reason}});
        std::lock_guard<std::mutex> lock(j.mutex);
        j.result = {{"id", j.id}, {"object", j.chat ? "chat.completion" : "text_completion"},
            {"created", j.created}, {"model", "Frankie"},
            {"choices", json::array({j.chat ? json{{"index", 0}, {"message", msg}, {"finish_reason", finish_reason}}
                                           : json{{"index", 0}, {"text", j.raw}, {"finish_reason", finish_reason}}})},
            {"usage", {{"prompt_tokens", j.prompt.size()}, {"completion_tokens", j.generated},
                       {"prompt_tokens_details", {{"cached_tokens", j.cached}}},
                       {"total_tokens", j.prompt.size() + j.generated}}}};
        if (j.input.value("stream_options", json::object()).value("include_usage", false)) {
            auto usage = j.result;
            usage["object"] = j.chat ? "chat.completion.chunk" : "text_completion";
            usage["choices"] = json::array();
            j.events.push_back("data: " + usage.dump() + "\n\n");
        }
        j.events.push_back("data: [DONE]\n\n");
        j.done = true;
        j.changed.notify_all();
    }

    void fail(job & j, const std::string & error) {
        std::lock_guard<std::mutex> lock(j.mutex);
        j.error = error;
        j.events.push_back("data: " + json({{"error", {{"message", error}}}}).dump() + "\n\n");
        j.events.push_back("data: [DONE]\n\n");
        j.done = true;
        j.changed.notify_all();
    }

    void set_abort_callback(bool enabled) {
        ggml_abort_callback callback = enabled ? +[](void * user) {
            return static_cast<brain_session *>(user)->cancelled.load();
        } : nullptr;
        for (auto * ctx : {brain.ctx.get(), brain.draft_ctx.get()}) {
            if (ctx) { llama_set_abort_callback(ctx, callback, &brain); }
        }
    }

    bool clear(job & j, bool retain = false) {
        auto & cache = caches[j.seq - 1];
        if (brain.speculative) {
            common_speculative_get_draft_params(brain.speculative.get(), j.seq) = {};
        }
        if (retain && !cache.prefix.empty()) {
            try {
                brain.restore_branch(cache.state, j.seq);
                return true;
            } catch (const std::exception & e) {
                std::cerr << "http_prompt_cache restore_failed seq=" << j.seq << " " << e.what() << "\n";
            }
        }
        cache = {};
        llama_memory_seq_rm(llama_get_memory(brain.ctx.get()), j.seq, -1, -1);
        if (brain.speculative) {
            llama_memory_seq_rm(llama_get_memory(brain.draft_ctx.get()), j.seq, -1, -1);
            common_speculative_set_state(brain.speculative.get(), j.seq, {});
        }
        return false;
    }

    void image_step(job & j, const mtmd_input_chunk * chunk, size_t count) {
        if (!j.image_batch) {
            if (mtmd_decode_use_non_causal(brain.vision.get(), chunk)) {
                throw std::runtime_error("chunked image prefill requires causal vision embeddings");
            }
            if (mtmd_encode_chunk(brain.vision.get(), chunk)) { throw std::runtime_error("image encoding failed"); }
            const size_t rows = mtmd_input_chunk_get_n_tokens(chunk);
            const auto width = llama_model_n_embd_inp(brain.model.get());
            const auto * data = mtmd_get_output_embd(brain.vision.get());
            j.image_rows.assign(data, data + rows * width);
            const bool mrope = mtmd_decode_use_mrope(brain.vision.get());
            j.image_batch.reset(new decode_embd_batch(j.image_rows.data(), rows, mrope ? 4 : 1, width));
            if (mrope) {
                std::vector<mtmd_decoder_pos> positions(rows);
                mtmd_helper_image_get_decoder_pos(mtmd_input_chunk_get_tokens_image(chunk), j.pos, positions.data());
                j.image_batch->set_position_mrope_2d(positions, j.seq);
            } else { j.image_batch->set_position_normal(j.pos, j.seq); }
            j.image_offset = 0;
            j.image_end = j.pos + mtmd_input_chunk_get_n_pos(chunk);
            return;
        }
        count = std::min(count, size_t(j.image_batch->batch.n_tokens) - j.image_offset);
        if (brain.decode(j.image_batch->get_view(j.image_offset, count))) { throw std::runtime_error("image prefill failed"); }
        j.image_offset += count;
        if (j.image_offset == size_t(j.image_batch->batch.n_tokens)) {
            j.offset += j.image_offset;
            j.pos = j.image_end;
            j.image_batch.reset();
            std::vector<float>().swap(j.image_rows);
        }
    }

    bool has_headroom(int64_t reserve = 350000) const {
        const auto now = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        return !brain.speech_active.load() || brain.speech_buffer_until_us.load() - now >= reserve;
    }

    void step() {
        if (!work.load() || !has_headroom()) { return; }
        std::lock_guard<std::recursive_mutex> compute_lock(brain.compute_mutex);
        if (!has_headroom()) { return; }
        std::vector<std::shared_ptr<job>> active;
        { std::lock_guard<std::mutex> lock(mutex); active = slots; }
        set_abort_callback(false);
        struct restore_abort {
            frankie_completions & owner;
            ~restore_abort() { owner.set_abort_callback(true); }
        } restore{*this};
        const auto * vocab = llama_model_get_vocab(brain.model.get());
        if (brain.speculative) {
            for (auto & slot : active) {
                if (!slot || slot->done || slot->cancelled || !slot->ready || slot->offset < slot->prompt.size()) { continue; }
                auto & j = *slot;
                j.draft.clear();
                if (j.maximum - j.generated > 1) {
                    common_speculative_get_draft_params(brain.speculative.get(), j.seq) = {
                        true, int32_t(std::min<size_t>(brain.options.mtp_tokens, j.maximum - j.generated - 1)),
                        j.pos, j.pending, &j.prompt, &j.draft,
                    };
                }
            }
            common_speculative_draft(brain.speculative.get());
            for (auto & slot : active) {
                if (!slot || slot->draft.empty()) { continue; }
                auto & j = *slot;
                j.draft.erase(std::find_if(j.draft.begin(), j.draft.end(), [&](llama_token t) {
                    return llama_vocab_is_eog(vocab, t);
                }), j.draft.end());
                llama_memory_seq_rm(llama_get_memory(brain.draft_ctx.get()), j.seq, j.pos, -1);
            }
        }
        auto batch = llama_batch_init(llama_n_batch(brain.ctx.get()), 0, 1);
        struct release { llama_batch & batch; ~release() { llama_batch_free(batch); } } release_batch{batch};
        struct output { std::shared_ptr<job> slot; std::vector<int> indices; bool verify; };
        std::vector<output> outputs;
        bool prefilled = false;
        const size_t start = prefill_cursor;
        for (size_t index = 0; index < active.size(); ++index) {
            const size_t slot_index = (start + index) % active.size();
            auto & slot = active[slot_index];
            if (!slot) { continue; }
            auto & j = *slot;
            if (j.done || j.cancelled.load()) {
                continue;
            }
            try {
                if (!j.ready) { prepare(j); }
                if (j.offset < j.prompt.size()) {
                    if (prefilled) { continue; }
                    prefilled = true;
                    prefill_cursor = (slot_index + 1) % active.size();
                    const size_t reserve = active.size() * (1 + brain.options.mtp_tokens);
                    const size_t count = std::min(brain.speech_active.load() ? size_t(32) : size_t(128), llama_n_batch(brain.ctx.get()) - reserve);
                    const auto media = j.media.find(j.offset);
                    if (media != j.media.end()) {
                        if (!j.image_batch && !has_headroom(600000)) { continue; }
                        image_step(j, media->second, count);
                        continue;
                    }
                    size_t end = std::min(j.prompt.size(), j.offset + count);
                    if (j.offset < j.cache_at) { end = std::min(end, j.cache_at); }
                    const auto next_image = j.media.lower_bound(j.offset);
                    if (next_image != j.media.end()) { end = std::min(end, next_image->first); }
                    while (j.offset < end) {
                        common_batch_add(batch, j.prompt[j.offset], j.pos++, {j.seq}, j.offset + 1 == j.prompt.size());
                        ++j.offset;
                    }
                    if (j.offset == j.prompt.size()) { outputs.push_back({slot, {batch.n_tokens - 1}, false}); }
                } else {
                    output item{slot, {}, true};
                    common_batch_add(batch, j.pending, j.pos, {j.seq}, true);
                    item.indices.push_back(batch.n_tokens - 1);
                    for (size_t k = 0; k < j.draft.size(); ++k) {
                        common_batch_add(batch, j.draft[k], j.pos + 1 + k, {j.seq}, true);
                        item.indices.push_back(batch.n_tokens - 1);
                    }
                    outputs.push_back(std::move(item));
                }
            } catch (const std::exception & e) { fail(j, e.what()); }
        }
        if (batch.n_tokens) {
            const int rc = brain.decode(batch);
            if (rc) {
                for (auto & slot : active) { if (slot && !slot->done) { fail(*slot, "text batch decode failed"); } }
            } else {
                for (auto & slot : active) {
                    if (slot && !slot->done && slot->cache_at && slot->offset == slot->cache_at) {
                        auto & cache = caches[slot->seq - 1];
                        if (cache.prefix.size() != slot->cache_at) {
                            try {
                                brain.capture_branch(cache.state, slot->pos, slot->seq);
                                cache.prefix.assign(slot->prompt.begin(), slot->prompt.begin() + slot->offset);
                            } catch (const std::exception & e) {
                                cache = {};
                                slot->cache_at = 0;
                                std::cerr << "http_prompt_cache capture_failed seq=" << slot->seq << " " << e.what() << "\n";
                            }
                        }
                    }
                }
                for (auto & entry : outputs) {
                    auto & j = *entry.slot;
                    try {
                        const auto ids = common_sampler_sample_and_accept_n(j.sampler.get(), brain.ctx.get(), entry.indices, j.draft);
                        if (entry.verify) {
                            j.pos += ids.size();
                            if (brain.speculative) {
                                j.drafted += j.draft.size();
                                j.accepted += ids.size() - 1;
                                common_speculative_accept(brain.speculative.get(), j.seq, ids.size() - 1);
                                if (ids.size() <= j.draft.size()) {
                                    if (!llama_memory_seq_rm(llama_get_memory(brain.ctx.get()), j.seq, j.pos, -1) ||
                                        !llama_memory_seq_rm(llama_get_memory(brain.draft_ctx.get()), j.seq, j.pos, -1)) {
                                        throw std::runtime_error("HTTP MTP accepted-prefix rollback failed");
                                    }
                                }
                            }
                        }
                        j.draft.clear();
                        for (const auto token : ids) {
                            j.pending = token;
                            if (llama_vocab_is_eog(vocab, token)) { finish(j, "stop"); break; }
                            j.raw += common_token_to_piece(vocab, token, true);
                            ++j.generated;
                            auto parsed = j.chat ? common_chat_parse(j.raw, true, j.parser) : common_chat_msg{};
                            const auto & text = j.chat ? parsed.content : j.raw;
                            json delta = json::object();
                            if (text.compare(0, j.sent.size(), j.sent) != 0) { throw std::runtime_error("text parser retracted streamed output"); }
                            if (text.size() > j.sent.size()) { delta[j.chat ? "content" : "text"] = text.substr(j.sent.size()); j.sent = text; }
                            if (parsed.reasoning_content.size() > j.reasoning.size()) {
                                delta["reasoning_content"] = parsed.reasoning_content.substr(j.reasoning.size());
                                j.reasoning = parsed.reasoning_content;
                            }
                            if (!delta.empty()) {
                                j.send(j.chat ? json{{"index", 0}, {"delta", delta}, {"finish_reason", nullptr}}
                                              : json{{"index", 0}, {"text", delta["text"]}, {"finish_reason", nullptr}});
                            }
                            if (j.generated >= j.maximum) { finish(j, "length"); break; }
                        }
                    } catch (const std::exception & e) { fail(j, e.what()); }
                }
            }
        }
        if (std::getenv("FRANKIE_HTTP_PROFILE")) {
            for (const auto & slot : active) { if (slot) {
                const auto now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
                std::cerr << "http_progress " << json({{"at", now}, {"seq", slot->seq},
                    {"pos", slot->pos}, {"offset", slot->offset},
                    {"prefill_tokens", slot->offset + (slot->image_batch ? slot->image_offset : 0)},
                    {"cached_tokens", slot->cached}, {"generated", slot->generated}}).dump() << "\n";
            } }
        }
        std::lock_guard<std::mutex> lock(mutex);
        for (auto & slot : slots) {
            if (slot && (slot->done || slot->cancelled.load())) {
                clear(*slot, slot->done && slot->error.empty());
                slot.reset();
            }
        }
        work = std::any_of(slots.begin(), slots.end(), [](const auto & slot) { return bool(slot); });
        brain.http_pending = work.load();
    }

    void handle(const httplib::Request & request, httplib::Response & response) {
        auto j = std::make_shared<job>();
        try {
            j->input = json::parse(request.body);
            j->chat = request.path == "/v1/chat/completions";
            if (j->chat) {
                if (j->input.at("messages").dump().find("[FRANKIE_") != std::string::npos) {
                    throw std::invalid_argument("reserved media marker");
                }
                for (auto & message : j->input.at("messages")) {
                    auto content = message.find("content");
                    if (content == message.end() || !content->is_array()) { continue; }
                    for (auto & part : *content) {
                        const auto type = part.value("type", "");
                        if (type == "text") { continue; }
                        if (type != "image_url" || message.value("role", "") != "user") {
                            throw std::invalid_argument("use text or user image_url content parts");
                        }
                        if (j->images.size() >= 4) { throw std::invalid_argument("at most four images per request"); }
                        const auto & image = part.at("image_url");
                        frankie_image_detail(image.value("detail", "auto"));
                        // Network and bitmap decoding do not hold the brain's compute lock.
                        auto bitmap = brain.decode_image(frankie_image_bytes(image.at("url").get<std::string>()));
                        const auto marker = "[FRANKIE_IMAGE_" + std::to_string(j->images.size()) + "]";
                        j->images.emplace(marker, std::move(bitmap));
                        part = {{"type", "media_marker"}, {"text", marker}};
                    }
                }
            }
            auto maximum = j->input.value("max_completion_tokens", json());
            if (maximum.is_null()) { maximum = j->input.value("max_tokens", json()); }
            if (maximum.is_null()) { maximum = j->maximum; }
            if (!maximum.is_number_integer() || maximum <= 0 || maximum > INT32_MAX) {
                throw std::runtime_error("max_tokens/max_completion_tokens must be a positive 32-bit integer");
            }
            if (j->input.value("n", 1) != 1) { throw std::runtime_error("n must be 1"); }
            j->maximum = maximum.get<size_t>();
            const auto choice = j->input.value("tool_choice", json());
            if (choice == "auto" || choice == "none") {
                if (choice == "none") { j->input["tools"] = json::array(); }
                j->input.erase("tool_choice");
            }
            if (j->input.value("stop", json()) == json::array()) { j->input.erase("stop"); }
            if (j->input.value("echo", json()) == false) { j->input.erase("echo"); }
            if (j->input.value("response_format", json()) == json{{"type", "text"}}) { j->input.erase("response_format"); }
            if (j->input.contains("stream_options") && j->input["stream_options"].is_null()) { j->input.erase("stream_options"); }
            for (const auto * key : {"stop", "logprobs", "top_logprobs", "response_format", "tool_choice", "echo"}) {
                if (j->input.contains(key) && !j->input[key].is_null()) { throw std::runtime_error(std::string(key) + " is not wired into the experiment yet"); }
            }
            j->id = (j->chat ? "chatcmpl-" : "cmpl-") + std::to_string(++serial);
            {
                std::lock_guard<std::mutex> lock(mutex);
                auto free = std::find(slots.begin(), slots.end(), nullptr);
                if (free == slots.end()) { response.status = 429; throw std::runtime_error("all HTTP slots are busy"); }
                j->seq = 1 + std::distance(slots.begin(), free);
                *free = j;
                work = true;
                brain.http_pending = true;
            }
            changed.notify_all();
            if (j->input.value("stream", false)) {
                response.set_chunked_content_provider("text/event-stream", [j](size_t, httplib::DataSink & sink) {
                    std::unique_lock<std::mutex> lock(j->mutex);
                    j->changed.wait_for(lock, std::chrono::milliseconds(100), [&] { return j->done || !j->events.empty(); });
                    if (!sink.is_writable()) { j->cancelled = true; return false; }
                    while (!j->events.empty()) {
                        auto event = std::move(j->events.front()); j->events.pop_front();
                        lock.unlock();
                        if (!sink.write(event.data(), event.size())) { j->cancelled = true; return false; }
                        lock.lock();
                    }
                    if (j->done) { sink.done(); }
                    return true;
                }, [j](bool) { j->cancelled = true; });
            } else {
                std::unique_lock<std::mutex> lock(j->mutex);
                while (!j->done) {
                    j->changed.wait_for(lock, std::chrono::milliseconds(100));
                    j->events.clear();
                    if (request.is_connection_closed()) { j->cancelled = true; return; }
                }
                if (!j->error.empty()) { throw std::runtime_error(j->error); }
                response.set_content(j->result.dump(), "application/json");
            }
        } catch (const std::exception & e) {
            j->cancelled = true;
            if (response.status < 400) { response.status = 400; }
            response.set_content(json({{"error", {{"message", e.what()}}}}).dump(), "application/json");
        }
    }

  public:
    frankie_completions(brain_session & b, httplib::Server & server, size_t count) : brain(b), slots(count), caches(count) {
        if (!count) { return; }
        if (count * (1 + brain.options.mtp_tokens) >= llama_n_batch(brain.ctx.get())) {
            throw std::runtime_error("batch size must exceed HTTP slots times the verification width");
        }
        brain.background_step = [this] { step(); };
        server.Get("/v1/models", [](const auto &, auto & res) {
            res.set_content(json({{"object", "list"}, {"data", json::array({
                {{"id", "Frankie"}, {"object", "model"}, {"created", 0}, {"owned_by", "local"}}
            })}}).dump(), "application/json");
        });
        server.Post("/v1/completions", [this](const auto & req, auto & res) { handle(req, res); });
        server.Post("/v1/chat/completions", [this](const auto & req, auto & res) { handle(req, res); });
        worker = std::thread([this] {
            while (!stopping.load()) {
                { std::unique_lock<std::mutex> lock(mutex); changed.wait(lock, [&] { return stopping.load() || work.load(); }); }
                if (stopping.load()) { break; }
                step();
                if (brain.speech_active.load()) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
                else { std::this_thread::yield(); }
            }
        });
    }
    ~frankie_completions() {
        stopping = true;
        changed.notify_all();
        if (worker.joinable()) { worker.join(); }
        brain.background_step = {};
        brain.http_pending = false;
    }
};
