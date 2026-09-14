#include "brain-session.h"
#include "ctc-vocabulary.h"

#include "llama-ext.h"
#include "mtmd-audio.h"
#include "sampling.h"
#include "brain-sampling.h"
#include "token-boundary.h"
#include "stb/stb_image.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>

brain_session::brain_session(const std::string & package, const frankie_options & config) :
    encoder_source(package, "ear", config.ear_model),
    bridge_source(package, "bridge"),
    brain_source(package, "brain"),
    vision_source(package, "vision"), options(config) {
    options.resolve_context();
    clip_context_params ep{};
    ep.use_gpu                = options.use_gpu;
    ep.flash_attn_type        = CLIP_FLASH_ATTN_TYPE_DISABLED;
    ep.model_reader           = component::callback;
    ep.model_reader_user_data = &encoder_source;
    ep.model_reader_size      = encoder_source.size();
    encoder.reset(clip_init("embedded-ear", ep).ctx_a);
    if (!encoder) {
        throw std::runtime_error("ear load failed");
    }
    ggml_context *                                      raw = nullptr;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
        gguf_init_from_callback(component::callback, &bridge_source, 1024 * 1024, bridge_source.size(),
                                { false, &raw }),
        gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> weights(raw, ggml_free);
    if (!metadata || !weights) {
        throw std::runtime_error("bridge load failed");
    }
    ear             = std::make_unique<mtmd_ear>(raw, options.use_gpu);
    weights.reset();
    auto mp         = llama_model_default_params();
    mp.n_gpu_layers = options.use_gpu ? 99 : 0;
    mp.load_mtp = options.mtp_tokens != 0;
    model.reset(llama_model_init_from_user(brain_source.metadata(), component::set_tensor, &brain_source, mp));
    if (!model) {
        throw std::runtime_error("brain load failed");
    }
    auto cp              = llama_context_default_params();
    cp.n_ctx             = options.context_tokens;
    cp.n_seq_max         = 1 + options.http_slots;
    if (options.http_slots) {
        cp.n_outputs_max = std::max(cp.n_seq_max, (1 + options.mtp_tokens) * options.http_slots);
        cp.n_outputs_max_per_seq = 1 + options.mtp_tokens;
    }
    cp.n_rs_seq          = options.mtp_tokens;
    cp.kv_unified        = true;
    cp.type_k            = options.cache_type == "q4_0" ? GGML_TYPE_Q4_0 : options.cache_type == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    cp.type_v            = cp.type_k;
    cp.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.offload_kqv       = options.use_gpu;
    cp.op_offload        = options.use_gpu;
    cp.n_batch           = options.batch_size ? options.batch_size : (options.use_gpu ? 512 : 128);
    cp.n_ubatch          = options.ubatch_size ? options.ubatch_size : std::min(cp.n_batch, options.use_gpu ? 512u : 128u);
    cp.n_threads         = options.threads;
    cp.n_threads_batch   = options.threads;
    if (options.mtp_tokens && (cp.n_batch <= options.mtp_tokens || cp.n_ubatch <= options.mtp_tokens)) {
        throw std::runtime_error("MTP requires batch and ubatch larger than the draft length");
    }
    cp.abort_callback = [](void * user) {
        return static_cast<brain_session *>(user)->cancelled.load();
    };
    cp.abort_callback_data = this;
    ctx.reset(llama_init_from_model(model.get(), cp));
    std::cerr << "Frankie context total=" << options.context_tokens << " allocated=" << (ctx ? llama_n_ctx(ctx.get()) : 0)
              << " voice=" << context_tokens() << " http_slots=" << options.http_slots
              << " http_per_slot=" << (options.http_slots ? options.http_context_tokens : 0)
              << " KV=" << options.cache_type << "\n";
    if (!ctx) {
        throw std::runtime_error("brain context failed");
    }
    if (options.mtp_tokens) {
        if (!llama_model_n_layer_nextn(model.get())) {
            throw std::runtime_error("MTP requested but the brain component contains no MTP head");
        }
        auto dp = cp;
        dp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        dp.ctx_other = ctx.get();
        dp.n_rs_seq = 0;
        draft_ctx.reset(llama_init_from_model(model.get(), dp));
        if (!draft_ctx) { throw std::runtime_error("MTP context failed"); }
        common_params_speculative sp;
        sp.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        sp.draft.ctx_tgt = ctx.get();
        sp.draft.ctx_dft = draft_ctx.get();
        sp.draft.n_max = options.mtp_tokens;
        sp.draft.n_min = 1;
        sp.draft.p_min = 0.0f;
        speculative.reset(common_speculative_init(sp, cp.n_seq_max));
        if (!speculative) { throw std::runtime_error("MTP initialization failed"); }
    }
    llama_set_embeddings_layer_inp(ctx.get(), 17, true);
    auto vp                   = mtmd_context_params_default();
    vp.use_gpu                = options.use_gpu;
    vp.n_threads              = options.threads;
    vp.image_max_tokens       = 1024;
    vp.model_reader           = component::callback;
    vp.model_reader_user_data = &vision_source;
    vp.model_reader_size      = vision_source.size();
    vision.reset(mtmd_init_from_file("embedded-vision", model.get(), vp));
    if (!vision || !mtmd_support_vision(vision.get())) {
        throw std::runtime_error("vision load failed");
    }
    templates = common_chat_templates_init(model.get(), "");
}

std::vector<float> brain_session::encode_audio(const std::vector<float> & pcm, std::string * transcript) {
    std::lock_guard<std::recursive_mutex> compute_lock(compute_mutex);
    const auto started = std::chrono::steady_clock::now();
    if (pcm.empty() || pcm.size() > 16000 * options.max_utterance_seconds) {
        throw std::runtime_error("audio bounds");
    }
    for (float value : pcm) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("nonfinite input audio");
        }
    }
    mtmd_audio_preprocessor_parakeet frontend(encoder.get());
    frontend.initialize();
    std::vector<mtmd_audio_mel> mel;
    if (!frontend.preprocess(pcm.data(), pcm.size(), mel) || mel.size() != 1) {
        throw std::runtime_error("frontend failed");
    }
    const auto frontend_done = std::chrono::steady_clock::now();
    clip_image_f32 image;
    image.set_size({ int(mel[0].n_len), 80 }, false, true);
    image.cpy_buf(mel[0].data);
    size_t nf = clip_n_output_tokens(encoder.get(), &image);
    if (nf == 0 || nf + 1 > audio_row_limit()) {
        throw std::runtime_error("encoder frame bounds");
    }
    std::vector<float>   frames(nf * 512);
    clip_image_f32_batch batch;
    batch.is_audio = true;
    batch.entries.push_back(std::move(image));
    if (!clip_image_batch_encode(encoder.get(), options.threads, &batch, frames)) {
        throw std::runtime_error("encoder failed");
    }
    const auto encoder_done = std::chrono::steady_clock::now();
    std::vector<int32_t> ids;
    auto rows = ear->process(frames.data(), nf, transcript ? &ids : nullptr);
    if (transcript) { *transcript = frankie_ctc_text(ids); }
    if (std::getenv("FRANKIE_PROFILE")) {
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        std::cerr << "ear_profile frames=" << nf << " frontend_ms=" << ms(started, frontend_done)
                  << " encoder_ms=" << ms(frontend_done, encoder_done)
                  << " bridge_ms=" << ms(encoder_done, std::chrono::steady_clock::now()) << "\n";
    }
    return rows;
}

brain_session::image_ptr brain_session::decode_image(const std::vector<unsigned char> & bytes) {
    int nx = 0, ny = 0, nc = 0;
    if (bytes.empty() || bytes.size() > 10 * 1024 * 1024 ||
        !stbi_info_from_memory(bytes.data(), bytes.size(), &nx, &ny, &nc) || nx <= 0 || ny <= 0 || nx > 4096 ||
        ny > 4096 || int64_t(nx) * ny > 4 * 1024 * 1024) {
        throw std::runtime_error("image format/dimension bounds");
    }
    auto      decoded = mtmd_helper_bitmap_init_from_buf(vision.get(), bytes.data(), bytes.size(), false,
                                                         mtmd_helper_init_opt_default());
    image_ptr image(decoded.bitmap, mtmd_bitmap_free);
    if (decoded.video_ctx) {
        mtmd_helper_video_free(decoded.video_ctx);
        throw std::runtime_error("video input unsupported");
    }
    if (!image || mtmd_bitmap_is_audio(image.get())) {
        throw std::runtime_error("image decode failed");
    }
    return image;
}

void brain_session::decode_text(const std::string & text, int & pos, size_t & used) {
    if (text.empty()) {
        return;
    }
    auto *                   vocab = llama_model_get_vocab(model.get());
    std::vector<llama_token> tokens(text.size() + 32);
    int n = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), false, true);
    if (n < 0 || used + n + 512 > context_tokens()) {
        throw std::runtime_error("prompt context exhausted");
    }
    used += n;
    const int batch_size = llama_n_batch(ctx.get());
    const auto started = std::chrono::steady_clock::now();
    for (int start = 0; start < n; start += batch_size) {
        if (cancelled.load()) {
            throw std::runtime_error("cancelled");
        }
        int  count     = std::min(batch_size, n - start);
        auto batch     = llama_batch_init(count, 0, 1);
        batch.n_tokens = count;
        for (int i = 0; i < count; ++i) {
            batch.token[i]     = tokens[start + i];
            batch.pos[i]       = pos++;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (start + i == n - 1);
        }
        int rc = decode(batch);
        llama_batch_free(batch);
        if (rc) {
            throw std::runtime_error("text decode failed");
        }
        if (n >= 8192 && ((start + count) / 8192 != start / 8192 || start + count == n)) {
            const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            std::cerr << "prefill tokens=" << start + count << "/" << n << " context=" << pos
                      << " elapsed_s=" << seconds << " tokens_per_s=" << (start + count) / seconds << "\n";
        }
    }
}

// The chat template preserves internal audio markers; each marker is replaced
// with learned ear rows during prefill. The server rejects markers in external
// text, instructions and tool output. Recurrent state rolls back only to a full sequence checkpoint.
void brain_session::reset(bool preserve_checkpoint) {
    std::lock_guard<std::recursive_mutex> compute_lock(compute_mutex);
    cache_valid = false;
    partial_prefix.clear();
    partial_marker.clear();
    partial_rows.clear();
    cached_prefix.clear();
    if (!preserve_checkpoint) {
        checkpoint.reset();
        checkpoint_prefix.clear();
    }
    clear_sequence();
}

int brain_session::decode(const llama_batch & batch) {
    const int rc = llama_decode(ctx.get(), batch);
    if (!rc && speculative && !common_speculative_process(speculative.get(), batch)) {
        return -1;
    }
    return rc;
}

void brain_session::capture_branch(branch_state & state, int pos, llama_seq_id seq) {
    const size_t size = llama_state_seq_get_size_ext(ctx.get(), seq, branch_flags);
    state.recurrent.resize(size);
    if (!size || llama_state_seq_get_data_ext(ctx.get(), state.recurrent.data(), size, seq, branch_flags) != size ||
        (speculative && !common_speculative_get_state(speculative.get(), seq, state.boundary))) {
        throw std::runtime_error("branch checkpoint capture failed");
    }
    state.pos = pos;
}

void brain_session::restore_branch(const branch_state & state, llama_seq_id seq) {
    // Restore recurrent state before removing the branch's attention suffix.
    if (llama_state_seq_set_data_ext(ctx.get(), state.recurrent.data(), state.recurrent.size(), seq, branch_flags) != state.recurrent.size() ||
        !llama_memory_seq_rm(llama_get_memory(ctx.get()), seq, state.pos, -1)) {
        throw std::runtime_error("branch checkpoint restore failed");
    }
    if (speculative) {
        if (!llama_memory_seq_rm(llama_get_memory(draft_ctx.get()), seq, state.pos, -1)) {
            throw std::runtime_error("branch draft rollback failed");
        }
        common_speculative_set_state(speculative.get(), seq, state.boundary);
    }
}

brain_session::sequence_state brain_session::capture_sequence() {
    sequence_state state;
    auto capture = [](llama_context * context, std::vector<uint8_t> & bytes) {
        const size_t size = llama_state_seq_get_size(context, 0);
        if (!size || size > size_t(16) * 1024 * 1024 * 1024) {
            throw std::runtime_error("sequence checkpoint size limit");
        }
        bytes.resize(size);
        if (llama_state_seq_get_data(context, bytes.data(), size, 0) != size) {
            throw std::runtime_error("sequence checkpoint capture failed");
        }
    };
    capture(ctx.get(), state.target);
    if (speculative) {
        capture(draft_ctx.get(), state.draft);
        if (!common_speculative_get_state(speculative.get(), 0, state.boundary)) {
            throw std::runtime_error("MTP checkpoint capture failed");
        }
    }
    return state;
}

void brain_session::clear_sequence() {
    if (options.http_slots) {
        if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1)) {
            throw std::runtime_error("voice sequence reset failed");
        }
    } else {
        llama_memory_clear(llama_get_memory(ctx.get()), true);
    }
    if (speculative) {
        if (options.http_slots) {
            if (!llama_memory_seq_rm(llama_get_memory(draft_ctx.get()), 0, -1, -1)) {
                throw std::runtime_error("voice draft reset failed");
            }
        } else {
            llama_memory_clear(llama_get_memory(draft_ctx.get()), true);
        }
        common_speculative_set_state(speculative.get(), 0, {});
    }
}

void brain_session::restore_sequence(const sequence_state & state) {
    clear_sequence();
    auto restore = [](llama_context * context, const std::vector<uint8_t> & bytes) {
        if (llama_state_seq_set_data(context, bytes.data(), bytes.size(), 0) != bytes.size()) {
            throw std::runtime_error("sequence checkpoint restore failed");
        }
    };
    restore(ctx.get(), state.target);
    if (speculative) {
        restore(draft_ctx.get(), state.draft);
        common_speculative_set_state(speculative.get(), 0, state.boundary);
    }
    if (std::getenv("FRANKIE_VERIFY_ROLLBACK")) {
        const auto actual = capture_sequence();
        if (actual.target != state.target || actual.draft != state.draft || actual.boundary != state.boundary) {
            throw std::runtime_error("sequence checkpoint readback mismatch");
        }
        std::cerr << "recurrent_rollback_verified bytes=" << state.size() << "\n";
    }
}

void brain_session::save_checkpoint(const std::string & prefix, int pos, size_t used) {
    checkpoint = std::make_shared<const sequence_state>(capture_sequence());
    checkpoint_prefix = prefix;
    checkpoint_pos = pos;
    checkpoint_used = used;
}

brain_session::saved_state brain_session::suspend_state() {
    std::lock_guard<std::recursive_mutex> compute_lock(compute_mutex);
    saved_state saved;
    saved.sequence = capture_sequence();
    // Speculation also needs this immutable prefix; sharing avoids a large copy.
    saved.checkpoint = checkpoint;
    saved.checkpoint_prefix = checkpoint_prefix;
    saved.checkpoint_pos = checkpoint_pos;
    saved.checkpoint_used = checkpoint_used;
    saved.cached_prefix = cached_prefix;
    saved.cached_pos = cached_pos;
    saved.cached_used = cached_used;
    saved.cache_valid = cache_valid;
    saved.partial_prefix = partial_prefix;
    saved.partial_marker = partial_marker;
    saved.partial_rows = partial_rows;
    saved.partial_pos = partial_pos;
    saved.partial_used = partial_used;
    return saved;
}

brain_session::listener_reaction brain_session::probe_listener(const request & input) {
    std::lock_guard<std::recursive_mutex> compute_lock(compute_mutex);
    if (partial_marker.empty() || partial_rows.size() < 8 * 5120) { return {}; }
    auto chat = input.chat;
    chat.enable_thinking = false;
    chat.add_generation_prompt = true;
    const auto formatted = common_chat_templates_apply(templates.get(), chat);
    const auto marker = formatted.prompt.find(partial_marker);
    if (marker == std::string::npos) { throw std::runtime_error("listener probe has no active audio marker"); }
    const std::string ask = "\n(You are listening while the user is still talking. Which one-word listener reaction fits what they have said so far? Answer with exactly one of: Hmm, Yeah, Wow, Aw, Nothing.)";
    const bool verify = std::getenv("FRANKIE_VERIFY_ROLLBACK") != nullptr;
    const auto expected = verify ? capture_sequence() : sequence_state{};
    branch_state branch;
    capture_branch(branch, partial_pos);
    auto restore = [&] {
        restore_branch(branch);
        if (verify) {
            const auto actual = capture_sequence();
            if (actual.target != expected.target || actual.draft != expected.draft || actual.boundary != expected.boundary) {
                throw std::runtime_error("listener sequence rollback mismatch");
            }
            std::cerr << "listener_rollback_verified bytes=" << actual.size() << "\n";
        }
    };
    listener_reaction result;
    try {
        int pos = partial_pos;
        size_t used = partial_used;
        decode_text(ask + formatted.prompt.substr(marker + partial_marker.size()), pos, used);
        const char * words[] = {"Hmm", "Yeah", "Wow", "Aw", "Nothing"};
        const auto * logits = llama_get_logits_ith(ctx.get(), -1);
        if (!logits) { throw std::runtime_error("listener probe has no logits"); }
        float maximum = -INFINITY;
        for (int i = 0; i < 5; ++i) {
            llama_token token;
            if (llama_tokenize(llama_model_get_vocab(model.get()), words[i], std::strlen(words[i]), &token, 1, false, false) != 1) {
                throw std::runtime_error("listener reaction must be one token");
            }
            result.probabilities[i] = logits[token];
            if (!std::isfinite(logits[token])) { throw std::runtime_error("invalid listener logits"); }
            if (logits[token] > maximum) { maximum = logits[token]; result.index = i; }
        }
        float sum = 0;
        for (auto & p : result.probabilities) { p = std::exp(p - maximum); sum += p; }
        for (auto & p : result.probabilities) { p /= sum; }
    } catch (...) {
        restore();
        throw;
    }
    restore();
    return result;
}

void brain_session::restore_state(saved_state saved) {
    std::lock_guard<std::recursive_mutex> compute_lock(compute_mutex);
    restore_sequence(saved.sequence);
    checkpoint = std::move(saved.checkpoint);
    checkpoint_prefix = std::move(saved.checkpoint_prefix);
    checkpoint_pos = saved.checkpoint_pos;
    checkpoint_used = saved.checkpoint_used;
    cached_prefix = std::move(saved.cached_prefix);
    cached_pos = saved.cached_pos;
    cached_used = saved.cached_used;
    cache_valid = saved.cache_valid;
    partial_prefix = std::move(saved.partial_prefix);
    partial_marker = std::move(saved.partial_marker);
    partial_rows = std::move(saved.partial_rows);
    partial_pos = saved.partial_pos;
    partial_used = saved.partial_used;
}

void brain_session::warm_prefix(common_chat_templates_inputs input) {
    std::lock_guard<std::recursive_mutex> compute_lock(compute_mutex);
    if (input.messages.size() != 1 || input.messages[0].role != "system") {
        throw std::runtime_error("warm prefix requires only system context");
    }
    input.add_generation_prompt = false;
    common_chat_msg boundary;
    boundary.role = "user";
    boundary.content = "[FRANKIE_WARM_BOUNDARY]";
    input.messages.push_back(boundary);
    auto formatted = common_chat_templates_apply(templates.get(), input);
    const auto cut = formatted.prompt.find(boundary.content);
    if (cut == std::string::npos || formatted.prompt.find(boundary.content, cut + 1) != std::string::npos) {
        throw std::runtime_error("ambiguous warm prefix boundary");
    }
    formatted.prompt.resize(cut);
    if (formatted.prompt.size() > 4 * 1024 * 1024) {
        throw std::runtime_error("formatted prefix too large");
    }
    reset();
    int pos = 0;
    size_t used = 0;
    if (warm_checkpoint && warm_prefix_text == formatted.prompt) {
        restore_sequence(*warm_checkpoint);
        pos = warm_pos; used = warm_used;
        checkpoint = warm_checkpoint;
        checkpoint_prefix = formatted.prompt; checkpoint_pos = pos; checkpoint_used = used;
        std::cerr << "system_prefix_cache_hit bytes=" << warm_checkpoint->size() << "\n";
    } else {
        decode_text(formatted.prompt, pos, used);
        if (cancelled.load()) { throw std::runtime_error("cancelled"); }
        save_checkpoint(formatted.prompt, pos, used);
        // Bound the host cache independently of the configured long context.
        warm_checkpoint.reset(); warm_prefix_text.clear();
        if (checkpoint->size() <= 256 * 1024 * 1024) {
            warm_checkpoint = checkpoint; warm_prefix_text = formatted.prompt;
            warm_pos = pos; warm_used = used;
        }
    }
    if (cancelled.load()) { throw std::runtime_error("cancelled"); }
    cached_prefix = formatted.prompt;
    cached_pos = pos;
    cached_used = used;
    cache_valid = true;
}

size_t brain_session::prompt_tokens(const request & request, const std::map<std::string, size_t> & pending_audio) const {
    const auto formatted = common_chat_templates_apply(templates.get(), request.chat);
    const auto * vocab = llama_model_get_vocab(model.get());
    size_t used = 0, offset = 0;
    auto count_text = [&](const std::string & text) {
        if (text.empty()) { return; }
        const int n = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, false, true);
        used += size_t(n < 0 ? -n : n);
    };
    for (;;) {
        const auto marker = formatted.prompt.find("[FRANKIE_", offset);
        if (marker == std::string::npos) { count_text(formatted.prompt.substr(offset)); break; }
        count_text(formatted.prompt.substr(offset, marker - offset));
        const auto end = formatted.prompt.find(']', marker);
        if (end == std::string::npos) { throw std::runtime_error("broken media marker"); }
        const auto key = formatted.prompt.substr(marker, end - marker + 1);
        const auto audio = request.audio_rows.find(key);
        const auto pending = pending_audio.find(key);
        const auto image = request.images.find(key);
        if (audio != request.audio_rows.end()) { used += audio->second.size() / 5120; }
        else if (pending != pending_audio.end()) { used += pending->second; }
        else if (image != request.images.end()) {
            mtmd_input_part part{};
            part.bitmap = image->second.get();
            const mtmd_input_part * parts[] = { &part };
            mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
            if (mtmd_tokenize_from_parts(vision.get(), chunks.get(), parts, 1, false)) {
                throw std::runtime_error("image tokenization failed");
            }
            used += mtmd_helper_get_n_tokens(chunks.get());
        } else { throw std::runtime_error("unknown media marker"); }
        offset = end + 1;
    }
    return used;
}

void brain_session::decode_rows(const std::vector<float> & rows, size_t start, size_t end, int & pos, size_t & used) {
    if (start > end || end > rows.size() / 5120 || used + end - start + 512 > context_tokens()) {
        throw std::runtime_error("audio context/row bounds");
    }
    used += end - start;
    const size_t batch_size = std::min(128u, llama_n_batch(ctx.get()));
    for (; start < end; start += batch_size) {
        if (cancelled.load()) { throw std::runtime_error("cancelled"); }
        const int count = std::min(batch_size, end - start);
        auto batch = llama_batch_init(count, 5120, 1);
        batch.n_tokens = count;
        std::free(batch.pos);
        batch.pos = static_cast<llama_pos *>(std::malloc(4 * count * sizeof(llama_pos)));
        std::memcpy(batch.embd, rows.data() + start * 5120, count * 5120 * sizeof(float));
        for (int i = 0; i < count; ++i) {
            for (int axis = 0; axis < 4; ++axis) { batch.pos[axis * count + i] = pos; }
            ++pos;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = false;
        }
        const int rc = decode(batch);
        llama_batch_free(batch);
        if (rc) { throw std::runtime_error("audio decode failed"); }
    }
}

void brain_session::prefill(const request & request, const std::string & prompt, int & pos, size_t & used,
                            const std::string & stop_marker, size_t stop_rows, const stage_callback & on_stage) {
    size_t offset = 0;
    const auto & audio_rows = request.audio_rows;
    const bool partial = !partial_marker.empty() && prompt.compare(0, partial_prefix.size(), partial_prefix) == 0 &&
        prompt.compare(partial_prefix.size(), partial_marker.size(), partial_marker) == 0 &&
        frankie_token_boundary(llama_model_get_vocab(model.get()), prompt, partial_prefix.size());
    const auto * vocab = llama_model_get_vocab(model.get());
    if (partial) {
        offset = partial_prefix.size();
        pos = partial_pos;
        used = partial_used;
        if (on_stage) { on_stage("cache_precommit_append"); }
    } else if (cache_valid && prompt.compare(0, cached_prefix.size(), cached_prefix) == 0 &&
        frankie_token_boundary(vocab, prompt, cached_prefix.size())) {
        if (on_stage) { on_stage("cache_append"); }
        offset = cached_prefix.size();
        pos = cached_pos;
        used = cached_used;
    } else if (checkpoint && prompt.compare(0, checkpoint_prefix.size(), checkpoint_prefix) == 0 &&
               frankie_token_boundary(vocab, prompt, checkpoint_prefix.size())) {
        restore_sequence(*checkpoint);
        if (on_stage) { on_stage("cache_checkpoint_restore"); }
        offset = checkpoint_prefix.size();
        pos = checkpoint_pos;
        used = checkpoint_used;
    } else {
        clear_sequence();
        if (on_stage) { on_stage("cache_replay"); }
    }
    cache_valid = false;
    for (;;) {
        size_t marker = prompt.find("[FRANKIE_", offset);
        if (marker == std::string::npos) {
            decode_text(prompt.substr(offset), pos, used);
            break;
        }
        decode_text(prompt.substr(offset, marker - offset), pos, used);
        size_t end = prompt.find(']', marker);
        if (end == std::string::npos) {
            throw std::runtime_error("broken audio marker");
        }
        const auto key   = prompt.substr(marker, end - marker + 1);
        auto       image = request.images.find(key);
        if (image != request.images.end()) {
            mtmd_input_part part{};
            part.bitmap                     = image->second.get();
            const mtmd_input_part * parts[] = { &part };
            mtmd::input_chunks_ptr  chunks(mtmd_input_chunks_init());
            if (mtmd_tokenize_from_parts(vision.get(), chunks.get(), parts, 1, false)) {
                throw std::runtime_error("image tokenization failed");
            }
            used += mtmd_helper_get_n_tokens(chunks.get());
            if (used + 512 > context_tokens()) {
                throw std::runtime_error("image context exhausted");
            }
            if (mtmd_helper_eval_chunks_with_callback(vision.get(), ctx.get(), chunks.get(), pos, 0,
                    std::min(128u, llama_n_batch(ctx.get())), false, &pos,
                    [](llama_batch batch, void * user) {
                        auto * self = static_cast<brain_session *>(user);
                        return self->speculative && !common_speculative_process(self->speculative.get(), batch) ? -1 : 0;
                    }, this)) {
                throw std::runtime_error("image evaluation failed");
            }
            offset = end + 1;
            continue;
        }
        auto found = audio_rows.find(key);
        if (found == audio_rows.end()) {
            throw std::runtime_error("unknown audio marker");
        }
        const auto & rows = found->second;
        const size_t nf = key == stop_marker ? stop_rows : rows.size() / 5120;
        const size_t start = partial && key == partial_marker ? partial_rows.size() / 5120 : 0;
        decode_rows(rows, start, nf, pos, used);
        if (key == stop_marker) { return; }
        offset = end + 1;
    }
}

void brain_session::precommit(const request & input, const std::string & marker, const std::vector<float> & rows, size_t count) {
    std::lock_guard<std::recursive_mutex> compute_lock(compute_mutex);
    if (count == 0 || rows.size() % 5120 || count >= rows.size() / 5120 || count >= audio_row_limit()) {
        throw std::runtime_error("precommit row bounds");
    }
    auto snapshot = input;
    snapshot.audio_rows[marker] = rows;
    const auto formatted = common_chat_templates_apply(templates.get(), snapshot.chat);
    const auto cut = formatted.prompt.find(marker);
    if (cut == std::string::npos || formatted.prompt.find(marker, cut + 1) != std::string::npos) {
        throw std::runtime_error("precommit marker missing or ambiguous");
    }
    const auto prefix = formatted.prompt.substr(0, cut);
    if (partial_marker == marker && partial_prefix == prefix && count <= partial_rows.size() / 5120) { return; }
    if (partial_marker != marker || partial_prefix != prefix) {
        partial_marker.clear();
        partial_rows.clear();
    }
    const size_t retained = partial_rows.size();
    std::copy(partial_rows.begin(), partial_rows.end(), snapshot.audio_rows[marker].begin());
    if (formatted.prompt.size() > 4 * 1024 * 1024 || input.chat.messages.size() > max_messages) {
        throw std::runtime_error("precommit prompt bounds");
    }
    int pos = 0;
    size_t used = 0;
    try {
        prefill(snapshot, formatted.prompt, pos, used, marker, count, {});
        llama_synchronize(ctx.get());
        partial_rows.resize(count * 5120);
        std::copy(rows.begin() + retained, rows.begin() + count * 5120, partial_rows.begin() + retained);
        partial_prefix = prefix;
        partial_marker = marker;
        partial_pos = pos;
        partial_used = used;
    } catch (...) {
        partial_marker.clear();
        partial_rows.clear();
        cache_valid = false;
        throw;
    }
}

void brain_session::finish_audio(const request & input, const std::string & marker, std::vector<float> & rows) {
    std::lock_guard<std::recursive_mutex> compute_lock(compute_mutex);
    const auto formatted = common_chat_templates_apply(templates.get(), input.chat);
    if (marker == partial_marker && formatted.prompt.compare(0, partial_prefix.size(), partial_prefix) == 0 &&
        formatted.prompt.compare(partial_prefix.size(), marker.size(), marker) == 0) {
        if (rows.size() < partial_rows.size() + 5120) { throw std::runtime_error("final audio shorter than precommit"); }
        std::copy(partial_rows.begin(), partial_rows.end(), rows.begin());
    }
}

void brain_session::configure_thinking(common_params_sampling & sampling, const common_chat_params & formatted, int budget) const {
    if (budget <= 0) { return; }
    if (formatted.thinking_start_tag.empty() || formatted.thinking_end_tags.empty()) {
        throw std::runtime_error("chat template does not support bounded thinking");
    }
    const auto * vocab = llama_model_get_vocab(model.get());
    sampling.reasoning_budget_tokens = budget;
    sampling.reasoning_budget_start = common_tokenize(vocab, formatted.thinking_start_tag, false, true);
    for (const auto & tag : formatted.thinking_end_tags) {
        sampling.reasoning_budget_end.push_back(common_tokenize(vocab, tag, false, true));
    }
    sampling.reasoning_budget_forced = sampling.reasoning_budget_end.front();
}

brain_session::response brain_session::generate(const request & request, const stream_callback & on_text, const stage_callback & on_stage) {
    std::lock_guard<std::recursive_mutex> compute_lock(compute_mutex);
    const auto & input      = request.chat;
    const auto & audio_rows = request.audio_rows;
    if (request.reasoning_budget < 0 || request.reasoning_budget > 32768 ||
        prompt_tokens(request, {}) + request.generation_tokens() > context_tokens()) {
        throw std::runtime_error("reasoning or generation context budget exceeded");
    }
    if (input.messages.size() > max_messages || input.tools.size() > 32 || audio_rows.size() > max_audio_segments || request.images.size() > 4) {
        throw std::runtime_error("request bounds");
    }
    for (const auto & audio : audio_rows) {
        if (audio.first.empty() || audio.second.empty() || audio.second.size() % 5120 ||
            audio.second.size() > audio_row_limit() * 5120) {
            throw std::runtime_error("audio row shape");
        }
        for (float value : audio.second) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("nonfinite audio row");
            }
        }
    }
    auto formatted = common_chat_templates_apply(templates.get(), input);
    if (formatted.prompt.size() > 4 * 1024 * 1024) {
        throw std::runtime_error("formatted prompt too large");
    }
    int pos = 0;
    size_t used = 0;
    try {
        prefill(request, formatted.prompt, pos, used, "", 0, on_stage);
    } catch (...) {
        partial_marker.clear();
        partial_rows.clear();
        cache_valid = false;
        throw;
    }
    partial_marker.clear();
    partial_rows.clear();
    if (on_stage) { on_stage("brain_prefill_done"); }
    save_checkpoint(formatted.prompt, pos, used);
    if (on_stage) { on_stage("brain_checkpoint_done"); }
    response               result;
    auto sampling = frankie_brain_sampling(input.enable_thinking);
    sampling.penalty_present = options.presence_penalty;
    sampling.penalty_last_n  = request.generation_tokens();
    sampling.seed            = 42;
    sampling.generation_prompt = formatted.generation_prompt;
    configure_thinking(sampling, formatted, request.reasoning_budget);
    std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler(common_sampler_init(model.get(), sampling),
                                                                            common_sampler_free);
    if (!sampler) {
        throw std::runtime_error("brain sampler failed");
    }
    common_chat_parser_params parser(formatted);
    parser.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    if (!formatted.parser.empty()) {
        parser.parser.load(formatted.parser);
    }
    bool finished = false;
    size_t answer_tokens = 0, generated_tokens = 0, drafted_tokens = 0, accepted_tokens = 0;
    const size_t maximum = std::min(context_tokens() - used,
        request.answer_limit ? size_t(request.answer_limit) + request.reasoning_budget + 16 : context_tokens());
    const auto * vocab = llama_model_get_vocab(model.get());
    llama_token pending = common_sampler_sample(sampler.get(), ctx.get(), -1);
    common_sampler_accept(sampler.get(), pending, true);
    if (on_stage) { on_stage("brain_first_token"); }
    const llama_tokens prompt_unused;
    while (generated_tokens < maximum) {
        if (cancelled.load()) { throw std::runtime_error("cancelled"); }
        if (llama_vocab_is_eog(vocab, pending)) { finished = true; break; }
        llama_tokens draft;
        if (speculative && generated_tokens + 1 < maximum) {
            common_speculative_get_draft_params(speculative.get(), 0) = {
                true, int32_t(std::min<size_t>(options.mtp_tokens, maximum - generated_tokens - 1)),
                pos, pending, &prompt_unused, &draft,
            };
            common_speculative_draft(speculative.get());
            // An EOG is sampled by the target, never committed as a speech row.
            draft.erase(std::find_if(draft.begin(), draft.end(), [&](llama_token t) {
                return llama_vocab_is_eog(vocab, t);
            }), draft.end());
            if (!llama_memory_seq_rm(llama_get_memory(draft_ctx.get()), 0, pos, -1)) {
                throw std::runtime_error("MTP draft rollback failed");
            }
        }
        auto batch = llama_batch_init(1 + draft.size(), 0, 1);
        common_batch_add(batch, pending, pos, { 0 }, true);
        for (size_t j = 0; j < draft.size(); ++j) {
            common_batch_add(batch, draft[j], pos + 1 + j, { 0 }, true);
        }
        const int rc = decode(batch);
        llama_batch_free(batch);
        if (rc) { throw std::runtime_error("generation decode failed"); }
        // The last sampled token has not executed. It becomes the next anchor;
        // only the current anchor and verified draft rows can feed the voice bridge.
        auto ids = common_sampler_sample_and_accept_n(sampler.get(), ctx.get(), draft);
        const size_t committed = ids.size();
        if (speculative) {
            common_speculative_accept(speculative.get(), 0, committed - 1);
            if (committed <= draft.size()) {
                // The retained anchor keeps rollback inside this verification batch.
                if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, pos + committed, -1) ||
                    !llama_memory_seq_rm(llama_get_memory(draft_ctx.get()), 0, pos + committed, -1)) {
                    throw std::runtime_error("MTP accepted-prefix rollback failed");
                }
            }
        }
        pos += committed;
        drafted_tokens += draft.size();
        accepted_tokens += committed - 1;
        const float * rows = llama_get_embeddings_layer_inp(ctx.get(), 17);
        for (size_t j = 0; j < committed; ++j) {
            const llama_token token = j ? ids[j - 1] : pending;
            std::array<float, 5120> row;
            std::copy_n(rows + j * row.size(), row.size(), row.begin());
            for (float v : row) {
                if (!std::isfinite(v)) { throw std::runtime_error("nonfinite hidden state"); }
            }
            result.raw.text += common_token_to_piece(vocab, token, true);
            ++generated_tokens;
            result.message = common_chat_parse(result.raw.text, true, parser);
            result.message.role = "assistant";
            if (!result.message.content.empty() || !result.message.tool_calls.empty()) {
                if (++answer_tokens > request.answer_limit && request.answer_limit) {
                    if (speculative && std::getenv("FRANKIE_PROFILE")) {
                        std::cerr << "brain_mtp output_limit row=" << j << " committed=" << committed
                                  << " drafted=" << drafted_tokens << " accepted=" << accepted_tokens << "\n";
                    }
                    throw output_limit("max_output_tokens");
                }
                // Reasoning never feeds the voice bridge. Retain only audible/answer rows.
                result.raw.ends.push_back(result.raw.text.size());
                result.raw.hidden.push_back(row);
            }
            if (!result.message.content.empty() && (result.content_offset == std::string::npos ||
                result.raw.text.compare(result.content_offset, result.message.content.size(), result.message.content) != 0)) {
                result.content_offset = result.raw.text.rfind(result.message.content);
            }
        }
        pending = ids.back();
        // Copy every committed row before callbacks can pump HTTP work.
        if (on_text) { on_text(result, false); }
        if (background_step) { background_step(); }
    }
    if (speculative) {
        std::cerr << "brain_mtp drafted=" << drafted_tokens << " accepted=" << accepted_tokens << "\n";
    }
    if (!finished) {
        throw output_limit("max_output_tokens");
    }
    result.message      = common_chat_parse(result.raw.text, false, parser);
    result.message.role = "assistant";
    if (result.message.tool_calls.size() > 1) {
        throw std::runtime_error("parallel calls not enabled");
    }
    for (auto & call : result.message.tool_calls) {
        bool known = false;
        for (const auto & tool : input.tools) {
            if (tool.name == call.name) {
                known = true;
            }
        }
        if (!known) {
            throw std::runtime_error("unadvertised tool");
        }
    }
    if (cancelled.load()) {
        throw std::runtime_error("cancelled");
    }
    if (on_text) {
        on_text(result, true);
    }
    cached_prefix = formatted.prompt + result.raw.text;
    cached_pos = pos;
    cached_used = used + generated_tokens;
    cache_valid = true;
    std::cerr << "brain generated_tokens=" << generated_tokens << " reasoning_budget=" << request.reasoning_budget
              << " reasoning_chars=" << result.message.reasoning_content.size() << " context_used=" << cached_used << "\n";
    return result;
}

void brain_session::report_memory() const {
    for (const auto & entry : {std::make_pair("brain", ctx.get()), std::make_pair("mtp", draft_ctx.get())}) {
        if (!entry.second) { continue; }
        for (const auto & [type, bytes] : llama_get_memory_breakdown(entry.second)) {
            const size_t weights = entry.second == ctx.get() ? bytes.model : 0;
            std::cerr << "memory " << entry.first << " backend=" << ggml_backend_buft_name(type) << " weights=" << weights
                      << " context=" << bytes.context << " compute=" << bytes.compute << "\n";
        }
    }
    for (const auto & [device, bytes] : clip_get_mem_usage(encoder.get())) {
        std::cerr << "memory parakeet backend=" << ggml_backend_dev_name(device) << " bytes=" << bytes << "\n";
    }
    for (const auto & [device, bytes] : mtmd_get_memory_usage(vision.get())) {
        std::cerr << "memory vision backend=" << ggml_backend_dev_name(device) << " bytes=" << bytes << "\n";
    }
}
