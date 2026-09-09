#include "brain-session.h"

#include "llama-ext.h"
#include "mtmd-audio.h"
#include "sampling.h"
#include "stb/stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

brain_session::brain_session(const std::string & package) :
    encoder_source(package, "ear"),
    bridge_source(package, "bridge"),
    brain_source(package, "brain"),
    vision_source(package, "vision") {
    clip_context_params ep{};
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
    weights.reset(raw);
    if (!metadata || !weights) {
        throw std::runtime_error("bridge load failed");
    }
    ear             = std::make_unique<mtmd_ear>(raw);
    auto mp         = llama_model_default_params();
    mp.n_gpu_layers = 99;
    model.reset(llama_model_init_from_user(brain_source.metadata(), component::set_tensor, &brain_source, mp));
    if (!model) {
        throw std::runtime_error("brain load failed");
    }
    auto cp              = llama_context_default_params();
    cp.n_ctx             = 4096;
    cp.n_batch           = 512;
    cp.n_ubatch          = 128;
    cp.n_threads         = 4;
    cp.n_threads_batch   = 4;
    cp.cb_eval_user_data = &tap;
    cp.cb_eval           = [](ggml_tensor * t, bool ask, void * user) {
        auto & tap    = *static_cast<capture *>(user);
        bool   wanted = tap.enabled && std::strcmp(ggml_get_name(t), "l_out-16") == 0;
        if (ask) {
            return wanted;
        }
        if (wanted) {
            if (t->type != GGML_TYPE_F32 || ggml_nelements(t) != 5120 || !ggml_is_contiguous(t)) {
                return false;
            }
            ggml_backend_tensor_get(t, tap.row.data(), 0, sizeof(tap.row));
            ++tap.calls;
        }
        return true;
    };
    cp.abort_callback = [](void * user) {
        return static_cast<brain_session *>(user)->cancelled.load();
    };
    cp.abort_callback_data = this;
    ctx.reset(llama_init_from_model(model.get(), cp));
    if (!ctx) {
        throw std::runtime_error("brain context failed");
    }
    auto vp                   = mtmd_context_params_default();
    vp.use_gpu                = true;
    vp.n_threads              = 4;
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

std::vector<float> brain_session::encode_audio(const std::vector<float> & pcm) {
    if (pcm.empty() || pcm.size() > 16000 * 20) {
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
    clip_image_f32 image;
    image.set_size({ int(mel[0].n_len), 80 }, false, true);
    image.cpy_buf(mel[0].data);
    size_t nf = clip_n_output_tokens(encoder.get(), &image);
    if (nf == 0 || nf > 1024) {
        throw std::runtime_error("encoder frame bounds");
    }
    std::vector<float>   frames(nf * 512);
    clip_image_f32_batch batch;
    batch.is_audio = true;
    batch.entries.push_back(std::move(image));
    if (!clip_image_batch_encode(encoder.get(), 4, &batch, frames)) {
        throw std::runtime_error("encoder failed");
    }
    return ear->process(frames.data(), nf);
}

brain_session::image_ptr brain_session::decode_image(const std::vector<unsigned char> & bytes) {
    int nx = 0, ny = 0, nc = 0;
    if (bytes.empty() || bytes.size() > 512 * 1024 ||
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
    if (n < 0 || used + n + 512 > 4096) {
        throw std::runtime_error("prompt context exhausted");
    }
    used += n;
    for (int start = 0; start < n; start += 128) {
        if (cancelled.load()) {
            throw std::runtime_error("cancelled");
        }
        int  count     = std::min(128, n - start);
        auto batch     = llama_batch_init(count, 0, 1);
        batch.n_tokens = count;
        for (int i = 0; i < count; ++i) {
            batch.token[i]     = tokens[start + i];
            batch.pos[i]       = pos++;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (start + i == n - 1);
        }
        int rc = llama_decode(ctx.get(), batch);
        llama_batch_free(batch);
        if (rc) {
            throw std::runtime_error("text decode failed");
        }
    }
}

// The chat template preserves internal audio markers; each marker is replaced
// with learned ear rows during prefill. The server rejects markers in external
// text, instructions and tool output. Each response clears and rebuilds KV.
brain_session::response brain_session::generate(const request & request) {
    const auto & input      = request.chat;
    const auto & audio_rows = request.audio_rows;
    if (input.messages.size() > 32 || input.tools.size() > 32 || audio_rows.size() > 8 || request.images.size() > 4) {
        throw std::runtime_error("request bounds");
    }
    for (const auto & audio : audio_rows) {
        if (audio.first.empty() || audio.second.empty() || audio.second.size() % 5120 ||
            audio.second.size() > 1025 * 5120) {
            throw std::runtime_error("audio row shape");
        }
        for (float value : audio.second) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("nonfinite audio row");
            }
        }
    }
    auto formatted = common_chat_templates_apply(templates.get(), input);
    if (formatted.prompt.size() > 128 * 1024) {
        throw std::runtime_error("formatted prompt too large");
    }
    llama_memory_clear(llama_get_memory(ctx.get()), true);
    tap.enabled   = false;
    int    pos    = 0;
    size_t offset = 0, used = 0;
    for (;;) {
        size_t marker = formatted.prompt.find("[FRANKIE_", offset);
        if (marker == std::string::npos) {
            decode_text(formatted.prompt.substr(offset), pos, used);
            break;
        }
        decode_text(formatted.prompt.substr(offset, marker - offset), pos, used);
        size_t end = formatted.prompt.find(']', marker);
        if (end == std::string::npos) {
            throw std::runtime_error("broken audio marker");
        }
        const auto key   = formatted.prompt.substr(marker, end - marker + 1);
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
            if (used + 512 > 4096) {
                throw std::runtime_error("image context exhausted");
            }
            if (mtmd_helper_eval_chunks(vision.get(), ctx.get(), chunks.get(), pos, 0, 128, false, &pos)) {
                throw std::runtime_error("image evaluation failed");
            }
            offset = end + 1;
            continue;
        }
        auto found = audio_rows.find(key);
        if (found == audio_rows.end()) {
            throw std::runtime_error("unknown audio marker");
        }
        auto & rows = found->second;
        size_t nf   = rows.size() / 5120;
        if (used + nf + 512 > 4096) {
            throw std::runtime_error("audio context exhausted");
        }
        used += nf;
        for (size_t start = 0; start < nf; start += 128) {
            if (cancelled.load()) {
                throw std::runtime_error("cancelled");
            }
            int  count     = std::min(size_t(128), nf - start);
            auto batch     = llama_batch_init(count, 5120, 1);
            batch.n_tokens = count;
            std::memcpy(batch.embd, rows.data() + start * 5120, count * 5120 * sizeof(float));
            for (int i = 0; i < count; ++i) {
                batch.pos[i]       = pos++;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = false;
            }
            int rc = llama_decode(ctx.get(), batch);
            llama_batch_free(batch);
            if (rc) {
                throw std::runtime_error("audio decode failed");
            }
        }
        offset = end + 1;
    }
    response               result;
    common_params_sampling sampling;
    sampling.temp            = 0.7f;
    sampling.top_p           = 0.8f;
    sampling.top_k           = 20;
    sampling.min_p           = 0.0f;
    sampling.penalty_present = 1.5f;
    sampling.penalty_last_n  = -1;
    sampling.seed            = 42;
    std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler(common_sampler_init(model.get(), sampling),
                                                                            common_sampler_free);
    if (!sampler) {
        throw std::runtime_error("brain sampler failed");
    }
    bool finished = false;
    for (int i = 0; i < 512; ++i) {
        if (cancelled.load()) {
            throw std::runtime_error("cancelled");
        }
        llama_token token = common_sampler_sample(sampler.get(), ctx.get(), -1);
        common_sampler_accept(sampler.get(), token, true);
        auto * vocab = llama_model_get_vocab(model.get());
        if (llama_vocab_is_eog(vocab, token)) {
            finished = true;
            break;
        }
        char buf[512];
        int  n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (n < 0) {
            throw std::runtime_error("token piece overflow");
        }
        tap.enabled        = true;
        tap.calls          = 0;
        auto batch         = llama_batch_init(1, 0, 1);
        batch.n_tokens     = 1;
        batch.token[0]     = token;
        batch.pos[0]       = pos++;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = true;
        int rc             = llama_decode(ctx.get(), batch);
        llama_batch_free(batch);
        llama_synchronize(ctx.get());
        tap.enabled = false;
        if (rc || tap.calls != 1) {
            throw std::runtime_error("generation/tap failed");
        }
        for (float v : tap.row) {
            if (!std::isfinite(v)) {
                throw std::runtime_error("nonfinite hidden state");
            }
        }
        result.raw.text.append(buf, n);
        result.raw.ends.push_back(result.raw.text.size());
        result.raw.hidden.push_back(tap.row);
    }
    if (!finished) {
        throw std::runtime_error("output token limit");
    }
    common_chat_parser_params parser(formatted);
    parser.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    if (!formatted.parser.empty()) {
        parser.parser.load(formatted.parser);
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
    return result;
}
