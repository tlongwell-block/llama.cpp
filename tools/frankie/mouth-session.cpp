#include "mouth-session.h"

#include "llama-ext.h"
#include "text-alignment.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <iostream>

mouth_session::mouth_session(const std::string & package, const frankie_options & config) :
    talker_source(package, "talker", config.talker_model),
    mouth_source(package, "mouth", config.mouth_model),
    side_source(package, "side"), options(config) {
    auto mp         = llama_model_default_params();
    mp.n_gpu_layers = options.use_gpu ? 99 : 0;
    model.reset(llama_model_init_from_user(talker_source.metadata(), component::set_tensor, &talker_source, mp));
    if (!model) {
        throw std::runtime_error("talker load failed");
    }
    auto cp            = llama_context_default_params();
    cp.n_ctx           = 2048;
    cp.n_batch         = 256;
    cp.n_ubatch        = 128;
    cp.n_threads       = options.threads;
    cp.n_threads_batch = options.threads;
    cp.embeddings      = true;
    cp.offload_kqv     = options.use_gpu;
    cp.op_offload      = options.use_gpu;
    cp.abort_callback  = [](void * user) {
        return static_cast<mouth_session *>(user)->cancelled.load();
    };
    cp.abort_callback_data = this;
    ctx.reset(llama_init_from_model(model.get(), cp));
    if (!ctx) {
        throw std::runtime_error("talker context failed");
    }
    auto params                   = mtmd_context_params_default();
    params.use_gpu                = options.use_gpu;
    params.n_threads              = options.threads;
    params.model_reader           = component::callback;
    params.model_reader_user_data = &mouth_source;
    params.model_reader_size      = mouth_source.size();
    mctx.reset(mtmd_init_from_file("embedded-mouth", model.get(), params));
    if (!mctx) {
        throw std::runtime_error("mouth load failed");
    }
    generator = std::make_unique<mtmd_helper::gen_audio>(ctx.get(), mctx.get());
    if (options.side_scale != 0.0f) {
        ggml_context *                                      raw = nullptr;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_from_callback(component::callback, &side_source, 1024 * 1024, side_source.size(), { false, &raw }),
            gguf_free);
        side_weights.reset(raw);
        if (!metadata || !side_weights) {
            throw std::runtime_error("Side load failed");
        }
        side = std::make_unique<mtmd_side>(raw, options.use_gpu);
        embeddings.resize(llama_model_get_tok_embd(model.get(), nullptr));
        if (embeddings.empty() || llama_model_get_tok_embd(model.get(), embeddings.data()) != embeddings.size()) {
            throw std::runtime_error("talker embeddings failed");
        }
    }
    std::vector<char> bytes, wav;
    if (options.voice.empty()) {
        bytes = mouth_source.asset("assets.voice.codes");
        reference = mouth_source.string_value("frankie.voice.ref_text");
        wav = mouth_source.asset("assets.voice.wav");
    } else {
        wav = frankie_read_file(options.voice, 16 * 1024 * 1024);
        if (!options.voice_codes.empty()) {
            bytes = frankie_read_file(options.voice_codes, 250 * 16 * sizeof(int32_t));
            const auto text = frankie_read_file(options.voice_text, 8192);
            reference.assign(text.begin(), text.end());
        }
    }
    if ((!bytes.empty() && bytes.size() % (16 * sizeof(int32_t))) || bytes.size() > 250 * 16 * sizeof(int32_t)) {
        throw std::runtime_error("reference codec shape");
    }
    codes.resize(bytes.size() / 4);
    if (!bytes.empty()) { std::memcpy(codes.data(), bytes.data(), bytes.size()); }
    for (auto code : codes) { if (code < 0 || code >= 2048) { throw std::runtime_error("reference codec ID out of range"); } }
    if (wav.size() < 44 || std::memcmp(wav.data(), "RIFF", 4) || std::memcmp(wav.data() + 8, "WAVE", 4)) {
        throw std::runtime_error("voice WAV");
    }
    auto decoded = mtmd_helper_bitmap_init_from_buf(mctx.get(), reinterpret_cast<const unsigned char *>(wav.data()),
                                                    wav.size(), false, mtmd_helper_init_opt_default());
    speaker.reset(decoded.bitmap);
    if (decoded.video_ctx) {
        mtmd_helper_video_free(decoded.video_ctx);
        throw std::runtime_error("packaged voice is video");
    }
    if (!speaker || !mtmd_bitmap_is_audio(speaker.get()) ||
        mtmd_bitmap_get_n_bytes(speaker.get()) > 24000 * 20 * sizeof(float)) {
        throw std::runtime_error("voice must decode to at most 20 seconds of audio");
    }
    const auto * samples = reinterpret_cast<const float *>(mtmd_bitmap_get_data(speaker.get()));
    const size_t n = mtmd_bitmap_get_n_bytes(speaker.get()) / sizeof(float);
    double energy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(samples[i])) { throw std::runtime_error("nonfinite voice sample"); }
        energy += double(samples[i]) * samples[i];
    }
    reference_rms = n ? std::sqrt(energy / n) : 0.0f;
    if (reference_rms < 1e-6f) { throw std::runtime_error("voice reference is silent"); }

    std::vector<char> expression_bytes;
    if (!options.expression.empty()) {
        expression_bytes = frankie_read_file(options.expression, 1024 * 1024);
    } else if (mouth_source.has_asset("assets.expression.gguf")) {
        expression_bytes = mouth_source.asset("assets.expression.gguf");
    }
    if (!expression_bytes.empty()) {
        ggml_context * raw = nullptr;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_from_buffer(expression_bytes.data(), expression_bytes.size(), {false, &raw}), gguf_free);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> weights(raw, ggml_free);
        if (!metadata || !weights) { throw std::runtime_error("expression load failed"); }
        expression = std::make_unique<mtmd_expression>(raw, options.use_gpu);
    }
}

std::vector<float> mouth_session::speak(const brain_session::response & response, const audio_callback & on_audio, const brain_session::stage_callback & on_stage) {
    aside_cancelled = true;
    std::lock_guard<std::mutex> lock(execution);
    return speak_impl(response, on_audio, on_stage, nullptr);
}

void mouth_session::warmup() {
    brain_session::response text;
    text.message.content = text.raw.text = "Warming up the mouth.";
    const plain_style style{1.0f, 3.0f, 256, -1, false};
    std::lock_guard<std::mutex> lock(execution);
    for (int i = 0; i < 2; ++i) { speak_impl(text, {}, {}, &style); }
}

void mouth_session::speak_backchannel(int reaction, bool strong, const audio_callback & on_audio) {
    if (reaction < 0 || reaction > 3) { throw std::runtime_error("invalid listener reaction"); }
    const char * normal[] = {"Hmm.", "Yeah.", "Oh wow.", "Oh no."};
    const char * emphatic[] = {"Mm-hmm.", "Yeah, yeah.", "Oh, wow!", "Oh no."};
    brain_session::response text;
    text.message.content = text.raw.text = strong ? emphatic[reaction] : normal[reaction];
    const plain_style style{0.6f, strong ? -3.0f : -6.0f, 16, reaction == 2 ? 1 : reaction == 3 ? 2 : -1, true};
    std::lock_guard<std::mutex> lock(execution);
    speak_impl(text, on_audio, {}, &style);
}

std::vector<float> mouth_session::speak_impl(const brain_session::response & response, const audio_callback & on_audio,
                                            const brain_session::stage_callback & on_stage, const plain_style * style) {
    auto is_cancelled = [&] { return cancelled.load() || (style && style->aside && aside_cancelled.load()); };
    if (on_stage) { on_stage("mouth_phrase_start"); }
    if (!response.message.tool_calls.empty()) {
        throw std::runtime_error("cannot speak tool arguments");
    }
    const auto & brain  = response.raw;
    std::string  spoken = response.message.content;
    const frankie_text_offsets input_offsets(spoken);
    size_t begin = 0, end = spoken.size();
    while (begin < end && input_offsets.whitespace[input_offsets.floor[begin]]) { ++begin; }
    while (end > begin && input_offsets.whitespace[input_offsets.floor[end - 1]]) { --end; }
    if (begin == end) {
        throw std::runtime_error("empty spoken response");
    }
    spoken = spoken.substr(begin, end - begin);
    if (spoken.size() > 8192 || brain.hidden.size() != brain.ends.size()) {
        throw std::runtime_error("response shape/bounds");
    }
    size_t lead = response.content_offset == std::string::npos ? brain.text.find(spoken) : response.content_offset + begin;
    if (lead == std::string::npos || brain.text.compare(lead, spoken.size(), spoken) != 0) {
        throw std::runtime_error("parsed content is not a contiguous generated span");
    }
    const frankie_normalized_text normalized(spoken);
    std::array<float, 2048> speaker_offset{};
    if (expression && style && style->emotion >= 0) {
        speaker_offset = expression->direction(style->emotion);
    } else if (expression && !style) {
        std::array<float, 5120> pooled{};
        size_t count = 0;
        const frankie_text_offsets offsets(brain.text);
        for (size_t i = 0; i < brain.ends.size(); ++i) {
            const size_t a = std::max(lead, i ? brain.ends[i - 1] : 0);
            const size_t b = std::min(lead + spoken.size(), brain.ends[i]);
            bool keep = false;
            for (size_t j = a; j < b; ++j) { keep |= !offsets.whitespace[offsets.floor[j]]; }
            if (!keep) { continue; }
            for (size_t j = 0; j < pooled.size(); ++j) { pooled[j] += brain.hidden[i][j]; }
            ++count;
        }
        if (!count) { throw std::runtime_error("missing expression states for spoken span"); }
        for (auto & v : pooled) { v /= count; }
        const auto result = expression->process(pooled);
        speaker_offset = result.offset;
        if (on_stage) { on_stage("mouth_expression_done"); }
    }
    std::vector<float> target;
    if (side && !style) {
    auto * vocab = llama_model_get_vocab(model.get());
    std::vector<llama_token> ids(normalized.text.size() + 16);
    int count = llama_tokenize(vocab, normalized.text.data(), normalized.text.size(), ids.data(), ids.size(), false, false);
    if (count <= 0 || count > 256) { throw std::runtime_error("talker token bounds"); }
    ids.resize(count);
    std::string roundtrip;
    std::vector<size_t> talker_ends;
    for (auto id : ids) {
        char piece[512];
        int n = llama_token_to_piece(vocab, id, piece, sizeof(piece), 0, false);
        if (n <= 0 || n > int(sizeof(piece))) { throw std::runtime_error("talker piece bounds"); }
        roundtrip.append(piece, n);
        talker_ends.push_back(roundtrip.size());
    }
    if (roundtrip != normalized.text) { throw std::runtime_error("talker token roundtrip"); }
    const auto aligned = frankie_align_text(brain.text, brain.ends, lead, spoken, talker_ends, normalized.spans(talker_ends));
    target.reserve(ids.size() * 2048);
    for (size_t i = 0; i < ids.size(); ++i) {
        const auto id = ids[i];
        const size_t chosen = aligned[i];
        if (chosen >= brain.hidden.size() || id < 0 || size_t(id + 1) * 2048 > embeddings.size()) {
            throw std::runtime_error("alignment bounds");
        }
        auto residual = side->process(brain.hidden[chosen]);
        for (size_t j = 0; j < 2048; ++j) {
            target.push_back(embeddings[size_t(id) * 2048 + j] + options.side_scale * residual[j]);
        }
    }
    }
    llama_memory_clear(llama_get_memory(ctx.get()), true);
    auto & gen = *generator;
    mtmd_helper_gen_audio_inp input{};
    input.seq_id        = 0;
    input.prompt        = normalized.text.c_str();
    input.prompt_len    = normalized.text.size();
    input.speaker_ref   = speaker.get();
    input.lang          = "en";
    input.code_temperature = 0.6f;
    input.top_k         = 50;
    input.top_p         = 1.0f;
    input.seed          = 42;
    input.out_type      = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;
    input.ref_text      = reference.empty() ? nullptr : reference.c_str();
    input.ref_text_len  = reference.size();
    input.ref_codes     = codes.empty() ? nullptr : codes.data();
    input.n_ref_frames  = codes.size() / 16;
    input.prime_frames  = input.n_ref_frames;
    input.target_embd   = target.empty() ? nullptr : target.data();
    input.n_target_embd = target.size() / 2048;
    input.speaker_offset = expression ? speaker_offset.data() : nullptr;
    input.n_speaker_offset = expression ? speaker_offset.size() : 0;
    if (is_cancelled()) {
        throw std::runtime_error("cancelled");
    }
    if (on_stage) { on_stage("mouth_conditioning_done"); }
    if (gen.set_input(&input)) {
        throw std::runtime_error("mouth input failed");
    }
    if (on_stage) { on_stage("mouth_input_primed"); }
    for (;;) {
        if (is_cancelled()) {
            throw std::runtime_error("cancelled");
        }
        int rc = gen.step_prompt(256);
        if (rc < 0) {
            throw std::runtime_error("mouth prefill failed");
        }
        if (!rc) {
            break;
        }
    }
    if (on_stage) { on_stage("mouth_prefill_done"); }
    common_params_sampling sampling;
    sampling.temp = 0.6f;
    sampling.seed = 42;
    sampling.top_k = 50;
    sampling.top_p = 1.0f;
    sampling.min_p = 0.0f;
    sampling.penalty_repeat = codes.empty() ? 1.05f : 1.5f;
    sampling.penalty_last_n = 4096;
    std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler(common_sampler_init(model.get(), sampling),
                                                                            common_sampler_free);
    if (!sampler) {
        throw std::runtime_error("mouth sampler failed");
    }
    size_t emitted = 0;
    double energy = 0.0;
    float gain = style ? style->gain : 1.0f;
    std::vector<float> rendered;
    auto emit = [&]() {
        int32_t rate = 0;
        const char * data = nullptr;
        size_t bytes = 0;
        int64_t samples = 0;
        if (gen.get_output(&rate, &data, &bytes, &samples) || rate != 24000 || samples < 0 ||
            size_t(samples) < emitted || bytes > 4 * 1024 * 1024 || bytes != size_t(samples) * sizeof(float)) {
            throw std::runtime_error("streaming mouth output bounds");
        }
        const auto * pcm = reinterpret_cast<const float *>(data);
        for (size_t i = emitted; i < size_t(samples); ++i) {
            if (!std::isfinite(pcm[i])) {
                throw std::runtime_error("nonfinite streaming PCM");
            }
        }
        if (size_t(samples) > emitted) {
            if (emitted == 0 && on_stage) { on_stage("mouth_first_chunk"); }
            if (expression) {
                float peak = 0.0f;
                for (size_t i = emitted; i < size_t(samples); ++i) {
                    energy += double(pcm[i]) * pcm[i];
                    peak = std::max(peak, std::abs(pcm[i]));
                }
                if (peak > 0) { gain = std::min(gain, 0.9f / peak); }
                const float rms = std::sqrt(energy / samples);
                if (rms > 0) { gain = std::min(gain, reference_rms * std::pow(10.0f, (style ? style->rms_db : 3.0f) / 20.0f) / rms); }
            }
            rendered.reserve(samples);
            for (size_t i = emitted; i < size_t(samples); ++i) { rendered.push_back(pcm[i] * gain); }
            if (on_audio) { on_audio(rendered.data() + emitted, size_t(samples) - emitted); }
            emitted = samples;
        }
    };
    bool          stop   = false;
    const float * hidden = llama_get_embeddings_ith(ctx.get(), -1);
    for (int frame = 0; frame < (style ? style->frames : 512) && !stop; ++frame) {
        if (is_cancelled()) {
            throw std::runtime_error("cancelled");
        }
        auto token = common_sampler_sample(sampler.get(), ctx.get(), -1);
        common_sampler_accept(sampler.get(), token, true);
        const float * next = nullptr;
        if (gen.step_gen(token, hidden, &next, &stop)) {
            throw std::runtime_error("mouth generation failed");
        }
        emit();
        if (!next) {
            break;
        }
        hidden = next;
    }
    if (is_cancelled()) {
        throw std::runtime_error("cancelled");
    }
    emit();
    int32_t      rate    = 0;
    const char * data    = nullptr;
    size_t       bytes   = 0;
    int64_t      samples = 0;
    if ((!stop && !(style && style->aside)) || gen.get_output(&rate, &data, &bytes, &samples) || rate != 24000 || samples <= 0 ||
        bytes > 4 * 1024 * 1024 || bytes != size_t(samples) * sizeof(float)) {
        throw std::runtime_error("mouth output bounds or unfinished");
    }
    if (rendered.size() != size_t(samples)) { throw std::runtime_error("incomplete rendered PCM"); }
    return rendered;
}

void mouth_session::report_memory() const {
    for (const auto & [type, bytes] : llama_get_memory_breakdown(ctx.get())) {
        std::cerr << "memory talker backend=" << ggml_backend_buft_name(type) << " weights=" << bytes.model
                  << " context=" << bytes.context << " compute=" << bytes.compute << "\n";
    }
    for (const auto & [device, bytes] : mtmd_get_memory_usage(mctx.get())) {
        std::cerr << "memory mouth backend=" << ggml_backend_dev_name(device) << " bytes=" << bytes << "\n";
    }
}
