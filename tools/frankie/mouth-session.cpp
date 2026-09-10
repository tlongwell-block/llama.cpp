#include "mouth-session.h"

#include "llama-ext.h"
#include "text-alignment.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

mouth_session::mouth_session(const std::string & package) :
    talker_source(package, "talker"),
    mouth_source(package, "mouth"),
    side_source(package, "side") {
    auto mp         = llama_model_default_params();
    mp.n_gpu_layers = 99;
    model.reset(llama_model_init_from_user(talker_source.metadata(), component::set_tensor, &talker_source, mp));
    if (!model) {
        throw std::runtime_error("talker load failed");
    }
    auto cp            = llama_context_default_params();
    cp.n_ctx           = 2048;
    cp.n_batch         = 256;
    cp.n_ubatch        = 128;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    cp.embeddings      = true;
    cp.abort_callback  = [](void * user) {
        return static_cast<mouth_session *>(user)->cancelled.load();
    };
    cp.abort_callback_data = this;
    ctx.reset(llama_init_from_model(model.get(), cp));
    if (!ctx) {
        throw std::runtime_error("talker context failed");
    }
    auto params                   = mtmd_context_params_default();
    params.use_gpu                = true;
    params.model_reader           = component::callback;
    params.model_reader_user_data = &mouth_source;
    params.model_reader_size      = mouth_source.size();
    mctx.reset(mtmd_init_from_file("embedded-mouth", model.get(), params));
    if (!mctx) {
        throw std::runtime_error("mouth load failed");
    }
    generator = std::make_unique<mtmd_helper::gen_audio>(ctx.get(), mctx.get());
    ggml_context *                                      raw = nullptr;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
        gguf_init_from_callback(component::callback, &side_source, 1024 * 1024, side_source.size(), { false, &raw }),
        gguf_free);
    side_weights.reset(raw);
    if (!metadata || !side_weights) {
        throw std::runtime_error("Side load failed");
    }
    side = std::make_unique<mtmd_side>(raw, true);
    embeddings.resize(llama_model_get_tok_embd(model.get(), nullptr));
    if (embeddings.empty() || llama_model_get_tok_embd(model.get(), embeddings.data()) != embeddings.size()) {
        throw std::runtime_error("talker embeddings failed");
    }
    auto bytes = mouth_source.asset("assets.voice.codes");
    if (bytes.empty() || bytes.size() % (16 * sizeof(int32_t))) {
        throw std::runtime_error("reference codec shape");
    }
    codes.resize(bytes.size() / 4);
    std::memcpy(codes.data(), bytes.data(), bytes.size());
    reference = mouth_source.string_value("frankie.voice.ref_text");
    auto wav  = mouth_source.asset("assets.voice.wav");
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
        throw std::runtime_error("packaged voice decode/bounds");
    }
}

std::vector<float> mouth_session::speak(const brain_session::response & response, const audio_callback & on_audio, const brain_session::stage_callback & on_stage) {
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
    std::vector<float> target;
    target.reserve(ids.size() * 2048);
    for (size_t i = 0; i < ids.size(); ++i) {
        const auto id = ids[i];
        const size_t chosen = aligned[i];
        if (chosen >= brain.hidden.size() || id < 0 || size_t(id + 1) * 2048 > embeddings.size()) {
            throw std::runtime_error("alignment bounds");
        }
        auto residual = side->process(brain.hidden[chosen]);
        for (size_t j = 0; j < 2048; ++j) {
            target.push_back(embeddings[size_t(id) * 2048 + j] + 0.5f * residual[j]);
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
    input.top_k         = 40;
    input.top_p         = 0.9f;
    input.seed          = 42;
    input.out_type      = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;
    input.ref_text      = reference.c_str();
    input.ref_text_len  = reference.size();
    input.ref_codes     = codes.data();
    input.n_ref_frames  = codes.size() / 16;
    input.prime_frames  = input.n_ref_frames;
    input.target_embd   = target.data();
    input.n_target_embd = ids.size();
    if (cancelled.load()) {
        throw std::runtime_error("cancelled");
    }
    if (on_stage) { on_stage("mouth_conditioning_done"); }
    if (gen.set_input(&input)) {
        throw std::runtime_error("mouth input failed");
    }
    if (on_stage) { on_stage("mouth_input_primed"); }
    for (;;) {
        if (cancelled.load()) {
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
    std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler(common_sampler_init(model.get(), sampling),
                                                                            common_sampler_free);
    if (!sampler) {
        throw std::runtime_error("mouth sampler failed");
    }
    size_t emitted = 0;
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
            on_audio(pcm + emitted, size_t(samples) - emitted);
            emitted = samples;
        }
    };
    bool          stop   = false;
    const float * hidden = llama_get_embeddings_ith(ctx.get(), -1);
    for (int frame = 0; frame < 256 && !stop; ++frame) {
        if (cancelled.load()) {
            throw std::runtime_error("cancelled");
        }
        auto token = common_sampler_sample(sampler.get(), ctx.get(), -1);
        common_sampler_accept(sampler.get(), token, true);
        const float * next = nullptr;
        if (gen.step_gen(token, hidden, &next, &stop)) {
            throw std::runtime_error("mouth generation failed");
        }
        if (on_audio && ((frame + 1) % 3 == 0 || stop)) {
            emit();
        }
        if (!next) {
            break;
        }
        hidden = next;
    }
    if (cancelled.load()) {
        throw std::runtime_error("cancelled");
    }
    if (on_audio) {
        emit();
    }
    int32_t      rate    = 0;
    const char * data    = nullptr;
    size_t       bytes   = 0;
    int64_t      samples = 0;
    if (!stop || gen.get_output(&rate, &data, &bytes, &samples) || rate != 24000 || samples <= 0 ||
        bytes > 4 * 1024 * 1024 || bytes != size_t(samples) * sizeof(float)) {
        throw std::runtime_error("mouth output bounds or unfinished");
    }
    std::vector<float> pcm(samples);
    std::memcpy(pcm.data(), data, bytes);
    for (float sample : pcm) {
        if (!std::isfinite(sample)) {
            throw std::runtime_error("nonfinite generated PCM");
        }
    }
    return pcm;
}
