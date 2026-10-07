#include "mouth-session.h"

#include "llama-ext.h"
#include "text-alignment.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <iostream>

mouth_session::mouth_session(const std::string & package, const frankie_options & config,
                             const voice_transcriber & transcribe) :
    talker_source(package, "talker", config.talker_model),
    mouth_source(package, "mouth", config.mouth_model),
    side_source(package, "side"), options(config) {
    if (options.speech_context_words > 1000) { throw std::runtime_error("speech context must be zero to 1000 words"); }
    if (options.speech_hold_words > options.speech_context_words) { throw std::runtime_error("held speech must fit the speech context"); }
    auto * metadata = mouth_source.metadata();
    const auto variant = gguf_find_key(metadata, "clip.gen.audio.model_variant");
    const bool is_breeze = variant >= 0 && gguf_get_kv_type(metadata, variant) == GGUF_TYPE_STRING &&
        std::string(gguf_get_val_str(metadata, variant)) == "breeze";
    auto mp         = llama_model_default_params();
    mp.n_gpu_layers = options.use_gpu ? 99 : 0;
    model.reset(llama_model_init_from_user(talker_source.metadata(), component::set_tensor, &talker_source, mp));
    if (!model) {
        throw std::runtime_error("talker load failed");
    }
    auto cp            = llama_context_default_params();
    // Breeze: two speech lanes of up to speech_context_rows each, plus room for an aside
    cp.n_ctx           = is_breeze ? 2 * speech_context_rows + 1024 : 2048;
    cp.n_seq_max       = is_breeze ? 3 : 1;
    cp.kv_unified      = true;
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
    if (is_breeze) {
        const auto key = gguf_find_key(talker_source.metadata(), "qwen3.attention.key_length");
        const auto head_dim = key >= 0 ? gguf_get_val_u32(talker_source.metadata(), key) :
            llama_model_n_embd(model.get()) / llama_model_n_head(model.get());
        speech_row_bytes = size_t(4) * head_dim * llama_model_n_head_kv(model.get()) * llama_model_n_layer(model.get());
        if (!options.voice_codes.empty()) { throw std::runtime_error("Breeze encodes its reference WAV directly; omit --voice-codes"); }
        breeze = std::make_unique<breeze_mouth>(side_source, mouth_source, options, cancelled);
        reference_rms = breeze->reference_rms;
    } else {
        if (options.voice_text.empty() != options.voice_codes.empty()) {
            throw std::runtime_error("Qwen TTS voice text and codes must be supplied together");
        }
        if (options.side_scale != 0.0f) {
            ggml_context *                                      raw = nullptr;
            std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
                gguf_init_from_callback(component::callback, &side_source, 1024 * 1024, side_source.size(), { false, &raw }),
                gguf_free);
            std::unique_ptr<ggml_context, decltype(&ggml_free)> side_weights(raw, ggml_free);
            if (!metadata || !side_weights) {
                throw std::runtime_error("Side load failed");
            }
            side = std::make_unique<mtmd_side>(raw, options.use_gpu);
        }
    }
    if (!breeze || !options.voice.empty()) {
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
        if (breeze) {
            if (!options.voice_text.empty()) {
                const auto text = frankie_read_file(options.voice_text, 8192);
                reference.assign(text.begin(), text.end());
            } else if (transcribe) {
                reference = transcribe(std::vector<float>(samples, samples + n));
            } else {
                throw std::runtime_error("Breeze WAV needs a reference transcriber or --voice-text-file");
            }
            breeze->set_reference(mctx.get(), speaker.get(), reference);
        }
    }

    if (breeze) {
        mtmd_release_audio_encoder(mctx.get());
        speaker.reset();
    }

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

    std::vector<char> delivery_bytes;
    if (!options.delivery.empty()) {
        delivery_bytes = frankie_read_file(options.delivery, 64 * 1024 * 1024);
    } else if (mouth_source.has_asset("assets.delivery.gguf")) {
        delivery_bytes = mouth_source.asset("assets.delivery.gguf", 64 * 1024 * 1024);
    }
    if (!delivery_bytes.empty()) {
        if (!breeze || !breeze->supports_context() || !options.speech_context_words) {
            throw std::runtime_error("brain-led delivery needs the Breeze mouth with a speech context");
        }
        delivery = std::make_unique<frankie_delivery>(delivery_bytes);
    }
}

std::vector<float> mouth_session::speak(const brain_session::response & response, const audio_callback & on_audio, const brain_session::stage_callback & on_stage) {
    aside_cancelled = true;
    std::lock_guard<std::mutex> lock(execution);
    try { return speak_impl(response, on_audio, on_stage, nullptr); }
    catch (...) { clear_speech_context(); throw; }
}

void mouth_session::clear_speech_context() {
    speech_words = 0;
    segments.clear();
    speech_rows[0] = speech_rows[1] = 0;
    speech_next[0] = speech_next[1] = 0;
    auto * memory = llama_get_memory(ctx.get());
    if (breeze) {
        llama_memory_seq_rm(memory, 0, -1, -1);
        llama_memory_seq_rm(memory, 1, -1, -1);
    } else {
        llama_memory_clear(memory, true);
    }
}

void mouth_session::evict_oldest_phrase() {
    const auto & oldest = segments.front();
    for (int lane = 0; lane < 2; ++lane) {
        if (oldest.end[lane] > oldest.begin[lane]) {
            llama_memory_seq_rm(llama_get_memory(ctx.get()), lane, oldest.begin[lane], oldest.end[lane]);
            speech_rows[lane] -= oldest.end[lane] - oldest.begin[lane];
        }
    }
    speech_words -= oldest.words;
    segments.pop_front();
}

size_t mouth_session::max_speech_rows() const {
    return std::min(speech_context_rows, speech_context_bytes / (speech_row_bytes * 256) * 256);
}

void mouth_session::reset_speech_context() {
    std::lock_guard<std::mutex> lock(execution);
    clear_speech_context();
    held.clear();
    reply_start = false;
}

// Between replies: keep the newest whole phrases, at most speech_hold_words words, so the next
// reply goes on in the same voice. Zero starts every reply over from the reference.
void mouth_session::hold_speech() {
    std::lock_guard<std::mutex> lock(execution);
    reply_start = true;
    if (!options.speech_hold_words) {
        clear_speech_context();
        return;
    }
    while (!segments.empty() && speech_words > options.speech_hold_words) { evict_oldest_phrase(); }
}

// Breeze's first codebook: guided logits, repetition penalty over this phrase's codes (1.1 with
// brain-led delivery, as MTPLX), reserved codec ids masked, then top-k 50 at temperature 0.9
// (mlx_audio breeze_tts.generate).
llama_token mouth_session::sample_breeze(const std::vector<llama_token> & history) {
    int32_t n = 0;
    const float * raw = generator->get_logits(&n);
    if (!raw || n != 2052) { throw std::runtime_error("Breeze logits"); }
    std::vector<float> logits(raw, raw + n);
    const float penalty = delivery ? 1.1f : 1.0f;
    std::vector<bool> seen(n);
    for (auto token : history) {
        if (token >= 0 && token < n && !seen[token]) {
            seen[token] = true;
            logits[token] = logits[token] < 0 ? logits[token] * penalty : logits[token] / penalty;
        }
    }
    for (int token = 2048; token < 2051; ++token) { logits[token] = -INFINITY; }
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + 50, order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    std::vector<double> weights(50);
    for (int i = 0; i < 50; ++i) { weights[i] = std::exp(double(logits[order[i]] - logits[order[0]]) / 0.9); }
    std::discrete_distribution<int> pick(weights.begin(), weights.end());
    return order[pick(rng)];
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
    if (spoken.size() > 8192 || (!brain.hidden.empty() && brain.hidden.size() != brain.ends.size())) {
        throw std::runtime_error("response shape/bounds");
    }
    size_t lead = response.content_offset == std::string::npos ? brain.text.find(spoken) : response.content_offset + begin;
    if (lead == std::string::npos || brain.text.compare(lead, spoken.size(), spoken) != 0) {
        throw std::runtime_error("parsed content is not a contiguous generated span");
    }
    const frankie_normalized_text normalized(spoken);
    std::array<float, 2048> speaker_offset{};
    int emotion = style ? style->emotion : -1;
    // Brain-led delivery replaces the expression head: Breeze then hears learned rows, not an emotion prompt.
    const bool guided = delivery && !style;
    if (expression && style && style->emotion >= 0) {
        speaker_offset = expression->direction(style->emotion);
    } else if (expression && !style && !guided) {
        if (brain.hidden.empty()) { throw std::runtime_error("the expression head needs layer-17 brain states"); }
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
        const auto winner = std::max_element(result.probabilities.begin(), result.probabilities.end());
        if (*winner > 0.5f) { emotion = winner - result.probabilities.begin(); }
        if (on_stage) { on_stage("mouth_expression_done"); }
    }
    std::vector<float> target, plain;
    size_t words = 0, breeze_tokens = 0;
    float guidance = 1.0f;
    if (breeze) {
        const auto text = frankie_spoken_numbers(normalized.text);
        const frankie_text_offsets offsets(text);
        bool space = true;
        for (bool next : offsets.whitespace) { words += space && !next; space = next; }
        breeze_tokens = breeze->token_count(text);
        if (!guided) {
            target = breeze->conditioning(text, emotion);
        } else {
            // Paired guidance: the directed lane hears the learned rows, the other lane the bare text.
            plain = breeze->unconditioned(text);
            const size_t width = delivery->width();
            if (brain.width == width && !brain.final.empty() && brain.final.size() % width == 0) {
                const size_t count = brain.final.size() / width;
                std::vector<double> sum(width);
                for (size_t r = 0; r < count; ++r) {
                    for (size_t j = 0; j < width; ++j) { sum[j] += brain.final[r * width + j]; }
                }
                std::vector<float> mean(width);
                for (size_t j = 0; j < width; ++j) { mean[j] = float(sum[j] / count); }
                const auto row = delivery->read(mean, count);
                target = breeze->learned(plain, row.rows);
                guidance = std::max(1.001f, row.strength);
                std::string why = "none";
                if (options.speech_hold_words) {
                    // An extreme feeling outranks the held voice: drop it and start from the reference.
                    held.follow(speech_rows[0] ? speech_words : 0);
                    frankie_hold::reason reason{};
                    if (held.why(row.feelings, reply_start, reason)) {
                        clear_speech_context();
                        held.clear();
                        char buffer[160];
                        std::snprintf(buffer, sizeof(buffer), "%s:%s:%.2f:%.2f", reason.rule,
                                      delivery->names[reason.axis].c_str(), reason.reading, reason.held);
                        why = buffer;
                    }
                    reply_start = false;
                    held.add(words, row.feelings);
                }
                std::cerr << "Breeze delivery: rows=" << count << " strength=" << row.strength << " weights=";
                for (const auto & [index, weight] : row.weights) { std::cerr << delivery->bank[index] << ":" << weight << ","; }
                std::cerr << " feelings=";
                for (double z : row.feelings) { std::cerr << z << ","; }
                std::cerr << " held=" << why << "\n";
            } else {
                target = breeze->conditioning(text, -1);
                guidance = 1.001f;
            }
        }
    }
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
        const size_t chosen = aligned[i];
        if (chosen >= brain.hidden.size()) {
            throw std::runtime_error("alignment bounds");
        }
        auto residual = side->process(brain.hidden[chosen]);
        for (size_t j = 0; j < 2048; ++j) {
            target.push_back(options.side_scale * residual[j]);
        }
    }
    }
    // Asides (backchannels, warmup) speak from the reference on their own sequence and never touch held speech.
    auto * memory = llama_get_memory(ctx.get());
    const llama_seq_id seq = breeze && style ? 2 : 0;
    const bool windowed = breeze && !style && options.speech_context_words && breeze->supports_context() && speech_row_bytes;
    const int max_frames = style ? style->frames : breeze ? int(std::min<size_t>(384, std::max<size_t>(75, breeze_tokens * 12))) : 512;
    const int lanes = guided ? 2 : 1;
    auto & gen = *generator;
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
            for (size_t i = emitted; i < size_t(samples); ++i) { rendered.push_back(pcm[i] * gain); }
            if (on_audio) { on_audio(rendered.data() + emitted, size_t(samples) - emitted); }
            emitted = samples;
        }
    };
    bool stop = false, continuing = false;
    int frames = 0;
    llama_pos start[2] = {0, 0};
    for (int attempt = 0;; ++attempt) {
        continuing = false;
        if (windowed) {
            // Slide: evict whole oldest phrases for this prompt and its reserved frames.
            const size_t required = target.size() / 2048 + 1 + max_frames, limit = max_speech_rows();
            while (!segments.empty() && (speech_words + words > options.speech_context_words || speech_rows[0] + required > limit)) {
                evict_oldest_phrase();
            }
            if (speech_rows[0] + required > limit) { clear_speech_context(); }
            continuing = speech_rows[0] && speech_rows[0] + required <= speech_context_rows;
        }
        if (seq == 2) {
            llama_memory_seq_rm(memory, 2, -1, -1);
        } else if (!continuing) {
            clear_speech_context();
        }
        std::vector<float> prompt = target, unconditional = plain;
        if (breeze) {
            breeze->prepend_reference(prompt, continuing);
            if (guided) { breeze->prepend_reference(unconditional, continuing); }
            const size_t cached = continuing ? speech_rows[0] : 0;
            if (windowed) { std::cerr << "Breeze context: cached=" << cached << " new=" << prompt.size() / 2048 << "\n"; }
        }
        for (int lane = 0; lane < lanes; ++lane) { start[lane] = continuing ? speech_next[lane] : 0; }
        if (on_stage) { on_stage(continuing ? "mouth_context_reused" : "mouth_context_reset"); }
        mtmd_helper_gen_audio_inp input{};
        input.seq_id        = seq;
        input.n_past        = start[0];
        if (breeze) {
            input.prompt_embd = prompt.data();
            input.n_prompt_embd = prompt.size() / 2048;
            if (guided) {
                input.uncond_prompt_embd   = unconditional.data();
                input.n_uncond_prompt_embd = unconditional.size() / 2048;
                input.uncond_seq_id        = 1;
                input.uncond_n_past        = start[1];
                input.cfg_scale            = guidance;
            }
        } else {
        input.prompt        = normalized.text.c_str();
        input.prompt_len    = normalized.text.size();
        input.speaker_ref   = speaker.get();
        input.lang          = "en";
        input.ref_text      = reference.empty() ? nullptr : reference.c_str();
        input.ref_text_len  = reference.size();
        input.ref_codes     = codes.empty() ? nullptr : codes.data();
        input.n_ref_frames  = codes.size() / 16;
        input.prime_frames  = input.n_ref_frames;
        input.target_offset   = target.empty() ? nullptr : target.data();
        input.n_target_offset = target.size() / 2048;
        input.speaker_offset = expression ? speaker_offset.data() : nullptr;
        input.n_speaker_offset = expression ? speaker_offset.size() : 0;
        }
        input.code_temperature = breeze ? 0.9f : 0.6f;
        input.top_k         = 50;
        input.top_p         = 1.0f;
        input.seed          = 42;
        input.out_type      = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;
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
        std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler(nullptr, common_sampler_free);
        if (!breeze) {
            common_params_sampling sampling;
            sampling.temp = 0.6f;
            sampling.seed = 42;
            sampling.top_k = 50;
            sampling.top_p = 1.0f;
            sampling.min_p = 0.0f;
            sampling.penalty_repeat = codes.empty() ? 1.05f : 1.5f;
            sampling.penalty_last_n = 4096;
            sampler.reset(common_sampler_init(model.get(), sampling));
            if (!sampler) {
                throw std::runtime_error("mouth sampler failed");
            }
        }
        std::vector<llama_token> history;
        stop = false;
        frames = 0;
        const float * hidden = llama_get_embeddings_ith(ctx.get(), -1);
        while (frames < max_frames && !stop) {
            if (is_cancelled()) {
                throw std::runtime_error("cancelled");
            }
            llama_token token;
            if (breeze) {
                token = sample_breeze(history);
                history.push_back(token);
            } else {
                token = common_sampler_sample(sampler.get(), ctx.get(), -1);
                common_sampler_accept(sampler.get(), token, true);
            }
            const float * next = nullptr;
            if (gen.step_gen(token, hidden, &next, &stop)) {
                throw std::runtime_error("mouth generation failed");
            }
            emit();
            if (!next) {
                break;
            }
            ++frames;
            hidden = next;
        }
        // An immediate EOS is not a spoken phrase: retry once from the reference.
        if (!breeze || frames || !stop) { break; }
        if (attempt) { throw std::runtime_error("Breeze produced no speech after a retry"); }
        if (seq != 2) { clear_speech_context(); }
    }
    if (is_cancelled()) {
        throw std::runtime_error("cancelled");
    }
    emit();
    int32_t      rate    = 0;
    const char * data    = nullptr;
    size_t       bytes   = 0;
    int64_t      samples = 0;
    // A Breeze phrase may stop at its frame budget; it is then not kept as context.
    if ((!stop && !breeze && !(style && style->aside)) || gen.get_output(&rate, &data, &bytes, &samples) || rate != 24000 || samples <= 0 ||
        bytes > 4 * 1024 * 1024 || bytes != size_t(samples) * sizeof(float)) {
        throw std::runtime_error("mouth output bounds or unfinished");
    }
    if (rendered.size() != size_t(samples)) { throw std::runtime_error("incomplete rendered PCM"); }
    if (seq == 2) {
        llama_memory_seq_rm(memory, 2, -1, -1);
    } else if (windowed) {
        // Keep the phrase when it finished within budget and every lane still fits.
        llama_pos end[2] = {0, 0};
        size_t rows[2] = {0, 0};
        bool safe = stop && frames > 0 && words <= options.speech_context_words;
        for (int lane = 0; lane < lanes; ++lane) {
            end[lane] = llama_memory_seq_pos_max(memory, lane) + 1;
            if (!continuing) { start[lane] = breeze->prefix_rows(); }
            rows[lane] = (continuing ? speech_rows[lane] : breeze->prefix_rows()) + size_t(end[lane] - start[lane]);
            safe = safe && end[lane] > start[lane] && rows[lane] <= speech_context_rows &&
                rows[lane] * speech_row_bytes <= speech_context_bytes;
        }
        if (safe) {
            speech_words += words;
            segments.push_back({words, {start[0], start[1]}, {end[0], end[1]}});
            for (int lane = 0; lane < lanes; ++lane) {
                speech_rows[lane] = rows[lane];
                speech_next[lane] = end[lane];
            }
        } else {
            clear_speech_context();
        }
        std::cerr << "Breeze context: rows=" << speech_rows[0] << " words=" << speech_words
                  << " bytes=" << speech_rows[0] * speech_row_bytes << " complete=" << (stop && frames > 0) << "\n";
    } else {
        clear_speech_context();
    }
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
