#pragma once
#include "brain-session.h"
#include "mtmd-backend.h"

// Breeze's native text encoder and projection, using the shared Gemma graph.
class breeze_mouth {
    llama_model_ptr model;
    llama_context_ptr ctx;
    std::unique_ptr<mtmd_backend> projection;
    std::vector<float> voice;
    std::vector<float> voice_eos;

    std::vector<float> project(const float * rows, size_t count) {
        ggml_context_ptr work(ggml_init({1024 * 1024, nullptr, true}));
        if (!work) { throw std::runtime_error("Breeze projection graph allocation"); }
        auto * input = ggml_new_tensor_2d(work.get(), GGML_TYPE_F32, 1152, count);
        ggml_set_input(input);
        auto * output = ggml_mul_mat(work.get(), ggml_get_tensor(projection->context(), "breeze.projection"), input);
        auto * graph = ggml_new_graph(work.get());
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        auto allocation = projection->allocate(graph);
        ggml_backend_tensor_set(input, rows, 0, ggml_nbytes(input));
        projection->compute(graph);
        std::vector<float> result(2048 * count);
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
        if (!std::all_of(result.begin(), result.end(), [](float x) { return std::isfinite(x); })) {
            throw std::runtime_error("Breeze projection produced non-finite values");
        }
        return result;
    }

    std::vector<float> encode_text(const std::string & prompt) {
        std::vector<llama_token> ids(256);
        const int count = llama_tokenize(llama_model_get_vocab(model.get()), prompt.data(), prompt.size(),
                                         ids.data(), ids.size(), true, true);
        if (count <= 0 || count > 256) { throw std::runtime_error("Breeze text encoder token bounds"); }
        auto batch = llama_batch_init(count, 0, 1);
        for (int i = 0; i < count; ++i) {
            batch.token[i] = ids[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = true;
        }
        batch.n_tokens = count;
        const int rc = llama_encode(ctx.get(), batch);
        llama_batch_free(batch);
        if (rc) { throw std::runtime_error("Breeze text encoding failed"); }
        const auto * rows = llama_get_embeddings(ctx.get());
        if (!rows) { throw std::runtime_error("Breeze text encoder produced no embeddings"); }
        return project(rows, count);
    }

  public:
    float reference_rms = 0.0f;

    breeze_mouth(component & encoder, component & mouth, const frankie_options & options, std::atomic<bool> & cancelled) {
        auto bytes = mouth.asset("assets.breeze.gguf");
        ggml_context * raw = nullptr;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_from_buffer(bytes.data(), bytes.size(), {false, &raw}), gguf_free);
        ggml_context_ptr weights(raw);
        if (!metadata || !weights) { throw std::runtime_error("Breeze conditioning asset"); }
        auto tensor = [&](const char * name, int64_t width, int64_t rows) {
            auto * t = ggml_get_tensor(raw, name);
            if (!t || t->type != GGML_TYPE_F32 || t->ne[0] != width ||
                (rows > 0 && t->ne[1] != rows) || t->ne[1] < 1 || t->ne[1] > 2048 || t->ne[2] != 1 || t->ne[3] != 1) {
                throw std::runtime_error(std::string("Breeze conditioning shape: ") + name);
            }
            const auto * v = static_cast<const float *>(t->data);
            if (!v || !std::all_of(v, v + ggml_nelements(t), [](float x) { return std::isfinite(x); })) {
                throw std::runtime_error("Breeze non-finite conditioning");
            }
            return t;
        };
        tensor("breeze.projection", 1152, 2048);
        auto * prefix = tensor("voice.prefix", 2048, 0);
        const auto * values = static_cast<const float *>(prefix->data);
        voice.assign(values, values + ggml_nelements(prefix));
        if (!options.voice.empty()) {
            if (!ggml_get_tensor(raw, "voice.eos")) {
                throw std::runtime_error("Breeze package lacks WAV conditioning assets; rebuild it with convert-breeze.py");
            }
            const auto * eos = static_cast<const float *>(tensor("voice.eos", 2048, 1)->data);
            voice_eos.assign(eos, eos + 2048);
        }
        reference_rms = *static_cast<const float *>(tensor("voice.rms", 1, 1)->data);
        if (reference_rms <= 0.0f || reference_rms > 1.0f) { throw std::runtime_error("Breeze voice RMS"); }
        projection = std::make_unique<mtmd_backend>(raw, options.use_gpu, "breeze.");
        const bool encoder_gpu = options.use_gpu && options.text_encoder_gpu;
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = encoder_gpu ? 99 : 0;
        model.reset(llama_model_init_from_user(encoder.metadata(), component::set_tensor, &encoder, mp));
        if (!model || llama_model_n_embd(model.get()) != 1152) { throw std::runtime_error("Breeze text encoder load"); }
        auto cp = llama_context_default_params();
        cp.n_ctx = cp.n_batch = cp.n_ubatch = 256;
        cp.n_threads = cp.n_threads_batch = options.threads;
        cp.embeddings = true;
        cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
        cp.offload_kqv = cp.op_offload = encoder_gpu;
        cp.abort_callback = [](void * p) { return static_cast<std::atomic<bool> *>(p)->load(); };
        cp.abort_callback_data = &cancelled;
        ctx.reset(llama_init_from_model(model.get(), cp));
        if (!ctx) { throw std::runtime_error("Breeze text encoder context"); }
    }

    void set_reference(mtmd_context * encoder, const mtmd_bitmap * pcm, const std::string & transcript) {
        if (transcript.find_first_not_of(" \t\r\n") == std::string::npos || transcript.size() > 8192 || voice_eos.size() != 2048) {
            throw std::runtime_error("Breeze reference transcript or EOS is missing");
        }
        std::vector<float> audio;
        if (!mtmd_helper_encode_audio(encoder, pcm, 2048, audio)) {
            throw std::runtime_error("Breeze reference audio encoding failed");
        }
        auto prefix = encode_text("[S0]" + transcript);
        prefix.insert(prefix.end(), audio.begin(), audio.end());
        prefix.insert(prefix.end(), voice_eos.begin(), voice_eos.end());
        voice = std::move(prefix);
    }

    std::vector<float> conditioning(const std::string & text, int emotion) {
        if (text.empty() || text.size() > 8192 || emotion < -1 || emotion > 3) {
            throw std::runtime_error("Breeze spoken text bounds");
        }
        std::string instruction = "Speak clearly and naturally";
        if (emotion >= 0 && emotion < 3) {
            const char * names[] = {"angry", "happy", "sad"};
            instruction += std::string(" with a ") + names[emotion] + " tone";
        }
        const auto prompt = "[S0]<ins_bos>" + instruction + ".<ins_eos>" + text;
        auto target = encode_text(prompt);
        std::vector<float> result;
        result.reserve(voice.size() + target.size());
        result.insert(result.end(), voice.begin(), voice.end());
        result.insert(result.end(), target.begin(), target.end());
        return result;
    }
};
