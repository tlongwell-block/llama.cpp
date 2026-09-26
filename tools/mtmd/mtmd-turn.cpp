// Streaming MaAI CPC/ALiBi architecture (https://github.com/MaAI-Kyoto/MaAI).
// Copyright (c) 2025 MaAI Development Team. MIT license: licenses/LICENSE-maai.
#include "mtmd-turn.h"
#include "mtmd-backend.h"
#include "mtmd-graph.h"
#include "clip-impl.h"
#include "clip-model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {
constexpr int dim = 256, heads = 4;
}

struct mtmd_turn::impl {
    mode kind;
    int frames = 1, kernel = 0, retained = 199;
    int valid = 0;
    mtmd_backend backend;
    ggml_context_ptr work;
    ggml_gallocr_ptr allocator;
    ggml_cgraph * graph = nullptr;
    std::array<ggml_tensor *, 2> input{}, features{};
    std::array<mtmd_lstm_state, 2> initial{}, final{};
    std::array<std::array<float, 320>, 2> previous{};
    ggml_tensor * combined = nullptr;
    ggml_tensor * probabilities = nullptr;
    ggml_tensor * backchannel = nullptr;
    ggml_tensor * mask = nullptr;
    struct cache_pair { ggml_tensor * in; ggml_tensor * out; };
    std::vector<cache_pair> caches;

    ggml_tensor * w(const std::string & name, int64_t a, int64_t b = 1, int64_t c = 1) {
        const auto key = "turn." + name;
        auto * t = ggml_get_tensor(backend.context(), key.c_str());
        if (!t || t->type != GGML_TYPE_F32 || t->ne[0] != a || t->ne[1] != b || t->ne[2] != c || t->ne[3] != 1) {
            throw std::runtime_error("invalid turn tensor: " + key);
        }
        return t;
    }

    explicit impl(ggml_context * weights, mode kind, bool gpu) : kind(kind), backend(weights, gpu, "turn.") {
        if (kind == mode::detection) {
            retained = 249;
        } else {
            auto * downsample = ggml_get_tensor(backend.context(), "turn.encoder.downsample.1.weight");
            if (!downsample || (downsample->ne[0] != 5 && downsample->ne[0] != 10)) {
                throw std::runtime_error("unsupported turn downsampling kernel");
            }
            kernel = downsample->ne[0];
            frames = (10 - kernel) / 2 + 1;
        }
        work.reset(ggml_init({16 * 1024 * 1024, nullptr, true}));
        if (!work) { throw std::runtime_error("cannot allocate turn graph"); }
        auto * ctx = work.get();
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, retained + frames, frames);
        for (int channel = 0; channel < 2; ++channel) {
            if (kind == mode::detection) {
                input[channel] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 1);
                features[channel] = ggml_relu(ctx, ggml_add(ctx, linear(input[channel], "decrease_dimension", 512, dim),
                                                           w("decrease_dimension.bias", dim)));
            } else {
                input[channel] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1920, 1);
                initial[channel] = {ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim),
                                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim)};
                features[channel] = encoder(input[channel], channel);
            }
        }
        auto * a = layer(features[0], nullptr, "ar_channel.layers.0");
        auto * b = layer(features[1], nullptr, "ar_channel.layers.0");
        for (int i = 0; i < 3; ++i) {
            const auto prefix = "ar.layers." + std::to_string(i);
            auto * next_a = layer(a, b, prefix);
            auto * next_b = layer(b, a, prefix);
            a = next_a; b = next_b;
        }
        if (kind == mode::detection) {
            combined = a;
            backchannel = ggml_sigmoid(ctx, ggml_add(ctx, linear(a, "bc_classifier", dim, 1), w("bc_classifier.bias", 1)));
        } else {
            combined = ggml_add(ctx,
                ggml_gelu_erf(ctx, norm(linear(a, "ar.combinator.h0_a", dim, dim), "ar.combinator.ln")),
                ggml_gelu_erf(ctx, norm(linear(b, "ar.combinator.h0_b", dim, dim), "ar.combinator.ln")));
            if (kind != mode::backchannel) {
                auto * logits = ggml_add(ctx, linear(combined, "vap_head", dim, 256), w("vap_head.bias", 256));
                // Expected occupancy in the first two future bins, normalized across speakers.
                auto * now = ggml_mul_mat(ctx, w("now.weight", 256, 2), ggml_soft_max(ctx, logits));
                probabilities = ggml_div(ctx, now, ggml_scale_bias(ctx, ggml_sum_rows(ctx, now), 1.0f, 1e-5f));
            }
            if (kind != mode::vap) {
                backchannel = ggml_sigmoid(ctx, ggml_add(ctx, linear(combined, "bc_head", dim, 1), w("bc_head.bias", 1)));
            }
        }
        ggml_set_input(mask);
        for (int channel = 0; channel < 2; ++channel) {
            ggml_set_input(input[channel]);
            for (auto * tensor : {initial[channel].h, initial[channel].c}) {
                if (tensor) { ggml_set_input(tensor); ggml_set_output(tensor); }
            }
            for (auto * tensor : {final[channel].h, final[channel].c, features[channel]}) {
                if (tensor) { ggml_set_output(tensor); }
            }
        }
        ggml_set_output(combined);
        graph = ggml_new_graph_custom(ctx, 4096, false);
        for (auto * output : {probabilities, backchannel}) {
            if (output) { ggml_set_output(output); ggml_build_forward_expand(graph, output); }
        }
        for (const auto & cache : caches) { ggml_build_forward_expand(graph, cache.out); }
        // Recurrent turn features need full precision on every backend.
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_MUL_MAT) { ggml_prec_set_acc(node, GGML_PREC_F32); }
        }
        allocator = backend.allocate(graph);
    }

    void step() {
        const int length = retained + frames;
        std::vector<float> values(length * frames, -INFINITY);
        for (int q = 0; q < frames; ++q) {
            for (int k = std::max(retained - valid, q); k <= retained + q; ++k) { values[q * length + k] = 0.0f; }
        }
        ggml_backend_tensor_set(mask, values.data(), 0, values.size() * sizeof(float));
        backend.compute(graph);
        for (auto & kv : caches) { ggml_backend_tensor_copy(kv.out, kv.in); }
        valid = std::min(retained, valid + frames);
    }

    ggml_tensor * linear(ggml_tensor * x, const std::string & prefix, int in, int out) {
        return ggml_mul_mat(work.get(), w(prefix + ".weight", in, out), x);
    }

    ggml_tensor * norm(ggml_tensor * x, const std::string & prefix) {
        auto * ctx = work.get();
        return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, 1e-5f), w(prefix + ".weight", dim)),
                            w(prefix + ".bias", dim));
    }

    ggml_tensor * encoder(ggml_tensor * x, int channel) {
        auto * ctx = work.get();
        const int kernels[] = {10, 8, 4, 4, 4};
        const int strides[] = {5, 4, 2, 2, 2};
        const int pads[] = {3, 2, 1, 1, 1};
        for (int i = 0; i < 5; ++i) {
            const auto p = "encoder.encoder.gEncoder.";
            const auto conv = p + std::string("conv") + std::to_string(i);
            const auto bn = p + std::string("batchNorm") + std::to_string(i);
            x = mtmd_conv_1d_f32(ctx, x, w(conv + ".weight", kernels[i], i ? dim : 1, dim), strides[i], pads[i]);
            // ChannelNorm uses sample variance (N-1); ggml_norm uses population variance.
            x = ggml_cont(ctx, ggml_transpose(ctx, x));
            x = ggml_add(ctx, x, w(conv + ".bias", dim));
            x = ggml_scale(ctx, ggml_norm(ctx, x, 1e-5f * 255.0f / 256.0f), std::sqrt(255.0f / 256.0f));
            x = ggml_relu(ctx, ggml_add(ctx, ggml_mul(ctx, x, w(bn + ".weight", dim)), w(bn + ".bias", dim)));
            x = ggml_cont(ctx, ggml_transpose(ctx, x));
        }
        if (x->ne[0] != 12 || x->ne[1] != dim) { throw std::runtime_error("invalid CPC frame geometry"); }
        // Discard the first and last CPC frames, exactly as the streaming encoder does.
        x = ggml_cont(ctx, ggml_transpose(ctx, ggml_view_2d(ctx, x, 10, dim, x->nb[1], sizeof(float))));
        const std::string ar = "encoder.encoder.gAR.baseNet.";
        auto * wx = ggml_mul_mat(ctx, w(ar + "weight_ih_l0", dim, 4 * dim), x);
        wx = ggml_add(ctx, wx, ggml_add(ctx, w(ar + "bias_ih_l0", 4 * dim), w(ar + "bias_hh_l0", 4 * dim)));
        auto state = initial[channel];
        ggml_tensor * states = nullptr;
        for (int t = 0; t < 10; ++t) {
            state = mtmd_lstm_step(ctx, ggml_view_1d(ctx, wx, 4 * dim, t * wx->nb[1]),
                                  w(ar + "weight_hh_l0", dim, 4 * dim), state);
            auto * row = ggml_reshape_2d(ctx, state.h, dim, 1);
            states = states ? ggml_concat(ctx, states, row, 1) : row;
        }
        final[channel] = state;
        x = mtmd_conv_1d_f32(ctx, ggml_cont(ctx, ggml_transpose(ctx, states)),
                            w("encoder.downsample.1.weight", kernel, dim, dim), 2, 0);
        x = ggml_cont(ctx, ggml_transpose(ctx, x));
        x = ggml_add(ctx, x, w("encoder.downsample.1.bias", dim));
        return ggml_gelu_erf(ctx, norm(x, "encoder.downsample.2.ln"));
    }

    ggml_tensor * remember(ggml_tensor * current) {
        auto * ctx = work.get();
        auto * past = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, retained);
        ggml_set_input(past);
        ggml_set_output(past);
        auto * joined = ggml_concat(ctx, past, current, 1);
        ggml_set_output(joined);
        auto * kept = ggml_view_2d(ctx, joined, dim, retained, joined->nb[1], frames * joined->nb[1]);
        ggml_set_output(kept);
        caches.push_back({past, kept});
        return joined;
    }

    ggml_tensor * attention(ggml_tensor * query, ggml_tensor * source, const std::string & p) {
        auto * ctx = work.get();
        auto * q = linear(query, p + ".query", dim, dim);
        auto * k = remember(linear(source, p + ".key", dim, dim));
        auto * v = remember(linear(source, p + ".value", dim, dim));
        const int length = retained + frames;
        q = ggml_permute(ctx, ggml_reshape_3d(ctx, q, dim / heads, heads, frames), 0, 2, 1, 3);
        k = ggml_permute(ctx, ggml_reshape_3d(ctx, k, dim / heads, heads, length), 0, 2, 1, 3);
        v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, dim / heads, heads, length), 1, 2, 0, 3));
        auto * scores = ggml_scale(ctx, ggml_mul_mat(ctx, k, q), 1.0f / 16.0f);
        auto * positions = ggml_arange(ctx, 0, length, 1);
        auto * slopes = ggml_reshape_3d(ctx, w(p + ".m", heads), 1, 1, heads);
        auto * bias = ggml_add(ctx, ggml_mul(ctx, ggml_repeat(ctx, ggml_reshape_3d(ctx, positions, length, 1, 1), scores), slopes), mask);
        scores = ggml_soft_max(ctx, ggml_add(ctx, scores, bias));
        auto * out = ggml_mul_mat(ctx, v, scores);
        out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
        return linear(ggml_reshape_2d(ctx, out, dim, frames), p + ".proj", dim, dim);
    }

    ggml_tensor * layer(ggml_tensor * x, ggml_tensor * other, const std::string & p) {
        auto * ctx = work.get();
        auto * z = norm(x, p + ".ln_self_attn");
        x = ggml_add(ctx, x, attention(z, z, p + ".mha"));
        if (other) { x = ggml_add(ctx, x, attention(norm(x, p + ".ln_src_attn"), other, p + ".mha_cross")); }
        z = linear(norm(x, p + ".ln_ffnetwork"), p + ".ffnetwork.0", dim, 3 * dim);
        return ggml_add(ctx, x, linear(ggml_gelu_erf(ctx, z), p + ".ffnetwork.3", 3 * dim, dim));
    }
};

mtmd_turn::mtmd_turn(ggml_context * weights, mode kind, bool gpu) : data(std::make_unique<impl>(weights, kind, gpu)) { reset(); }
mtmd_turn::~mtmd_turn() = default;

void mtmd_turn::reset() {
    data->valid = 0;
    for (auto & p : data->previous) { p.fill(0); }
    for (auto & state : data->initial) {
        if (state.h) { ggml_backend_tensor_memset(state.h, 0, 0, ggml_nbytes(state.h)); }
        if (state.c) { ggml_backend_tensor_memset(state.c, 0, 0, ggml_nbytes(state.c)); }
    }
    for (auto & kv : data->caches) { ggml_backend_tensor_memset(kv.in, 0, 0, ggml_nbytes(kv.in)); }
}

mtmd_turn::result mtmd_turn::process(const std::array<float, 1600> & user, const std::array<float, 1600> & system) {
    const std::array<const std::array<float, 1600> *, 2> audio = data->kind == mode::vap ?
        std::array{&user, &system} : std::array{&system, &user};
    for (int ch = 0; ch < 2; ++ch) {
        const auto & x = *audio[ch];
        if (!std::all_of(x.begin(), x.end(), [](float v) { return std::isfinite(v); })) {
            throw std::runtime_error("turn audio contains non-finite samples");
        }
        std::array<float, 1920> padded;
        std::copy(data->previous[ch].begin(), data->previous[ch].end(), padded.begin());
        std::copy(x.begin(), x.end(), padded.begin() + 320);
        ggml_backend_tensor_set(data->input[ch], padded.data(), 0, sizeof(padded));
    }
    data->step();
    result r;
    if (data->probabilities) {
        ggml_backend_tensor_get(data->probabilities, r.next_speaker.data(), (data->frames - 1) * 2 * sizeof(float), 2 * sizeof(float));
        // The BC checkpoint encodes (system, user); public results use (user, system).
        if (data->kind == mode::duplex) { std::swap(r.next_speaker[0], r.next_speaker[1]); }
    }
    if (data->backchannel) {
        ggml_backend_tensor_get(data->backchannel, &r.backchannel, (data->frames - 1) * sizeof(float), sizeof(float));
    }
    for (float out : {r.next_speaker[0], r.next_speaker[1], r.backchannel}) {
        if (!std::isfinite(out) || out < 0 || out > 1.00001f) { throw std::runtime_error("invalid turn probability"); }
    }
    for (int ch = 0; ch < 2; ++ch) {
        std::copy(audio[ch]->end() - 320, audio[ch]->end(), data->previous[ch].begin());
        ggml_backend_tensor_copy(data->final[ch].h, data->initial[ch].h);
        ggml_backend_tensor_copy(data->final[ch].c, data->initial[ch].c);
    }
    return r;
}

std::vector<float> mtmd_turn::encoded() const {
    const size_t count = dim * data->frames;
    std::vector<float> result(2 * count);
    for (int ch = 0; ch < 2; ++ch) { ggml_backend_tensor_get(data->features[ch], result.data() + ch * count, 0, count * sizeof(float)); }
    return result;
}

std::vector<float> mtmd_turn::hidden() const {
    std::vector<float> result(dim * data->frames);
    ggml_backend_tensor_get(data->combined, result.data(), 0, result.size() * sizeof(float));
    return result;
}

float mtmd_turn::detect(const std::array<float, 512> & user, const std::array<float, 512> & system) {
    if (data->kind != mode::detection) { throw std::runtime_error("not a backchannel detector"); }
    for (int ch = 0; ch < 2; ++ch) {
        const auto & input = ch ? system : user;
        if (!std::all_of(input.begin(), input.end(), [](float x) { return std::isfinite(x); })) {
            throw std::runtime_error("non-finite detector features");
        }
        ggml_backend_tensor_set(data->input[ch], input.data(), 0, sizeof(input));
    }
    data->step();
    float result;
    ggml_backend_tensor_get(data->backchannel, &result, 0, sizeof(result));
    if (!std::isfinite(result) || result < 0 || result > 1) { throw std::runtime_error("invalid backchannel probability"); }
    return result;
}

struct mtmd_backchannel::impl {
    std::unique_ptr<clip_ctx, decltype(&clip_free)> encoder{nullptr, clip_free};
    mtmd_turn head;
    int threads;
    uint64_t steps = 0;
    std::array<float, 2> previous{};
    std::array<std::vector<uint8_t>, 2> cache;
    std::array<std::array<float, 512>, 2> features{};

    impl(ggml_context * weights, const char * path, const mtmd_context_params & params)
        : head(weights, mtmd_turn::mode::detection, params.use_gpu), threads(params.n_threads) {
        clip_context_params options{};
        options.use_gpu = params.use_gpu;
        options.device = params.device;
        options.cb_eval = params.cb_eval;
        options.cb_eval_user_data = params.cb_eval_user_data;
        options.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
        options.model_reader = params.model_reader;
        options.model_reader_user_data = params.model_reader_user_data;
        options.model_reader_size = params.model_reader_size;
        auto loaded = clip_init(path, options);
        encoder.reset(loaded.ctx_a);
        if (loaded.ctx_v) { clip_free(loaded.ctx_v); }
        if (loaded.ctx_gen_a) { clip_free(loaded.ctx_gen_a); }
        if (!encoder || clip_get_hparams(encoder.get())->gen_model_variant != "mimi") {
            throw std::runtime_error("BC-Det requires the continuous Mimi encoder");
        }
    }
};

mtmd_backchannel::mtmd_backchannel(ggml_context * weights, const char * path, const mtmd_context_params & params)
    : data(std::make_unique<impl>(weights, path, params)) {}
mtmd_backchannel::~mtmd_backchannel() = default;

void mtmd_backchannel::reset() {
    data->steps = 0;
    data->previous.fill(0);
    for (auto & cache : data->cache) { cache.clear(); }
    data->head.reset();
}

float mtmd_backchannel::process(const std::array<float, 1280> & user, const std::array<float, 1280> & system) {
    for (int ch = 0; ch < 2; ++ch) {
        const auto & input = ch ? system : user;
        if (!std::all_of(input.begin(), input.end(), [](float x) { return std::isfinite(x); })) {
            throw std::runtime_error("non-finite backchannel audio");
        }
        // Match MaAI's causal linear 16-to-24 kHz resampler, including its one
        // startup zero. No lookahead or repeated overlap enters the encoder.
        std::vector<float> pcm(1920);
        const int lead = data->steps == 0 ? 1 : 0;
        const float start = data->steps == 0 ? 0 : float(double(data->steps * 1280) - 2.0 / 3.0);
        for (int i = lead; i < 1920; ++i) {
            const volatile float offset = float(i - lead) * (2.0f / 3.0f);
            const float position = offset + start;
            const int64_t floor = int64_t(std::floor(position));
            const float frac = position - float(floor);
            const int index = int(floor - int64_t(data->steps * 1280));
            const float a = index < 0 ? data->previous[ch] : input[std::min(index, 1279)];
            const float b = input[std::clamp(index + 1, 0, 1279)];
            pcm[i] = (1.0f - frac) * a + frac * b;
        }
        data->previous[ch] = input.back();
        clip_image_f32_batch batch;
        batch.is_audio = true;
        batch.entries.emplace_back();
        batch.entries[0].set_size({1920, 1}, false, true);
        batch.entries[0].cpy_buf(pcm);
        std::vector<float> output(512);
        std::vector<uint8_t> next;
        clip_encode_params params;
        params.imgs = &batch;
        params.n_threads = data->threads;
        params.out_embd = &output;
        params.state_in = &data->cache[ch];
        params.state_out = &next;
        if (!clip_encode(data->encoder.get(), &params) || output.size() != 512) {
            throw std::runtime_error("streaming Mimi encode failed");
        }
        data->cache[ch] = std::move(next);
        std::copy(output.begin(), output.end(), data->features[ch].begin());
    }
    if (data->steps++ == 0) { return -1; }
    return data->head.detect(data->features[0], data->features[1]);
}

std::vector<float> mtmd_backchannel::encoded() const {
    std::vector<float> out(data->features[0].begin(), data->features[0].end());
    out.insert(out.end(), data->features[1].begin(), data->features[1].end());
    return out;
}
