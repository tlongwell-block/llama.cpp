#include "mtmd-ear.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

struct mtmd_ear::impl {
    ggml_context * weights;

    ggml_tensor * weight(const std::string & name, int64_t a, int64_t b = 1) {
        auto * t = ggml_get_tensor(weights, name.c_str());
        if (!t || !t->data || t->type != GGML_TYPE_F32 || t->ne[0] != a || t->ne[1] != b || t->ne[2] != 1 || t->ne[3] != 1) {
            throw std::runtime_error("invalid ear tensor: " + name);
        }
        return t;
    }

    explicit impl(ggml_context * ctx) : weights(ctx) {
        if (!ctx) { throw std::runtime_error("missing ear weights"); }
        weight("ear.ctc.weight", 512, 1025); weight("ear.ctc.bias", 1025);
        weight("ear.proj.weight", 1025, 5120);
        weight("ear_tone.probe.weight", 1024, 6); weight("ear_tone.probe.bias", 6);
        weight("ear_tone.word_emb", 5120, 6);
        for (const std::string prefix : {"ear", "ear_tone"}) {
            const int width = prefix == "ear" ? 512 : 1024;
            weight(prefix + ".ln.weight", width); weight(prefix + ".ln.bias", width);
            weight(prefix + ".l1.weight", width, 1024); weight(prefix + ".l1.bias", 1024);
            weight(prefix + ".l2.weight", 1024, 5120); weight(prefix + ".l2.bias", 5120);
            weight(prefix + ".gate", 1);
        }
    }

    ggml_tensor * linear(ggml_context * ctx, const std::string & name, ggml_tensor * x, int out) {
        return ggml_add(ctx, ggml_mul_mat(ctx, weight(name + ".weight", x->ne[0], out), x), weight(name + ".bias", out));
    }

    ggml_tensor * residual(ggml_context * ctx, const std::string & prefix, ggml_tensor * x) {
        auto * norm = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, 1e-5f), weight(prefix + ".ln.weight", x->ne[0])), weight(prefix + ".ln.bias", x->ne[0]));
        auto * up = linear(ctx, prefix + ".l1", norm, 1024);
        return ggml_mul(ctx, linear(ctx, prefix + ".l2", ggml_gelu_erf(ctx, up), 5120), weight(prefix + ".gate", 1));
    }
};

mtmd_ear::mtmd_ear(ggml_context * weights) : data(std::make_unique<impl>(weights)) {}
mtmd_ear::~mtmd_ear() = default;

std::vector<float> mtmd_ear::process(const float * frames, size_t n_frames) {
    if (!frames || n_frames == 0 || n_frames > 1024) { throw std::runtime_error("ear requires 1..1024 frames"); }
    if (!std::all_of(frames, frames + n_frames * 512, [](float x) { return std::isfinite(x); })) {
        throw std::runtime_error("non-finite ear input");
    }
    const size_t mem_size = 2 * 1024 * 1024 + n_frames * 256 * 1024;
    const auto work = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init({mem_size, nullptr, false}), ggml_free);
    if (!work) { throw std::runtime_error("cannot allocate ear graph"); }
    auto * ctx = work.get();
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, n_frames);
    std::copy_n(frames, n_frames * 512, static_cast<float *>(input->data));
    auto * probs = ggml_soft_max(ctx, data->linear(ctx, "ear.ctc", input, 1025));
    auto * rows = ggml_add(ctx, ggml_mul_mat(ctx, data->weight("ear.proj.weight", 1025, 5120), probs), data->residual(ctx, "ear", input));

    // Pool over time, not over feature channels. Variance uses population normalization.
    auto * time_major = ggml_cont(ctx, ggml_transpose(ctx, input));
    auto * mean = ggml_mean(ctx, time_major);
    auto * centered = ggml_sub(ctx, time_major, mean);
    auto * variance = ggml_mean(ctx, ggml_sqr(ctx, centered));
    auto * stddev = ggml_sqrt(ctx, ggml_scale_bias(ctx, variance, 1.0f, 1e-6f));
    auto * pool = ggml_concat(ctx, ggml_reshape_1d(ctx, mean, 512), ggml_reshape_1d(ctx, stddev, 512), 0);
    auto * tone_probs = ggml_soft_max(ctx, data->linear(ctx, "ear_tone.probe", pool, 6));
    auto * word = ggml_cont(ctx, ggml_transpose(ctx, data->weight("ear_tone.word_emb", 5120, 6)));
    auto * tone = ggml_add(ctx, ggml_mul_mat(ctx, word, tone_probs), data->residual(ctx, "ear_tone", pool));
    auto * output = ggml_concat(ctx, rows, ggml_reshape_2d(ctx, tone, 5120, 1), 1);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    auto plan = ggml_graph_plan(graph, 2, nullptr);
    std::vector<uint8_t> scratch(plan.work_size);
    plan.work_data = scratch.data();
    if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) { throw std::runtime_error("ear graph execution failed"); }
    const auto * begin = static_cast<const float *>(output->data);
    std::vector<float> result(begin, begin + (n_frames + 1) * 5120);
    if (!std::all_of(result.begin(), result.end(), [](float x) { return std::isfinite(x); })) {
        throw std::runtime_error("non-finite ear output");
    }
    return result;
}
