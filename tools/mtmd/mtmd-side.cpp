#include "mtmd-side.h"

#include "ggml.h"
#include "mtmd-backend.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

struct mtmd_side::impl {
    ggml_context * weight_ctx = nullptr;
    ggml_context_ptr work;
    ggml_cgraph * graph = nullptr;
    std::unique_ptr<mtmd_backend> backend;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;

    ggml_tensor * weight(const std::string & name, int64_t a, int64_t b = 1) {
        auto * t = ggml_get_tensor(weight_ctx, ("side." + name).c_str());
        if (!t || t->type != GGML_TYPE_F32 || t->ne[0] != a || t->ne[1] != b || t->ne[2] != 1 || t->ne[3] != 1) {
            throw std::runtime_error("invalid side tensor: " + name);
        }
        return t;
    }

    explicit impl(ggml_context * loaded, bool use_gpu) : weight_ctx(loaded) {
        if (!loaded) { throw std::runtime_error("missing side weights"); }
        weight("ln.weight", 5120); weight("ln.bias", 5120);
        weight("up.weight", 5120, 1024); weight("up.bias", 1024);
        weight("down.weight", 1024, 2048); weight("down.bias", 2048);
        backend = std::make_unique<mtmd_backend>(weight_ctx, use_gpu, "side.");
        weight_ctx = backend->context();
        init_graph();
    }

    void init_graph() {
        work.reset(ggml_init({1024 * 1024, nullptr, true}));
        if (!work) { throw std::runtime_error("cannot allocate side graph"); }
        auto * ctx = work.get();
        input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5120);
        auto * norm = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, input, 1e-5f), weight("ln.weight", 5120)), weight("ln.bias", 5120));
        auto * up = ggml_add(ctx, ggml_mul_mat(ctx, weight("up.weight", 5120, 1024), norm), weight("up.bias", 1024));
        output = ggml_add(ctx, ggml_mul_mat(ctx, weight("down.weight", 1024, 2048), ggml_gelu_erf(ctx, up)), weight("down.bias", 2048));
        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        buffer = backend->allocate(ctx, graph);
    }
};

mtmd_side::mtmd_side(ggml_context * weights, bool use_gpu) : data(std::make_unique<impl>(weights, use_gpu)) {}
mtmd_side::~mtmd_side() = default;

std::array<float, 2048> mtmd_side::process(const std::array<float, 5120> & hidden) {
    if (!std::all_of(hidden.begin(), hidden.end(), [](float x) { return std::isfinite(x); })) {
        throw std::runtime_error("side input contains non-finite values");
    }
    ggml_backend_tensor_set(data->input, hidden.data(), 0, sizeof(hidden));
    data->backend->compute(data->graph);
    std::array<float, 2048> result;
    ggml_backend_tensor_get(data->output, result.data(), 0, sizeof(result));
    if (!std::all_of(result.begin(), result.end(), [](float x) { return std::isfinite(x); })) {
        throw std::runtime_error("side produced non-finite values");
    }
    return result;
}

struct mtmd_expression::impl {
    mtmd_backend backend;
    ggml_context_ptr work;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * input = nullptr;
    ggml_tensor * probabilities = nullptr;
    ggml_tensor * offset = nullptr;
    ggml_cgraph * graph = nullptr;

    explicit impl(ggml_context * weights, bool use_gpu) : backend(weights, use_gpu, "expression.") {
        auto weight = [&](const char * name, int64_t a, int64_t b = 1) {
            const std::string key = std::string("expression.") + name;
            auto * t = ggml_get_tensor(backend.context(), key.c_str());
            if (!t || t->type != GGML_TYPE_F32 || t->ne[0] != a || t->ne[1] != b || t->ne[2] != 1 || t->ne[3] != 1) {
                throw std::runtime_error("invalid expression tensor: " + key);
            }
            return t;
        };
        auto * sd_source = ggml_get_tensor(weights, "expression.sd");
        if (!sd_source || !sd_source->data || sd_source->type != GGML_TYPE_F32 || ggml_nelements(sd_source) != 5120) {
            throw std::runtime_error("missing expression normalization");
        }
        const auto * sd = static_cast<const float *>(sd_source->data);
        if (!std::all_of(sd, sd + 5120, [](float x) { return x > 0; })) {
            throw std::runtime_error("invalid expression standard deviation");
        }
        work.reset(ggml_init({1024 * 1024, nullptr, true}));
        if (!work) { throw std::runtime_error("cannot allocate expression graph"); }
        auto * ctx = work.get();
        input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5120);
        auto * h = ggml_div(ctx, ggml_sub(ctx, input, weight("mu", 5120)), weight("sd", 5120));
        auto * logits = ggml_add(ctx, ggml_mul_mat(ctx, weight("weight", 5120, 4), h), weight("bias", 4));
        probabilities = ggml_soft_max(ctx, logits);
        auto * gated = ggml_clamp(ctx, ggml_scale_bias(ctx, probabilities, 2.0f, -1.0f), 0.0f, 1.0f);
        offset = ggml_mul_mat(ctx, weight("directions", 4, 2048), gated);
        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, offset);
        buffer = backend.allocate(ctx, graph);
    }
};

mtmd_expression::mtmd_expression(ggml_context * weights, bool use_gpu) : data(std::make_unique<impl>(weights, use_gpu)) {}
mtmd_expression::~mtmd_expression() = default;

mtmd_expression::result mtmd_expression::process(const std::array<float, 5120> & pooled) {
    if (!std::all_of(pooled.begin(), pooled.end(), [](float x) { return std::isfinite(x); })) {
        throw std::runtime_error("expression input contains non-finite values");
    }
    ggml_backend_tensor_set(data->input, pooled.data(), 0, sizeof(pooled));
    data->backend.compute(data->graph);
    result value;
    ggml_backend_tensor_get(data->probabilities, value.probabilities.data(), 0, sizeof(value.probabilities));
    ggml_backend_tensor_get(data->offset, value.offset.data(), 0, sizeof(value.offset));
    if (!std::all_of(value.offset.begin(), value.offset.end(), [](float x) { return std::isfinite(x); })) {
        throw std::runtime_error("expression produced non-finite values");
    }
    return value;
}

std::array<float, 2048> mtmd_expression::direction(size_t emotion) const {
    if (emotion >= 4) { throw std::runtime_error("expression class out of range"); }
    std::array<float, 8192> directions;
    ggml_backend_tensor_get(ggml_get_tensor(data->backend.context(), "expression.directions"),
                            directions.data(), 0, sizeof(directions));
    std::array<float, 2048> result;
    for (size_t i = 0; i < result.size(); ++i) { result[i] = directions[4 * i + emotion]; }
    return result;
}
