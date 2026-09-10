#include "mtmd-side.h"

#include "ggml.h"
#include "mtmd-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {
struct context_deleter {
    void operator()(ggml_context * ctx) const { ggml_free(ctx); }
};
using context_ptr = std::unique_ptr<ggml_context, context_deleter>;
}

struct mtmd_side::impl {
    context_ptr weights;
    ggml_context * weight_ctx = nullptr;
    std::vector<float> storage;
    context_ptr work;
    ggml_cgraph * graph = nullptr;
    std::unique_ptr<mtmd_backend> backend;
    mtmd_buffer_ptr buffer{nullptr, ggml_backend_buffer_free};
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;

    ggml_tensor * weight(const std::string & name, int64_t a, int64_t b = 1) {
        auto * t = ggml_get_tensor(weight_ctx, ("side." + name).c_str());
        if (!t || t->type != GGML_TYPE_F32 || t->ne[0] != a || t->ne[1] != b || t->ne[2] != 1 || t->ne[3] != 1) {
            throw std::runtime_error("invalid side tensor: " + name);
        }
        return t;
    }

    explicit impl(const std::string & path, bool use_gpu) {
        constexpr size_t limit = 32 * 1024 * 1024;
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() <= 0 || file.tellg() > static_cast<std::streamoff>(limit)) {
            throw std::runtime_error("side fixture must be at most 32 MiB");
        }
        const size_t file_size = static_cast<size_t>(file.tellg());
        ggml_context * raw = nullptr;
        auto * meta = gguf_init_from_file(path.c_str(), {true, &raw});
        weights.reset(raw);
        weight_ctx = raw;
        if (!meta) { throw std::runtime_error("cannot load side GGUF"); }
        const auto metadata = std::unique_ptr<gguf_context, decltype(&gguf_free)>(meta, gguf_free);
        const size_t data_offset = gguf_get_data_offset(meta);
        if (gguf_get_n_tensors(meta) != 6 || data_offset > file_size) {
            throw std::runtime_error("invalid side tensor inventory");
        }
        weight("ln.weight", 5120); weight("ln.bias", 5120);
        weight("up.weight", 5120, 1024); weight("up.bias", 1024);
        weight("down.weight", 1024, 2048); weight("down.bias", 2048);
        size_t total = 0;
        for (int64_t i = 0; i < gguf_get_n_tensors(meta); ++i) {
            auto * t = ggml_get_tensor(weights.get(), gguf_get_tensor_name(meta, i));
            const size_t bytes = ggml_nbytes(t);
            const size_t offset = gguf_get_tensor_offset(meta, i);
            if (bytes > limit - total || offset > file_size - data_offset || bytes > file_size - data_offset - offset) {
                throw std::runtime_error("invalid side tensor extent");
            }
            total += bytes;
        }
        storage.resize(total / sizeof(float));
        size_t cursor = 0;
        for (int64_t i = 0; i < gguf_get_n_tensors(meta); ++i) {
            auto * t = ggml_get_tensor(weights.get(), gguf_get_tensor_name(meta, i));
            t->data = storage.data() + cursor;
            file.seekg(data_offset + gguf_get_tensor_offset(meta, i));
            if (!file.read(static_cast<char *>(t->data), ggml_nbytes(t))) {
                throw std::runtime_error("cannot read side tensor");
            }
            const auto * begin = static_cast<const float *>(t->data);
            if (!std::all_of(begin, begin + ggml_nelements(t), [](float x) { return std::isfinite(x); })) {
                throw std::runtime_error("non-finite side weights");
            }
            cursor += ggml_nelements(t);
        }
        backend = std::make_unique<mtmd_backend>(weight_ctx, use_gpu, "side.");
        weight_ctx = backend->context();
        init_graph();
    }

    explicit impl(ggml_context * loaded, bool use_gpu) : weight_ctx(loaded) {
        if (!loaded) { throw std::runtime_error("missing side weights"); }
        weight("ln.weight", 5120); weight("ln.bias", 5120);
        weight("up.weight", 5120, 1024); weight("up.bias", 1024);
        weight("down.weight", 1024, 2048); weight("down.bias", 2048);
        for (auto * t = ggml_get_first_tensor(loaded); t; t = ggml_get_next_tensor(loaded, t)) {
            if (std::string(t->name).rfind("side.", 0) != 0) { continue; }
            if (!t->data || t->type != GGML_TYPE_F32) { throw std::runtime_error("side weights must be host F32"); }
            const auto * begin = static_cast<const float *>(t->data);
            if (!std::all_of(begin, begin + ggml_nelements(t), [](float v) { return std::isfinite(v); })) {
                throw std::runtime_error("non-finite side weights");
            }
        }
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
        buffer = backend->allocate(ctx);
    }
};

mtmd_side::mtmd_side(const std::string & path, bool use_gpu) : data(std::make_unique<impl>(path, use_gpu)) {}
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
