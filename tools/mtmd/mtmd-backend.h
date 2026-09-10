#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <cmath>
#include <cstdio>

using mtmd_buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;

// Own a device copy of immutable component weights. No CPU fallback for GPU requests.
class mtmd_backend {
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend{nullptr, ggml_backend_free};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> weights{nullptr, ggml_free};
    mtmd_buffer_ptr buffer{nullptr, ggml_backend_buffer_free};
    std::string name;
    size_t peak_graph_bytes = 0;

public:
    mtmd_backend(ggml_context * source, bool use_gpu, const std::string & prefix) : name(prefix) {
        if (!source) { throw std::runtime_error("missing component weights"); }
        backend.reset(ggml_backend_init_by_type(use_gpu ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
        if (!backend) { throw std::runtime_error("component backend unavailable"); }
        if (!use_gpu) { ggml_backend_cpu_set_n_threads(backend.get(), 2); }
        size_t count = 0, bytes = 0;
        constexpr size_t limit = 128 * 1024 * 1024;
        for (auto * t = ggml_get_first_tensor(source); t; t = ggml_get_next_tensor(source, t)) {
            if (std::string(t->name).rfind(prefix, 0) != 0) { continue; }
            if (++count > 256 || !t->data || t->type != GGML_TYPE_F32 || ggml_nbytes(t) > limit - bytes) {
                throw std::runtime_error("invalid component weight inventory");
            }
            const auto * values = static_cast<const float *>(t->data);
            for (int64_t i = 0; i < ggml_nelements(t); ++i) {
                if (!std::isfinite(values[i])) { throw std::runtime_error("non-finite component weight"); }
            }
            bytes += ggml_nbytes(t);
        }
        weights.reset(ggml_init({(count + 1) * ggml_tensor_overhead(), nullptr, true}));
        if (!weights) { throw std::runtime_error("cannot allocate component weight metadata"); }
        for (auto * t = ggml_get_first_tensor(source); t; t = ggml_get_next_tensor(source, t)) {
            if (std::string(t->name).rfind(prefix, 0) != 0) { continue; }
            ggml_set_name(ggml_dup_tensor(weights.get(), t), t->name);
        }
        buffer = allocate(weights.get());
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::fprintf(stderr, "memory component=%s backend=%s weights=%zu\n", name.c_str(), ggml_backend_name(backend.get()), ggml_backend_buffer_get_size(buffer.get()));
        for (auto * t = ggml_get_first_tensor(source); t; t = ggml_get_next_tensor(source, t)) {
            if (std::string(t->name).rfind(prefix, 0) != 0) { continue; }
            ggml_backend_tensor_set(ggml_get_tensor(weights.get(), t->name), t->data, 0, ggml_nbytes(t));
        }
    }

    ggml_context * context() const { return weights.get(); }

    mtmd_buffer_ptr allocate(ggml_context * ctx) {
        mtmd_buffer_ptr result(ggml_backend_alloc_ctx_tensors(ctx, backend.get()), ggml_backend_buffer_free);
        if (!result) { throw std::runtime_error("cannot allocate component tensors"); }
        if (ctx != weights.get()) {
            const auto bytes = ggml_backend_buffer_get_size(result.get());
            if (bytes > peak_graph_bytes) {
                peak_graph_bytes = bytes;
                std::fprintf(stderr, "memory component=%s backend=%s peak_graph_buffer=%zu\n", name.c_str(), ggml_backend_name(backend.get()), bytes);
            }
        }
        return result;
    }

    void compute(ggml_cgraph * graph) {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_MUL_MAT) { ggml_prec_set_src(node, GGML_PREC_F32, 1); }
            if (!ggml_backend_supports_op(backend.get(), node)) {
                throw std::runtime_error(std::string("component backend does not support ") + ggml_op_name(node->op));
            }
        }
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("component graph execution failed");
        }
    }
};
