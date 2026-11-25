// AudioSeal streaming generator, see audioseal/libs/moshi/modules/{seanet,conv,lstm,streaming}.py.
// Copyright (c) Meta Platforms, Inc. and affiliates. MIT license.
#include "mtmd-watermark.h"
#include "mtmd-backend.h"
#include "mtmd-graph.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {
constexpr int dim = 128, hidden = 512, bits = 16;
// SEANet ratios [8, 5, 4, 2]: the encoder walks them backwards, the decoder forwards.
constexpr int ratios[] = {8, 5, 4, 2};
}

struct mtmd_watermark::impl {
    mtmd_backend backend;
    ggml_context * source;
    ggml_context_ptr work;
    ggml_gallocr_ptr allocator;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr, * msg = nullptr, * output = nullptr;
    // Streaming history: each slot's input is the previous step's output.
    struct slot { ggml_tensor * in; ggml_tensor * out; };
    std::vector<slot> slots;
    size_t state_size = 0;
    std::mutex mutex;

    ggml_tensor * w(const std::string & name, int64_t a, int64_t b = 1, int64_t c = 1) const {
        const auto key = "wm." + name;
        auto * t = ggml_get_tensor(backend.context(), key.c_str());
        if (!t || t->ne[0] != a || t->ne[1] != b || t->ne[2] != c || t->ne[3] != 1) {
            throw std::runtime_error("invalid watermark tensor: " + key);
        }
        return t;
    }

    ggml_tensor * history(ggml_tensor * value, ggml_tensor * next) {
        ggml_set_input(value);
        slots.push_back({value, ggml_cont(work.get(), next)});
        state_size += ggml_nelements(value);
        return value;
    }

    // Causal conv on [T, IC]: the kernel - stride left context is the previous step's tail,
    // exactly the streaming conv's single zero pad followed by its retained input.
    ggml_tensor * conv(ggml_tensor * x, const std::string & name, int in, int out, int kernel, int stride = 1) {
        auto * ctx = work.get();
        const int pad = kernel - stride;
        if (pad > 0) {
            auto * left = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pad, in);
            x = ggml_concat(ctx, left, x, 0);
            history(left, ggml_view_2d(ctx, x, pad, in, x->nb[1], (x->ne[0] - pad) * x->nb[0]));
        }
        auto * y = mtmd_conv_1d_f32(ctx, x, w(name + ".weight", kernel, in, out), stride, 0);
        return ggml_add(ctx, y, ggml_reshape_2d(ctx, w(name + ".bias", out), 1, out));
    }

    // Causal transposed conv on [T, IC]: the kernel - stride overlap tail is added to the next step.
    ggml_tensor * convtr(ggml_tensor * x, const std::string & name, int in, int out, int stride) {
        auto * ctx = work.get();
        const int kernel = 2 * stride, tail = kernel - stride;
        const int64_t length = x->ne[0] * stride;
        // The converter stores the weight as [IC, K, OC], ready for one matmul per step.
        auto * weight = ggml_reshape_2d(ctx, w(name + ".weight", in, kernel, out), in, kernel * out);
        auto * col = ggml_mul_mat(ctx, weight, ggml_cont(ctx, ggml_transpose(ctx, x)));
        auto * full = ggml_col2im_1d(ctx, col, stride, out, 0);  // [length + tail, OC]
        auto * previous = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tail, out);
        history(previous, ggml_view_2d(ctx, full, tail, out, full->nb[1], length * full->nb[0]));
        auto * head = ggml_add(ctx, ggml_view_2d(ctx, full, tail, out, full->nb[1], 0), previous);
        auto * y = head;
        if (length > tail) {
            y = ggml_concat(ctx, head, ggml_view_2d(ctx, full, length - tail, out, full->nb[1], tail * full->nb[0]), 0);
        }
        return ggml_add(ctx, y, ggml_reshape_2d(ctx, w(name + ".bias", out), 1, out));
    }

    ggml_tensor * residual(ggml_tensor * x, const std::string & name, int channels) {
        auto * ctx = work.get();
        auto * h = conv(ggml_elu(ctx, x), name + ".block.1", channels, channels / 2, 3);
        h = conv(ggml_elu(ctx, h), name + ".block.3", channels / 2, channels, 1);
        return ggml_add(ctx, x, h);
    }

    // Two-layer LSTM over one frame, with the skip connection. x: [1, hidden].
    ggml_tensor * lstm(ggml_tensor * x, const std::string & name) {
        auto * ctx = work.get();
        auto * in = ggml_reshape_1d(ctx, x, hidden);
        auto * cur = in;
        for (int layer = 0; layer < 2; ++layer) {
            const auto l = std::to_string(layer);
            mtmd_lstm_state state{ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden), ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden)};
            auto * wx = ggml_mul_mat(ctx, w(name + ".weight_ih_l" + l, hidden, 4 * hidden), cur);
            wx = ggml_add(ctx, wx, ggml_add(ctx, w(name + ".bias_ih_l" + l, 4 * hidden), w(name + ".bias_hh_l" + l, 4 * hidden)));
            const auto next = mtmd_lstm_step(ctx, wx, w(name + ".weight_hh_l" + l, hidden, 4 * hidden), state);
            history(state.h, next.h);
            history(state.c, next.c);
            cur = next.h;
        }
        return ggml_reshape_2d(ctx, ggml_add(ctx, cur, in), 1, hidden);
    }

    impl(ggml_context * weights, bool gpu) : backend(weights, gpu, "wm."), source(weights) {
        work.reset(ggml_init({8 * 1024 * 1024, nullptr, true}));
        if (!work) { throw std::runtime_error("cannot allocate watermark graph"); }
        auto * ctx = work.get();
        input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, frame, 1);
        msg = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, dim);
        ggml_set_input(input);
        ggml_set_input(msg);

        // Encoder: model.0 conv, then per ratio a residual block, ELU and strided conv.
        auto * x = conv(input, "encoder.model.0", 1, 32, 7);
        int channels = 32, index = 1;
        for (int i = 3; i >= 0; --i, index += 3, channels *= 2) {
            const auto p = "encoder.model.";
            x = residual(x, p + std::to_string(index), channels);
            x = conv(ggml_elu(ctx, x), p + std::to_string(index + 2), channels, 2 * channels, 2 * ratios[i], ratios[i]);
        }
        x = lstm(x, "encoder.model.13");
        x = conv(ggml_elu(ctx, x), "encoder.model.15", hidden, dim, 7);
        x = ggml_add(ctx, x, msg);

        // Decoder: conv, LSTM, then per ratio ELU, transposed conv and a residual block.
        x = conv(x, "decoder.model.0", dim, hidden, 7);
        x = lstm(x, "decoder.model.1");
        channels = hidden; index = 3;
        for (int i = 0; i < 4; ++i, index += 3, channels /= 2) {
            const auto p = "decoder.model.";
            x = convtr(ggml_elu(ctx, x), p + std::to_string(index), channels, channels / 2, ratios[i]);
            x = residual(x, p + std::to_string(index + 1), channels / 2);
        }
        output = conv(ggml_elu(ctx, x), "decoder.model.15", 32, 1, 7);
        if (output->ne[0] != frame || output->ne[1] != 1) { throw std::runtime_error("invalid watermark frame geometry"); }
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx, 2048, false);
        ggml_build_forward_expand(graph, output);
        for (const auto & s : slots) { ggml_set_output(s.out); ggml_build_forward_expand(graph, s.out); }
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_MUL_MAT) { ggml_prec_set_acc(node, GGML_PREC_F32); }
        }
        allocator = backend.allocate(graph);
    }
};

mtmd_watermark::mtmd_watermark(ggml_context * weights, bool use_gpu) : data(std::make_unique<impl>(weights, use_gpu)) {}
mtmd_watermark::~mtmd_watermark() = default;

std::vector<float> mtmd_watermark::message(uint16_t payload) const {
    const auto * table = ggml_get_tensor(data->source, "wm.message.weight");
    if (!table || table->type != GGML_TYPE_F32 || table->ne[0] != dim || table->ne[1] != 2 * bits || !table->data) {
        throw std::runtime_error("invalid watermark message table");
    }
    const auto * rows = static_cast<const float *>(table->data);
    std::vector<float> sum(dim, 0.0f);
    for (int i = 0; i < bits; ++i) {
        const int bit = (payload >> (bits - 1 - i)) & 1;
        for (int h = 0; h < dim; ++h) { sum[h] += rows[(2 * i + bit) * dim + h]; }
    }
    return sum;
}

std::vector<float> mtmd_watermark::initial_state() const {
    return std::vector<float>(data->state_size, 0.0f);
}

void mtmd_watermark::process(std::vector<float> & state, const std::vector<float> & message, const float * in, float * out) {
    if (state.size() != data->state_size || message.size() != dim) {
        throw std::runtime_error("invalid watermark stream state");
    }
    if (!std::all_of(in, in + frame, [](float v) { return std::isfinite(v); })) {
        throw std::runtime_error("non-finite watermark input");
    }
    std::vector<float> wm(frame);
    std::lock_guard<std::mutex> lock(data->mutex);
    size_t offset = 0;
    for (const auto & s : data->slots) {
        ggml_backend_tensor_set(s.in, state.data() + offset, 0, ggml_nbytes(s.in));
        offset += ggml_nelements(s.in);
    }
    ggml_backend_tensor_set(data->input, in, 0, frame * sizeof(float));
    ggml_backend_tensor_set(data->msg, message.data(), 0, dim * sizeof(float));
    data->backend.compute(data->graph);
    ggml_backend_tensor_get(data->output, wm.data(), 0, frame * sizeof(float));
    std::vector<float> next(state.size());
    offset = 0;
    for (const auto & s : data->slots) {
        ggml_backend_tensor_get(s.out, next.data() + offset, 0, ggml_nbytes(s.in));
        offset += ggml_nelements(s.in);
    }
    for (int i = 0; i < frame; ++i) {
        if (!std::isfinite(wm[i])) { throw std::runtime_error("non-finite watermark output"); }
        out[i] = in[i] + wm[i];
    }
    state = std::move(next);
}
