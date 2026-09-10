#include "mtmd-vad.h"

#include "ggml.h"
#include "mtmd-backend.h"
#include "mtmd-graph.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {
struct context_deleter {
    void operator()(ggml_context * ctx) const { ggml_free(ctx); }
};
using context_ptr = std::unique_ptr<ggml_context, context_deleter>;
}

struct mtmd_vad::impl {
    context_ptr weights;
    ggml_context * loaded = nullptr;
    std::vector<float> weight_data;
    context_ptr work;
    ggml_cgraph * graph = nullptr;
    std::unique_ptr<mtmd_backend> backend;
    mtmd_buffer_ptr buffer{nullptr, ggml_backend_buffer_free};
    ggml_tensor * input = nullptr;
    ggml_tensor * h_in = nullptr;
    ggml_tensor * c_in = nullptr;
    ggml_tensor * h_out = nullptr;
    ggml_tensor * c_out = nullptr;
    ggml_tensor * probability = nullptr;
    std::array<float, 64> context{};

    ggml_tensor * weight(const std::string & name, int64_t a, int64_t b = 1, int64_t c = 1) {
        auto * t = ggml_get_tensor(loaded, ("vad." + name).c_str());
        if (!t || t->type != GGML_TYPE_F32 || t->ne[0] != a || t->ne[1] != b || t->ne[2] != c || t->ne[3] != 1) {
            throw std::runtime_error("invalid VAD tensor: " + name);
        }
        return t;
    }

    ggml_tensor * conv(ggml_tensor * x, ggml_tensor * w, int stride, int padding) {
        return mtmd_conv_1d_f32(work.get(), x, w, stride, padding);
    }

    explicit impl(const std::string & path, bool use_gpu) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() <= 0 || file.tellg() > 2 * 1024 * 1024) {
            throw std::runtime_error("VAD fixture must be at most 2 MiB");
        }
        ggml_context * raw = nullptr;
        auto * meta = gguf_init_from_file(path.c_str(), {true, &raw});
        weights.reset(raw);
        if (!meta) {
            throw std::runtime_error("cannot load VAD GGUF");
        }
        const auto metadata = std::unique_ptr<gguf_context, decltype(&gguf_free)>(meta, gguf_free);
        const auto file_size = static_cast<size_t>(file.tellg());
        const size_t data_offset = gguf_get_data_offset(meta);
        if (gguf_get_n_tensors(meta) != 14 || data_offset > file_size) {
            throw std::runtime_error("invalid VAD tensor inventory");
        }
        size_t total = 0;
        for (int64_t i = 0; i < gguf_get_n_tensors(meta); ++i) {
            auto * t = ggml_get_tensor(weights.get(), gguf_get_tensor_name(meta, i));
            const size_t bytes = ggml_nbytes(t);
            const size_t offset = gguf_get_tensor_offset(meta, i);
            if (t->type != GGML_TYPE_F32 || bytes > 2 * 1024 * 1024 - total ||
                    offset > file_size - data_offset || bytes > file_size - data_offset - offset) {
                throw std::runtime_error("invalid VAD tensor extent");
            }
            total += bytes;
        }
        weight_data.resize(total / sizeof(float));
        size_t cursor = 0;
        for (int64_t i = 0; i < gguf_get_n_tensors(meta); ++i) {
            auto * t = ggml_get_tensor(weights.get(), gguf_get_tensor_name(meta, i));
            t->data = weight_data.data() + cursor;
            file.seekg(data_offset + gguf_get_tensor_offset(meta, i));
            if (!file.read(static_cast<char *>(t->data), ggml_nbytes(t))) {
                throw std::runtime_error("cannot read VAD tensor");
            }
            cursor += ggml_nelements(t);
        }
        loaded = weights.get();
        backend = std::make_unique<mtmd_backend>(loaded, use_gpu, "vad.");
        loaded = backend->context();
        build();
    }

    explicit impl(ggml_context * source, bool use_gpu) : loaded(source) {
        if (!loaded) {
            throw std::runtime_error("missing VAD weights");
        }
        backend = std::make_unique<mtmd_backend>(loaded, use_gpu, "vad.");
        loaded = backend->context();
        build();
    }

    void build() {
        work.reset(ggml_init({16 * 1024 * 1024, nullptr, true}));
        if (!work) {
            throw std::runtime_error("cannot allocate VAD graph");
        }
        auto * ctx = work.get();
        input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 576, 1);
        h_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
        c_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
        auto * z = conv(ggml_pad_reflect_1d(ctx, input, 0, 64), weight("stft_conv.weight", 256, 1, 258), 128, 0);
        auto * re = ggml_view_2d(ctx, z, 4, 129, z->nb[1], 0);
        auto * im = ggml_view_2d(ctx, z, 4, 129, z->nb[1], 129 * z->nb[1]);
        z = ggml_sqrt(ctx, ggml_add(ctx, ggml_sqr(ctx, re), ggml_sqr(ctx, im)));
        const int channels[] = {129, 128, 64, 64, 128};
        for (int i = 0; i < 4; ++i) {
            const auto name = "conv" + std::to_string(i + 1);
            z = conv(z, weight(name + ".weight", 3, channels[i], channels[i + 1]), i == 1 || i == 2 ? 2 : 1, 1);
            auto * bias = ggml_reshape_2d(ctx, weight(name + ".bias", channels[i + 1]), 1, channels[i + 1]);
            z = ggml_relu(ctx, ggml_add(ctx, z, bias));
        }
        z = ggml_reshape_1d(ctx, z, 128);
        auto * wx = ggml_add(ctx, ggml_mul_mat(ctx, weight("lstm_cell.Wx", 128, 512), z),
                                 weight("lstm_cell.bias", 512));
        const auto state = mtmd_lstm_step(ctx, wx, weight("lstm_cell.Wh", 128, 512), {h_in, c_in});
        h_out = state.h;
        c_out = state.c;
        probability = ggml_sigmoid(ctx, ggml_add(ctx,
                ggml_mul_mat(ctx, ggml_reshape_2d(ctx, weight("final_conv.weight", 1, 128, 1), 128, 1), ggml_relu(ctx, h_out)),
                weight("final_conv.bias", 1)));
        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, probability);
        buffer = backend->allocate(ctx);
    }
};

mtmd_vad::mtmd_vad(const std::string & path, bool use_gpu) : data(std::make_unique<impl>(path, use_gpu)) { reset(); }
mtmd_vad::mtmd_vad(ggml_context * weights, bool use_gpu) : data(std::make_unique<impl>(weights, use_gpu)) { reset(); }
mtmd_vad::~mtmd_vad() = default;

void mtmd_vad::reset() {
    data->context.fill(0);
    ggml_backend_tensor_memset(data->h_in, 0, 0, 128 * sizeof(float));
    ggml_backend_tensor_memset(data->c_in, 0, 0, 128 * sizeof(float));
}

float mtmd_vad::process(const std::array<float, 512> & samples) {
    if (!std::all_of(samples.begin(), samples.end(), [](float x) { return std::isfinite(x); })) {
        throw std::runtime_error("VAD input contains non-finite samples");
    }
    std::array<float, 576> input;
    std::copy(data->context.begin(), data->context.end(), input.begin());
    std::copy(samples.begin(), samples.end(), input.begin() + 64);
    ggml_backend_tensor_set(data->input, input.data(), 0, sizeof(input));
    data->backend->compute(data->graph);
    float p;
    ggml_backend_tensor_get(data->probability, &p, 0, sizeof(p));
    if (!std::isfinite(p)) {
        throw std::runtime_error("VAD produced a non-finite probability");
    }
    std::copy(samples.end() - 64, samples.end(), data->context.begin());
    ggml_backend_tensor_copy(data->h_out, data->h_in);
    ggml_backend_tensor_copy(data->c_out, data->c_in);
    return p;
}
