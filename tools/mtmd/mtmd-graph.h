#pragma once

#include "ggml.h"

// Shared F32 convolution and recurrent cell for the small streaming audio models.
inline ggml_tensor * mtmd_conv_1d_f32(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w,
                                     int stride, int padding) {
    auto * col = ggml_im2col(ctx, w, x, stride, 0, padding, 0, 1, 0, false, GGML_TYPE_F32);
    auto * y = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1] * col->ne[2]),
                                ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]));
    return ggml_reshape_2d(ctx, y, col->ne[1], w->ne[2]);
}

struct mtmd_lstm_state { ggml_tensor * h; ggml_tensor * c; };

// wx is the input projection, including bias. Gate order follows PyTorch: i, f, g, o.
inline mtmd_lstm_state mtmd_lstm_step(ggml_context * ctx, ggml_tensor * wx, ggml_tensor * wh,
                                     mtmd_lstm_state state) {
    const int64_t n = state.h->ne[0];
    auto * gates = ggml_add(ctx, wx, ggml_mul_mat(ctx, wh, state.h));
    auto gate = [&](int i) { return ggml_view_1d(ctx, gates, n, i * n * sizeof(float)); };
    auto * c = ggml_add(ctx, ggml_mul(ctx, ggml_sigmoid(ctx, gate(1)), state.c),
                            ggml_mul(ctx, ggml_sigmoid(ctx, gate(0)), ggml_tanh(ctx, gate(2))));
    return {ggml_mul(ctx, ggml_sigmoid(ctx, gate(3)), ggml_tanh(ctx, c)), c};
}
