#include "models.h"

// voice-prompt encoder: raw 24kHz waveform -> one conditioning row per 12.5Hz frame
// mimi encoder (SEANet + transformer + downsample), then flow_lm.speaker_proj_weight

// pre-norm block with layer scale on both residual paths, see mimi_transformer.py
ggml_tensor * clip_graph_pockettts_spkenc::tfm_layer_forward(ggml_tensor * cur, const clip_layer & layer, ggml_tensor * inp_pos, ggml_tensor * kq_mask, int il) const {
    ggml_tensor * inp = cur;

    cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);

    ggml_tensor * Qcur = build_mm(layer.q_w, cur);
    ggml_tensor * Kcur = build_mm(layer.k_w, cur);
    ggml_tensor * Vcur = build_mm(layer.v_w, cur);

    const int64_t n_pos = cur->ne[1];
    Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
    Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
    Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);

    const int rope_type = model.mimi_quant_in[0] ? GGML_ROPE_TYPE_NEOX : GGML_ROPE_TYPE_NORMAL;
    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, d_head, rope_type, 0,
                         hparams.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, d_head, rope_type, 0,
                         hparams.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    cur = build_attn(layer.o_w, nullptr, Qcur, Kcur, Vcur, kq_mask, kq_scale, il);
    cur = ggml_mul(ctx0, cur, layer.ls_1_w);
    cur = ggml_add(ctx0, cur, inp);

    inp = cur;
    cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
    cur = build_ffn(cur, layer.ff_up_w, nullptr, nullptr, nullptr, layer.ff_down_w, nullptr, FFN_GELU, il);
    cur = ggml_mul(ctx0, cur, layer.ls_2_w);
    cur = ggml_add(ctx0, cur, inp);

    return cur;
}

ggml_cgraph * clip_graph_pockettts_spkenc::build() {
    // the preprocessor hands over the waveform as a single-row "mel", already [n_samples, 1]
    ggml_tensor * inp_raw = build_inp_raw(1);
    ggml_tensor * cur = ggml_reshape_2d(ctx0, inp_raw, inp_raw->ne[0], inp_raw->ne[1]);

    clip_graph_pockettts_seanet seanet(*this);
    cur = seanet.encode(cur);
    cb(cur, "mimi_enc", -1);

    // [T, 512] -> transformer works on [512, T]
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, cur->ne[1]);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    // the mimi transformer is causal with a sliding window, see _build_attention_mask()
    ggml_tensor * kq_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cur->ne[1], cur->ne[1]);
    ggml_set_name(kq_mask, "kq_mask");
    ggml_set_input(kq_mask);

    for (int il = 0; il < n_layer; il++) {
        cur = tfm_layer_forward(cur, model.layers[il], inp_pos, kq_mask, il);
    }
    cb(cur, "mimi_enc_tfm", -1);

    // downsample to the model frame rate, [512, T] -> [T, 512] -> [T / 16, 32]
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    cur = seanet.conv1d(cur, model.downsample_w, nullptr, hparams.mimi_downsample, 1, true);
    cb(cur, "mimi_downsample", -1);

    // voice latent -> backbone embd
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    if (model.spk_proj_w) {
        cur = build_mm(model.spk_proj_w, cur);
    } else {
        // Split RVQ: the semantic and acoustic quantizers start from the same latent.
        ggml_tensor * sum = nullptr;
        for (int group = 0; group < 2; ++group) {
            auto * residual = build_mm(model.mimi_quant_in[group], cur);
            const auto * tables = model.mimi_quant_cb[group];
            const auto * norms = model.mimi_quant_norm[group];
            for (int64_t i = 0; i < tables->ne[2]; ++i) {
                auto * table = ggml_view_2d(ctx0, const_cast<ggml_tensor *>(tables), tables->ne[0], tables->ne[1],
                                           tables->nb[1], i * tables->nb[2]);
                auto * norm = ggml_view_1d(ctx0, const_cast<ggml_tensor *>(norms), norms->ne[0], i * norms->nb[1]);
                auto * logits = ggml_sub(ctx0, ggml_mul_mat(ctx0, table, residual), norm);
                auto * codes = ggml_argmax(ctx0, logits);
                auto * quantized = ggml_get_rows(ctx0, table, codes);
                residual = ggml_sub(ctx0, residual, quantized);
                auto * embedding = model.gen_code_out_embd_w;
                if (group) {
                    const auto * all = model.gen_code_embd_w;
                    embedding = ggml_view_2d(ctx0, model.gen_code_embd_w, all->ne[0], all->ne[1], all->nb[1], i * all->nb[2]);
                }
                auto * row = ggml_get_rows(ctx0, embedding, codes);
                sum = sum ? ggml_add(ctx0, sum, row) : row;
            }
        }
        cur = sum;
    }
    cb(cur, "spk_proj", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
