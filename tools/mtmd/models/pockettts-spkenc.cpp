#include "models.h"

// voice-prompt encoder: raw 24kHz waveform -> one conditioning row per 12.5Hz frame
// mimi encoder (SEANet + transformer + downsample), then flow_lm.speaker_proj_weight

// pre-norm block with layer scale on both residual paths, see mimi_transformer.py
ggml_tensor * mtmd_graph_pockettts_spkenc::tfm_layer_forward(ggml_tensor * cur, const mtmd_layer & layer, ggml_tensor * inp_pos, ggml_tensor * kq_mask, int il,
                                                                 mtmd_graph_pockettts_seanet * state) const {
    const int64_t n_pos      = cur->ne[1];
    const int     tfm_d_head = (int) cur->ne[0] / tfm_n_head;

    ggml_tensor * inp = cur;

    cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, tfm_eps, il);

    ggml_tensor * Qcur = build_mm(layer.q_w, cur);
    ggml_tensor * Kcur = build_mm(layer.k_w, cur);
    ggml_tensor * Vcur = build_mm(layer.v_w, cur);

    Qcur = ggml_reshape_3d(ctx0, Qcur, tfm_d_head, tfm_n_head, n_pos);
    Kcur = ggml_reshape_3d(ctx0, Kcur, tfm_d_head, tfm_n_head, n_pos);
    Vcur = ggml_reshape_3d(ctx0, Vcur, tfm_d_head, tfm_n_head, n_pos);

    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, tfm_d_head, tfm_rope_type, 0,
                         tfm_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, tfm_d_head, tfm_rope_type, 0,
                         tfm_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    if (state) {
        const int64_t n_embd = tfm_d_head * tfm_n_head;
        const int64_t prefix = hparams.mimi_tfm_context - 1;
        const auto append = [&](ggml_tensor * x, const std::string & name) {
            x = ggml_reshape_2d(ctx0, x, n_embd, n_pos);
            if (hparams.gen_model_variant == "mimi") {
                ggml_tensor * full = ggml_set_rows(ctx0, ggml_dup(ctx0, state->state_in.at(name)), x,
                                                   state->state_in.at("cache_pos"));
                state->state_out.push_back({name, full});
                return ggml_reshape_3d(ctx0, full, tfm_d_head, tfm_n_head, hparams.mimi_tfm_context);
            }
            ggml_tensor * full = ggml_concat(ctx0, state->state_in.at(name), x, 1);
            state->state_out.push_back({name, ggml_cont(ctx0,
                ggml_view_2d(ctx0, full, n_embd, prefix, full->nb[1], (size_t) n_pos * full->nb[1]))});
            return ggml_reshape_3d(ctx0, full, tfm_d_head, tfm_n_head, prefix + n_pos);
        };
        Kcur = append(Kcur, "tfm_k_" + std::to_string(il));
        Vcur = append(Vcur, "tfm_v_" + std::to_string(il));
    }

    cur = build_attn(layer.o_w, nullptr, Qcur, Kcur, Vcur, kq_mask, 1.0f / sqrtf((float) tfm_d_head), il);
    cur = ggml_mul(ctx0, cur, layer.ls_1_w);
    cur = ggml_add(ctx0, cur, inp);

    inp = cur;
    cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, tfm_eps, il);
    cur = build_ffn(cur, layer.ff_up_w, nullptr, nullptr, nullptr, layer.ff_down_w, nullptr, tfm_ffn_type, il);
    cur = ggml_mul(ctx0, cur, layer.ls_2_w);
    cur = ggml_add(ctx0, cur, inp);

    return cur;
}

ggml_tensor * mtmd_graph_pockettts_spkenc::encode(mtmd_graph_pockettts_seanet & seanet, ggml_tensor * x, const std::vector<mtmd_layer> & layers) {
    ggml_tensor * cur = seanet.encode(x);
    cb(cur, "mimi_enc", -1);

    // [T, 512] -> transformer works on [512, T]
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    const int64_t n_pos = cur->ne[1];
    const bool fixed_cache = streaming && hparams.gen_model_variant == "mimi";
    if (fixed_cache) {
        ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_pos);
        ggml_set_input(positions);
        ggml_set_name(positions, "inp_cache_pos");
        seanet.state_in["cache_pos"] = positions;
    }

    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    // the mimi transformer is causal with a sliding window, see _build_attention_mask()
    const int64_t n_kv = fixed_cache ? hparams.mimi_tfm_context :
        n_pos + (streaming ? hparams.mimi_tfm_context - 1 : 0);
    ggml_tensor * kq_mask = build_inp_attn_mask(n_kv, n_pos);

    for (size_t il = 0; il < layers.size(); il++) {
        cur = tfm_layer_forward(cur, layers[il], inp_pos, kq_mask, (int) il, streaming ? &seanet : nullptr);
    }
    if (streaming) {
        seanet.state_out.push_back({"tfm_pos", ggml_scale_bias(ctx0, seanet.state_in.at("tfm_pos"), 1.0f, (float) n_pos)});
    }
    cb(cur, "mimi_enc_tfm", -1);

    // downsample to the model frame rate, [512, T] -> [T, 512] -> [T / mimi_downsample, C]
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    cur = seanet.conv1d(cur, model.downsample_w, nullptr, hparams.mimi_downsample, 1, true,
                        streaming ? "down" : "");
    cb(cur, "mimi_downsample", -1);

    return cur;
}

ggml_cgraph * mtmd_graph_pockettts_spkenc::build() {
    // the preprocessor hands over the waveform as a single-row "mel", already [n_samples, 1]
    ggml_tensor * inp_raw = build_inp_raw(1);
    ggml_tensor * cur = ggml_reshape_2d(ctx0, inp_raw, inp_raw->ne[0], inp_raw->ne[1]);

    const bool continuous = hparams.gen_model_variant == "mimi";
    if (continuous || model.mimi_quant_in[0]) {
        tfm_rope_type = GGML_ROPE_TYPE_NEOX;
    }
    if (continuous) {
        tfm_ffn_type = FFN_GELU_ERF;
    }

    mtmd_graph_pockettts_seanet seanet(*this);
    if (streaming) {
        seanet.state_inputs(list_mimi_encoder_state_slots(hparams, model));
    }
    cur = encode(seanet, cur, model.layers);

    // voice latent -> backbone embd
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    if (!continuous) {
        if (model.spk_proj_w) {
            cur = build_mm(model.spk_proj_w, cur);
        } else {
            // Split RVQ: the semantic and acoustic quantizers start from the same latent.
            ggml_tensor * sum = nullptr;
            for (int group = 0; group < 2; ++group) {
                auto * residual = build_mm(model.mimi_quant_in[group], cur);
                ggml_tensor * tables = model.mimi_quant_cb[group];
                ggml_tensor * norms = model.mimi_quant_norm[group];
                for (int64_t i = 0; i < tables->ne[2]; ++i) {
                    auto * table = ggml_view_2d(ctx0, tables, tables->ne[0], tables->ne[1],
                                               tables->nb[1], i * tables->nb[2]);
                    auto * norm = ggml_view_1d(ctx0, norms, norms->ne[0], i * norms->nb[1]);
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
    }
    cb(cur, "spk_proj", -1);

    if (streaming) {
        seanet.state_outputs(gf);
    }
    ggml_build_forward_expand(gf, cur);
    if (continuous) {
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if (node->op == GGML_OP_MUL_MAT) {
                ggml_prec_set_src(node, GGML_PREC_F32, 1);
                ggml_prec_set_acc(node, GGML_PREC_F32);
            }
        }
    }
    return gf;
}

std::vector<c2w_state_slot> list_mimi_encoder_state_slots(const mtmd_hparams & hp, const mtmd_model & model) {
    const auto & enc = model.seanet;
    std::vector<c2w_state_slot> slots{{"tfm_pos", 1, 1}};
    for (size_t i = 0; i < model.layers.size(); ++i) {
        const auto n = model.layers[i].q_w->ne[1];
        slots.push_back({"tfm_k_" + std::to_string(i), n, hp.mimi_tfm_context - (hp.gen_model_variant == "mimi" ? 0 : 1)});
        slots.push_back({"tfm_v_" + std::to_string(i), n, hp.mimi_tfm_context - (hp.gen_model_variant == "mimi" ? 0 : 1)});
    }
    slots.push_back({"enc_in", enc.conv_in_w->ne[0] - 1, enc.conv_in_w->ne[1]});
    for (int i = 0; i < hp.seanet_n_stage; ++i) {
        const auto & stage = enc.stages[i];
        slots.push_back({"enc_res_" + std::to_string(i), stage.res_conv1_w->ne[0] - 1, stage.res_conv1_w->ne[1]});
        slots.push_back({"enc_down_" + std::to_string(i), stage.scale_conv_w->ne[0] - hp.seanet_ratios[i], stage.scale_conv_w->ne[1]});
    }
    slots.push_back({"enc_out", enc.conv_out_w->ne[0] - 1, enc.conv_out_w->ne[1]});
    slots.push_back({"down", model.downsample_w->ne[0] - hp.mimi_downsample, model.downsample_w->ne[1]});
    return slots;
}
