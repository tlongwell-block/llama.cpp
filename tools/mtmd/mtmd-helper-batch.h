#pragma once
#include "llama.h"
#include "mtmd.h"

#include <vector>

// helper struct to make working with embd batch easier
// note: this will be removed after llama_batch_ext refactoring
struct decode_embd_batch {
    int n_pos_per_embd;
    int n_mmproj_embd;
    std::vector<llama_pos>      pos;
    std::vector<llama_pos>      pos_view; // used by mrope
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t>         logits;
    llama_batch batch;
    decode_embd_batch(float * embd, int32_t n_tokens, int n_pos_per_embd, int n_mmproj_embd) : n_pos_per_embd(n_pos_per_embd), n_mmproj_embd(n_mmproj_embd) {
        GGML_ASSERT(n_tokens > 0 && n_pos_per_embd > 0 && n_mmproj_embd > 0);
        pos     .resize((size_t) n_tokens * (size_t) n_pos_per_embd);
        n_seq_id.resize(n_tokens);
        seq_ids .resize(n_tokens + 1);
        logits  .resize(n_tokens);
        seq_id_0.resize(1);
        seq_ids [n_tokens] = nullptr;
        batch = {
            /*n_tokens       =*/ n_tokens,
            /*tokens         =*/ nullptr,
            /*embd           =*/ embd,
            /*pos            =*/ pos.data(),
            /*n_seq_id       =*/ n_seq_id.data(),
            /*seq_id         =*/ seq_ids.data(),
            /*logits         =*/ logits.data(),
        };
    }

    void set_position_normal(llama_pos pos_0, llama_seq_id seq_id) {
        seq_id_0[0] = seq_id;
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.pos     [i] = pos_0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }

    // M-RoPE for image
    void set_position_mrope_2d(const std::vector<mtmd_decoder_pos> & rel_pos, llama_seq_id seq_id) {
        GGML_ASSERT(n_pos_per_embd == 4);
        GGML_ASSERT(!rel_pos.empty() && (int32_t)rel_pos.size() == batch.n_tokens);
        seq_id_0[0] = seq_id;
        for (int32_t i = 0; i < batch.n_tokens; i++) {
            const size_t idx = (size_t) i;
            const size_t n_tokens = (size_t) batch.n_tokens;
            pos[idx                    ] = rel_pos[i].t;
            pos[idx + n_tokens         ] = rel_pos[i].y;
            pos[idx + n_tokens * 2     ] = rel_pos[i].x;
            pos[idx + n_tokens * 3     ] = rel_pos[i].z;
        }
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }

    // M-RoPE for audio
    void set_position_mrope_1d(llama_pos pos_0, llama_seq_id seq_id) {
        GGML_ASSERT(n_pos_per_embd == 4);
        seq_id_0[0] = seq_id;
        for (int i = 0; i < batch.n_tokens; i++) {
            const size_t idx = (size_t) i;
            const size_t n_tokens = (size_t) batch.n_tokens;
            pos[idx                    ] = pos_0 + i;
            pos[idx + n_tokens         ] = pos_0 + i;
            pos[idx + n_tokens * 2     ] = pos_0 + i;
            pos[idx + n_tokens * 3     ] = pos_0 + i;
        }
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }

    llama_batch get_view(int offset, int n_tokens) {
        GGML_ASSERT(offset >= 0 && n_tokens > 0 && offset + n_tokens <= batch.n_tokens);
        llama_pos * pos_ptr;
        pos_view.clear();
        pos_view.reserve((size_t) n_tokens * (size_t) n_pos_per_embd);
        if (n_pos_per_embd > 1) {
            // mrope
            // for example, with layout of src: 1234...1234...1234...1234...
            //       offset 2 will give us dst: 34...34...34...34...
            for (int i = 0; i < n_pos_per_embd; i++) {
                // assume n_tokens is less than or equal to batch.n_tokens
                // batch.n_tokens is number of **total** tokens
                // n_tokens is number of viewed token
                size_t src_idx = (size_t) i * (size_t) batch.n_tokens + (size_t) offset;
                pos_view.insert(pos_view.end(),
                    pos.data() + src_idx,
                    pos.data() + src_idx + n_tokens);
            }
            pos_ptr = pos_view.data();
        } else {
            // normal
            pos_ptr = pos.data() + offset;
        }
        return {
            /*n_tokens       =*/ n_tokens,
            /*tokens         =*/ nullptr,
            /*embd           =*/ batch.embd     + offset * n_mmproj_embd,
            /*pos            =*/ pos_ptr,
            /*n_seq_id       =*/ batch.n_seq_id + offset,
            /*seq_id         =*/ batch.seq_id   + offset,
            /*logits         =*/ batch.logits   + offset,
        };
    }
};
