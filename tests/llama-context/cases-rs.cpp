// recurrent state cases

#include "test-case.h"

#include "test-log.h"
#include "test-util.h"

#include "common.h"

#include "../../src/llama-ext.h" // llama_arch_from_string

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

// roll back multiple sequences, then replay them in a single batch whose per-seq token count exceeds n_ubatch: each
// seq's replay spans several ubatches while its rollback restore is still pending. Compared against a reference context
// that never advanced past the rollback point and decodes the identical replay batch. Only the first replayed token of
// each seq is compared.
struct case_multi_seq_split_replay : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "replay"; }

    const char * description() const override {
        return "replay rolled-back sequences in a batch wider than n_ubatch, against a reference context";
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr uint32_t  n_seqs     = 2;
        constexpr uint32_t  n_ubatch   = 16;
        constexpr uint32_t  n_prompt   = 20;
        constexpr uint32_t  n_rollback = 3;
        constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
        constexpr llama_pos p0         = n_prompt - n_rollback;

        // both contexts have the same layout, only what they decode differs
        const auto make_ctx_multi = [&]() {
            ctx_opts opts;
            opts.n_ctx      = 512;
            opts.n_batch    = 256;
            opts.n_ubatch   = n_ubatch;
            opts.n_seq_max  = n_seqs;
            opts.n_rs_seq   = 8;
            opts.kv_unified = 0;
            return mr.ctx(opts);
        };

        llama_context_ptr ctx_roll = make_ctx_multi();
        llama_context_ptr ctx_ref  = make_ctx_multi();
        if (!ctx_roll || !ctx_ref) {
            return case_result::fail();
        }

        if (llama_n_rs_seq(ctx_roll.get()) < n_rollback) {
            return case_result::skip("n_rs_seq is too small");
        }

        const auto tok = [&](uint32_t seq, llama_pos pos) {
            return gen_token(n_vocab, (llama_seq_id) seq, pos);
        };

        bool ok = true;

        // keep the anchor at p0 - 1 when rolling back the tail
        for (uint32_t s = 0; s < n_seqs && ok; ++s) {
            ok = ok && decode_gen(ctx_roll.get(), n_vocab, (llama_seq_id) s, 0, (llama_pos) p0 - 1);
            ok = ok && decode_gen(ctx_ref.get(),  n_vocab, (llama_seq_id) s, 0, (llama_pos) p0 - 1);

            ok = ok && decode_gen(ctx_roll.get(), n_vocab, (llama_seq_id) s, (llama_pos) p0 - 1, (llama_pos) n_prompt);
            ok = ok && decode_gen(ctx_ref.get(),  n_vocab, (llama_seq_id) s, (llama_pos) p0 - 1, (llama_pos) p0);

            ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0, -1);

            // a second partial removal while one is pending must be refused
            ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0 - 1, -1);
        }
        CTX_REQUIRE(ok, "the multi-seq prefill / rollback failed");

        // all seqs replay in a single batch
        const auto decode_replay = [&](llama_context * ctx) {
            common_batch batch(ctx);
            for (uint32_t s = 0; s < n_seqs; ++s) {
                for (uint32_t i = 0; i < n_replay; ++i) {
                    const llama_pos pos = p0 + (llama_pos) i;
                    batch.add(tok(s, pos), pos, (llama_seq_id) s, true);
                }
            }
            return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
        };

        ok = decode_replay(ctx_roll.get()) && decode_replay(ctx_ref.get());
        CTX_REQUIRE(ok, "the multi-seq replay decode failed");

        // the first replayed token of each seq reads the state right after the rollback, so its
        //   logits should match; the later tokens are not compared because small state
        //   differences at the start grow with each replayed token
        // random dummy models can still drift up to ~1.7e-5, so the bound is 1e-4
        constexpr float nmse_eps = 1e-4f;

        logits_diff diff;
        for (uint32_t s = 0; s < n_seqs; ++s) {
            const uint32_t i = s*n_replay; // first replayed token of this seq

            const float * l_roll = llama_get_logits_ith(ctx_roll.get(), i);
            const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  i);
            if (l_roll == nullptr || l_ref == nullptr) {
                TLOG_ERR("%s: missing multi-seq logits at index %u\n", __func__, i);
                return case_result::fail();
            }

            diff.add(l_roll, l_ref, n_vocab);
        }

        CTX_REQUIRE(diff.nmse() <= nmse_eps,
                "multi-seq split replay logits mismatch (max diff " + std::to_string(diff.diff_max) +
                ", nmse " + std::to_string(diff.nmse()) + ", first difference at index " + std::to_string(diff.i_first) + ")");

        TLOG_INF("%s: multi-seq split replay matched (max diff %g, nmse %g)\n", __func__, (double) diff.diff_max, diff.nmse());

        // the replay above can leave the 2 contexts in slightly different states, and the tail
        //   below assumes they start from the same state - so copy the roll state into ref
        for (uint32_t s = 0; s < n_seqs && ok; ++s) {
            std::vector<uint8_t> state;

            ok = ok && get_seq_state(ctx_roll.get(), (llama_seq_id) s, 0, state);
            ok = ok && set_seq_state(ctx_ref.get(),  (llama_seq_id) s, 0, state);

            // do a second round of state save/load to negate any SWA-positioning effects
            if (ok) {
                std::vector<uint8_t> restored;
                ok = ok && get_seq_state(ctx_ref.get(), (llama_seq_id) s, 0, restored);

                // what was loaded into ref must come back out bit-identical
                CTX_REQUIRE(ok && state == restored,
                        "the seq " + std::to_string(s) + " state did not round-trip bit-exact (sizes " +
                        std::to_string(state.size()) + "/" + std::to_string(restored.size()) + ")");

                ok = ok && set_seq_state(ctx_roll.get(), (llama_seq_id) s, 0, restored);
            }
        }
        CTX_REQUIRE(ok, "the multi-seq state copy failed");

        // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
        //   in ctx_ref only, then compare identical seq-1-only continuations
        constexpr uint32_t n_tail = 4;

        {
            common_batch batch_tail(ctx_ref.get());
            for (uint32_t i = 0; i < n_tail; ++i) {
                const llama_pos pos = p0 + (llama_pos) (n_replay + i);
                batch_tail.add(tok(0, pos + 7), pos, 0, false);
            }
            ok = llama_process(ctx_ref.get(), LLAMA_PROCESS_TYPE_DECODE, batch_tail.get()) == 0;
        }

        logits_diff diff_tail;
        for (uint32_t i = 0; i < n_tail && ok; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            ok = decode_one(ctx_roll.get(), tok(1, pos), pos, 1) &&
                 decode_one(ctx_ref.get(),  tok(1, pos), pos, 1);
            if (!ok) {
                break;
            }

            const float * l_roll = llama_get_logits_ith(ctx_roll.get(), 0);
            const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  0);
            if (l_roll == nullptr || l_ref == nullptr) {
                TLOG_ERR("%s: missing tail logits at step %u\n", __func__, i);
                return case_result::fail();
            }

            diff_tail.add(l_roll, l_ref, n_vocab);
        }

        CTX_REQUIRE(ok && diff_tail.nmse() <= 0.0f,
                "a seq-1-only decode leaked the state of seq 0 (nmse " + std::to_string(diff_tail.nmse()) + ")");

        TLOG_INF("%s: seq-1-only decode is independent of seq 0 (max diff %g, nmse %g)\n",
                __func__, (double) diff_tail.diff_max, diff_tail.nmse());

        return case_result::pass();
    }
};

// save a rolled-back single-seq state, restore it into a fresh and into a dirty context, and
// verify exact logit matches on replay
struct case_rollback : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "rollback"; }

    const char * description() const override {
        return "a rolled-back recurrent state restores into fresh and dirty contexts with identical replay";
    }

    case_result run(model_run & mr) const override {
        const llama_vocab * vocab   = llama_model_get_vocab(mr.model);
        const int           n_vocab = llama_vocab_n_tokens(vocab);

        const auto make_rs_ctx = [&]() {
            ctx_opts opts;
            opts.n_seq_max = 1;
            opts.n_rs_seq  = 8;
            return mr.ctx(opts);
        };

        llama_context_ptr ctx_src = make_rs_ctx();
        llama_context_ptr ctx_dst = make_rs_ctx();
        if (!ctx_src || !ctx_dst) {
            return case_result::fail();
        }

        const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src.get());
        if (n_rs_seq == 0) {
            return case_result::skip("n_rs_seq is disabled");
        }

        constexpr uint32_t n_rollback = 3;
        if (n_rs_seq < n_rollback) {
            return case_result::skip("n_rs_seq is too small");
        }

        llama_tokens tokens;
        if (!mr.has_vocab()) {
            tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        } else {
            tokens = common_tokenize(ctx_src.get(), "The quick brown fox jumps over the lazy dog", true);
        }
        if (tokens.empty()) {
            TLOG_ERR("%s: not enough prompt tokens\n", __func__);
            return case_result::fail();
        }
        tokens.resize(n_rs_seq + 1, tokens.back());

        const llama_pos rollback_pos = (llama_pos) tokens.size() - n_rollback;

        // Decode the full prompt on the source, then roll back three positions.
        // Replaying them crosses DSV4's ratio-4 compressor boundary.
        // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
        if (!decode_tokens(ctx_src.get(), tokens)) {
            TLOG_ERR("%s: failed to decode the prompt\n", __func__);
            return case_result::fail();
        }
        if (!llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, rollback_pos, -1)) {
            TLOG_ERR("%s: rollback failed\n", __func__);
            return case_result::fail();
        }

        // Save the rolled-back state and restore it into a fresh context.
        common_prompt_checkpoint ckpt;
        ckpt.update_tgt(ctx_src.get(), 0, 0);
        ckpt.load_tgt(ctx_dst.get(), 0, 0);

        // exact match - the same context configuration replaying the same tokens
        constexpr float nmse_eps = 0.0;

        std::vector<std::vector<float>> logits_src_replay(n_rollback);
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src.get(), tokens[pos], pos) ||
                !decode_one(ctx_dst.get(), tokens[pos], pos)) {
                TLOG_ERR("%s: replay failed at position %d\n", __func__, pos);
                return case_result::fail();
            }

            const float * logits_src = llama_get_logits_ith(ctx_src.get(), 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst.get(), 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                TLOG_ERR("%s: missing replay logits at position %d\n", __func__, pos);
                return case_result::fail();
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);

            const logits_diff diff = compare_logits(logits_src, logits_dst, n_vocab);
            CTX_REQUIRE(diff.nmse() <= nmse_eps,
                    "replay logits mismatch at position " + std::to_string(pos) + ", nmse " + std::to_string(diff.nmse()));
        }

        // the replay ran in single-token ubatches, which hold no snapshot that deep
        CTX_REQUIRE(!llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, rollback_pos, -1) &&
                    !llama_memory_seq_rm(llama_get_memory(ctx_dst.get()), 0, rollback_pos, -1),
                "a rollback past the snapshots of single-token ubatches was accepted");

        // The same rollback through a partial-only checkpoint, as a session branch saves it.
        // Rebuild with one prompt batch on both contexts: the snapshots do not span separate decodes.
        llama_memory_clear(llama_get_memory(ctx_src.get()), true);
        llama_memory_clear(llama_get_memory(ctx_dst.get()), true);
        if (!decode_tokens(ctx_src.get(), tokens) || !decode_tokens(ctx_dst.get(), tokens)) {
            TLOG_ERR("%s: failed to rebuild the prompt\n", __func__);
            return case_result::fail();
        }
        if (!llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, rollback_pos, -1) ||
            !llama_memory_seq_rm(llama_get_memory(ctx_dst.get()), 0, rollback_pos, -1)) {
            TLOG_ERR("%s: partial rollback failed\n", __func__);
            return case_result::fail();
        }

        common_prompt_checkpoint ckpt_partial;
        ckpt_partial.update_tgt(ctx_src.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        ckpt_partial.load_tgt(ctx_dst.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src.get(), tokens[pos], pos) ||
                !decode_one(ctx_dst.get(), tokens[pos], pos)) {
                TLOG_ERR("%s: partial replay failed at position %d\n", __func__, pos);
                return case_result::fail();
            }

            const float * logits_src = llama_get_logits_ith(ctx_src.get(), 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst.get(), 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                TLOG_ERR("%s: missing partial replay logits at position %d\n", __func__, pos);
                return case_result::fail();
            }

            const logits_diff diff = compare_logits(logits_src, logits_dst, n_vocab);
            CTX_REQUIRE(diff.nmse() <= nmse_eps,
                    "partial replay logits mismatch at position " + std::to_string(pos) + ", nmse " + std::to_string(diff.nmse()));
        }

        // Repeat the load into a context that already has its own rollback state:
        // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
        // non-zero at load time. The restore must wipe that state and still match.
        llama_context_ptr ctx_dirty = make_rs_ctx();
        if (!ctx_dirty) {
            return case_result::fail();
        }

        llama_tokens noise = tokens;
        for (auto & tok : noise) {
            tok = (tok + 1) % n_vocab;
            if (tok < 0) {
                tok = 0;
            }
        }
        if (!decode_tokens(ctx_dirty.get(), noise)) {
            TLOG_ERR("%s: failed to decode the dirty prompt\n", __func__);
            return case_result::fail();
        }
        if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty.get()), 0, rollback_pos, -1)) {
            TLOG_ERR("%s: dirty rollback failed\n", __func__);
            return case_result::fail();
        }

        ckpt.load_tgt(ctx_dirty.get(), 0, 0);

        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_dirty.get(), tokens[pos], pos)) {
                TLOG_ERR("%s: dirty replay failed at position %d\n", __func__, pos);
                return case_result::fail();
            }

            const float * logits_dirty = llama_get_logits_ith(ctx_dirty.get(), 0);
            if (logits_dirty == nullptr) {
                TLOG_ERR("%s: missing dirty logits at position %d\n", __func__, pos);
                return case_result::fail();
            }

            const logits_diff diff = compare_logits(logits_src_replay[i].data(), logits_dirty, n_vocab);
            CTX_REQUIRE(diff.nmse() <= nmse_eps,
                    "dirty-ctx logits mismatch at position " + std::to_string(pos) + ", nmse " + std::to_string(diff.nmse()));
        }

        TLOG_INF("%s: recurrent rollback checkpoint restored successfully\n", __func__);

        return case_result::pass();
    }
};

// after a ubatch of n tokens, probe rollbacks from deepest to shallowest: every rollback past the
// snapshots that ubatch guarantees is refused and the deepest guaranteed one is accepted
struct case_rollback_bound : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "rollback_bound"; }

    const char * description() const override {
        return "seq_rm accepts exactly the rollbacks whose snapshots the last ubatch wrote";
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr llama_pos n_prefix = 16;

        char arch_name[64] = {};
        llama_model_meta_val_str(mr.model, "general.architecture", arch_name, sizeof(arch_name));
        const llm_arch arch = llama_arch_from_string(arch_name);

        // the DSV4 cache rebuilds snapshot d from the state before the ubatch, so d <= n is exact;
        // the recurrent memory gets snapshots 0..n-1 from ggml_ssm_scan, only for n <= n_rs_seq + 1
        const bool from_ubatch_start = arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41;

        const auto make_ctx = [&]() {
            ctx_opts opts;
            opts.n_batch   = 64;
            opts.n_ubatch  = 64;
            opts.n_seq_max = 1;
            opts.n_rs_seq  = 8;
            return mr.ctx(opts);
        };

        llama_context_ptr ctx = make_ctx();
        if (!ctx) {
            return case_result::fail();
        }

        const uint32_t n_rs_seq = llama_n_rs_seq(ctx.get());
        if (n_rs_seq == 0) {
            return case_result::skip("n_rs_seq is disabled");
        }

        for (const uint32_t n : { 1u, 2u, n_rs_seq + 1, n_rs_seq + 2 }) {
            const uint32_t r_max = from_ubatch_start ? std::min(n, n_rs_seq) : (n <= n_rs_seq + 1 ? n - 1 : 0);

            ctx = make_ctx();
            if (!ctx) {
                return case_result::fail();
            }

            const llama_pos pos_end = n_prefix + (llama_pos) n;

            CTX_REQUIRE(decode_gen(ctx.get(), n_vocab, 0, 0, n_prefix) &&
                        decode_gen(ctx.get(), n_vocab, 0, n_prefix, pos_end),
                    "the prefix / ubatch decode failed");

            for (uint32_t r = n_rs_seq + 1; r > r_max; --r) {
                CTX_REQUIRE(!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, pos_end - (llama_pos) r, -1),
                        "rollback " + std::to_string(r) + " after a " + std::to_string(n) + "-token ubatch was accepted");
            }

            if (r_max > 0) {
                CTX_REQUIRE(llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, pos_end - (llama_pos) r_max, -1),
                        "rollback " + std::to_string(r_max) + " after a " + std::to_string(n) + "-token ubatch was refused");
            }
        }

        TLOG_INF("%s: rollback bounds hold for ubatches of 1, 2, %u and %u tokens\n", __func__, n_rs_seq + 1, n_rs_seq + 2);

        return case_result::pass();
    }
};

// decode a prompt into seq 0, share its cells with a second sequence, then keep decoding both
struct case_shared_seq_reserve : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "shared"; }

    const char * description() const override {
        return "sequences sharing cells keep decoding correctly after a seq_cp";
    }

    std::string skip_reason(model_run & mr) const override {
        if (arch_reserves_single_seq(mr.model)) {
            return "the shared-seq batch is a multi-seq graph";
        }

        return ctx_test::skip_reason(mr);
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr uint32_t n_seqs     = 2;
        constexpr uint32_t n_prompt   = 128;
        constexpr uint32_t n_continue = 32;

        ctx_opts opts;
        opts.n_ctx      = 256;
        opts.n_batch    = 256;
        opts.n_ubatch   = 64;
        opts.n_seq_max  = n_seqs;
        opts.kv_unified = 1; // only a unified cache shares cells on seq_cp

        auto ctx = mr.ctx(opts);
        if (!ctx) {
            return case_result::fail();
        }

        if (!decode_gen(ctx.get(), n_vocab, 0, 0, (llama_pos) n_prompt)) {
            TLOG_ERR("%s: prompt decode failed\n", __func__);
            return case_result::fail();
        }

        // this is what llama-batched-bench does for -pps
        llama_memory_seq_cp(llama_get_memory(ctx.get()), 0, 1, -1, -1);

        for (uint32_t i = 0; i < n_continue; ++i) {
            const llama_pos pos = (llama_pos) (n_prompt + i);

            common_batch batch(ctx.get());
            for (uint32_t s = 0; s < n_seqs; ++s) {
                batch.add(gen_token(n_vocab, (llama_seq_id) s, pos), pos, (llama_seq_id) s, true);
            }
            CTX_REQUIRE(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0,
                    "shared-seq decode failed at step " + std::to_string(i));

            for (uint32_t s = 0; s < n_seqs; ++s) {
                const float * logits = llama_get_logits_ith(ctx.get(), (int) s);
                if (logits == nullptr) {
                    TLOG_ERR("%s: missing shared-seq logits at index %u\n", __func__, s);
                    return case_result::fail();
                }

                bool finite = true;
                for (int t_ = 0; t_ < n_vocab; ++t_) {
                    finite = finite && std::isfinite(logits[t_]);
                }

                CTX_REQUIRE(finite, "non-finite shared-seq logit at step " + std::to_string(i) + ", seq " + std::to_string(s));
            }
        }

        TLOG_INF("%s: shared-seq decode succeeded (%u tokens after seq_cp)\n", __func__, n_continue*n_seqs);

        return case_result::pass();
    }
};

// capture the layer inputs, intermediate and final, of a multi-seq batch with sparse outputs: the batch interleaves
// the seqs and is wider than n_ubatch, so the ubatches extract the rows out of batch order. Every captured row is
// compared against single-seq reference runs, and the logits against a context with the taps off.
struct case_layer_inp : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "layer_inp"; }

    const char * description() const override {
        return "layer inputs of a split multi-seq batch match single-seq runs, and the taps leave the logits unchanged";
    }

    std::string skip_reason(model_run & mr) const override {
        char arch_name[64] = {};
        llama_model_meta_val_str(mr.model, "general.architecture", arch_name, sizeof(arch_name));
        const llm_arch arch = llama_arch_from_string(arch_name);
        if (arch != LLM_ARCH_QWEN35 && arch != LLM_ARCH_QWEN4EXP) {
            return "requires the final-state tap of Qwen3.5 or Qwen4exp";
        }

        return ctx_test::skip_reason(mr);
    }

    case_result run(model_run & mr) const override {
        const int      n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));
        const uint32_t n_layer = (uint32_t) llama_model_n_layer(mr.model);
        const size_t   n_embd  = (size_t) llama_model_n_embd(mr.model);
        const size_t   n_final = (size_t) llama_model_n_embd_layer_inp(mr.model);

        char arch_name[64] = {};
        llama_model_meta_val_str(mr.model, "general.architecture", arch_name, sizeof(arch_name));
        if (llama_arch_from_string(arch_name) == LLM_ARCH_QWEN4EXP) {
            CTX_REQUIRE(n_final > n_embd && n_final == (size_t) llama_model_n_embd_out(mr.model),
                    "the final tap is not the hyper-connection stream (" + std::to_string(n_final) + " floats)");
        } else {
            CTX_REQUIRE(n_final == n_embd, "the final tap is not n_embd wide (" + std::to_string(n_final) + " floats)");
        }

        constexpr uint32_t  n_seqs   = 3;
        constexpr llama_pos n_prefix = 5;
        constexpr uint32_t  n_new[n_seqs] = { 11, 6, 14 }; // uneven, so that the seqs leave the ubatches at different times

        const auto width     = [&](uint32_t il) { return il == n_layer ? n_final : n_embd; };
        const auto is_output = [&](uint32_t s, uint32_t i) { return i % 3 == 1 || i + 1 == n_new[s]; };

        // nextn off, unmasked (MTP target) and masked: each crops the output rows at a different point of the graph
        bool nextn        = false;
        bool nextn_masked = false;

        const auto make = [&](bool taps) {
            ctx_opts opts;
            opts.n_ctx      = 256;
            opts.n_batch    = 64;
            opts.n_ubatch   = 8;
            opts.n_seq_max  = n_seqs;
            opts.kv_unified = 0;

            auto ctx = mr.ctx(opts);
            if (ctx) {
                llama_set_embeddings_nextn(ctx.get(), nextn, nextn_masked);
            }
            for (uint32_t il = 0; ctx && taps && il <= n_layer; ++il) {
                llama_set_embeddings_layer_inp(ctx.get(), il, true);
            }
            return ctx;
        };

        // the captures of one run, rows[k] is the (seq, new token) of batch index k
        struct capture {
            std::vector<std::pair<uint32_t, uint32_t>> rows;
            std::vector<std::vector<float>>            layer; // per layer id, rows.size()*width floats
        };

        // decode the prefixes, then the new tokens of seqs in a single batch, round-robin over the seqs
        const auto decode = [&](llama_context * ctx, const std::vector<uint32_t> & seqs, bool taps, capture & out) {
            for (uint32_t s : seqs) {
                if (!decode_gen(ctx, n_vocab, (llama_seq_id) s, 0, n_prefix)) {
                    return false;
                }
            }

            common_batch batch(ctx);
            for (uint32_t i = 0; i < *std::max_element(n_new, n_new + n_seqs); ++i) {
                for (uint32_t s : seqs) {
                    if (i < n_new[s]) {
                        const llama_pos pos = n_prefix + (llama_pos) i;
                        batch.add(gen_token(n_vocab, (llama_seq_id) s, pos), pos, (llama_seq_id) s, is_output(s, i));
                        out.rows.emplace_back(s, i);
                    }
                }
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                return false;
            }

            for (uint32_t il = 0; taps && il <= n_layer; ++il) {
                const float * data = llama_get_embeddings_layer_inp(ctx, il);
                if (data == nullptr) {
                    return false;
                }
                out.layer.emplace_back(data, data + out.rows.size()*width(il));
            }
            return true;
        };

        std::vector<uint32_t> all(n_seqs);
        for (uint32_t s = 0; s < n_seqs; ++s) {
            all[s] = s;
        }

        for (const auto & mode : { std::make_pair(false, false), std::make_pair(true, false), std::make_pair(true, true) }) {
        nextn        = mode.first;
        nextn_masked = mode.second;
        const std::string tag = !nextn ? "nextn off" : nextn_masked ? "nextn masked" : "nextn unmasked";

            auto ctx_tap = make(true);
            auto ctx_off = make(false);
            CTX_REQUIRE(ctx_tap && ctx_off, "failed to create the contexts");

            capture cap;
            capture cap_off;
            CTX_REQUIRE(decode(ctx_tap.get(), all, true, cap), tag + ": " + "the multi-seq decode with taps failed");
            CTX_REQUIRE(decode(ctx_off.get(), all, false, cap_off), tag + ": " + "the multi-seq decode without taps failed");
            CTX_REQUIRE(cap.rows.size() > 2*llama_n_ubatch(ctx_tap.get()), tag + ": " + "the batch does not span several ubatches");

            // the multi-seq batch index of each (seq, new token)
            std::vector<std::vector<int32_t>> idx(n_seqs);
            for (uint32_t s = 0; s < n_seqs; ++s) {
                idx[s].assign(n_new[s], -1);
            }
            for (size_t k = 0; k < cap.rows.size(); ++k) {
                idx[cap.rows[k].first][cap.rows[k].second] = (int32_t) k;
            }

            // rows that differ only in batch order must be told apart
            CTX_REQUIRE(compare_logits(cap.layer[0].data(), cap.layer[0].data() + n_embd, n_embd).nmse() > 0.0,
                    tag + ": " + "the first two batch rows have identical inputs");

            // unmasked, the nextn rows are the final state of every token too, read through a separate buffer
            if (nextn && !nextn_masked) {
                CTX_REQUIRE(n_final == (size_t) llama_model_n_embd_out(mr.model), tag + ": the nextn rows have another width");
                logits_diff diff_nextn;
                for (size_t k = 0; k < cap.rows.size(); ++k) {
                    const float * h = llama_get_embeddings_nextn_ith(ctx_tap.get(), (int32_t) k);
                    CTX_REQUIRE(h, tag + ": missing nextn row " + std::to_string(k));
                    diff_nextn.add(h, cap.layer[n_layer].data() + k*n_final, n_final);
                }
                TLOG_INF("%s: %s: final tap against the nextn rows: max diff %g\n", __func__, tag.c_str(), (double) diff_nextn.diff_max);
                CTX_REQUIRE(diff_nextn.diff_max == 0.0f, tag + ": the final tap differs from the nextn rows (first difference at " +
                        std::to_string(diff_nextn.i_first) + ")");
            }

            logits_diff diff_off;
            std::vector<logits_diff> diff_layer(n_layer + 1);
            logits_diff diff_ref;

            for (size_t k = 0; k < cap.rows.size(); ++k) {
                if (is_output(cap.rows[k].first, cap.rows[k].second)) {
                    const float * l_tap = llama_get_logits_ith(ctx_tap.get(), (int32_t) k);
                    const float * l_off = llama_get_logits_ith(ctx_off.get(), (int32_t) k);
                    CTX_REQUIRE(l_tap && l_off, "missing multi-seq logits at batch index " + std::to_string(k));
                    diff_off.add(l_off, l_tap, n_vocab);
                }
            }

            for (uint32_t s = 0; s < n_seqs; ++s) {
                auto ctx_ref = make(true);
                CTX_REQUIRE(ctx_ref, "failed to create the reference context");

                capture ref;
                CTX_REQUIRE(decode(ctx_ref.get(), { s }, true, ref), "the single-seq decode of seq " + std::to_string(s) + " failed");

                for (uint32_t i = 0; i < n_new[s]; ++i) {
                    const int32_t k = idx[s][i];
                    for (uint32_t il = 0; il <= n_layer; ++il) {
                        const size_t w = width(il);
                        diff_layer[il].add(ref.layer[il].data() + i*w, cap.layer[il].data() + (size_t) k*w, w);
                    }
                    if (is_output(s, i)) {
                        const float * l_ref = llama_get_logits_ith(ctx_ref.get(), (int32_t) i);
                        const float * l_tap = llama_get_logits_ith(ctx_tap.get(), k);
                        CTX_REQUIRE(l_ref && l_tap, "missing reference logits of seq " + std::to_string(s));
                        diff_ref.add(l_ref, l_tap, n_vocab);
                    }
                }
            }

            for (uint32_t il = 0; il <= n_layer; ++il) {
                TLOG_INF("%s: %s: layer %u (%zu floats) against single-seq runs: max diff %g, nmse %g\n",
                        __func__, tag.c_str(), il, width(il), (double) diff_layer[il].diff_max, diff_layer[il].nmse());
                CTX_REQUIRE(diff_layer[il].nmse() <= NMSE_THRESHOLD,
                        tag + ": layer " + std::to_string(il) + " inputs differ from the single-seq runs (nmse " +
                        std::to_string(diff_layer[il].nmse()) + ", first difference at " + std::to_string(diff_layer[il].i_first) + ")");
            }

            TLOG_INF("%s: %s: logits against single-seq runs: max diff %g, nmse %g; against taps off: max diff %g, nmse %g\n",
                    __func__, tag.c_str(), (double) diff_ref.diff_max, diff_ref.nmse(), (double) diff_off.diff_max, diff_off.nmse());
            CTX_REQUIRE(diff_ref.nmse() <= NMSE_THRESHOLD, tag + ": logits differ from the single-seq runs (nmse " + std::to_string(diff_ref.nmse()) + ")");
            CTX_REQUIRE(diff_off.nmse() <= NMSE_THRESHOLD, tag + ": the taps changed the logits (nmse " + std::to_string(diff_off.nmse()) + ")");
        }

        return case_result::pass();
    }
};

std::vector<ctx_test_ptr> make_rs_cases() {
    std::vector<ctx_test_ptr> cases;

    cases.push_back(std::make_unique<case_rollback>());
    cases.push_back(std::make_unique<case_rollback_bound>());
    cases.push_back(std::make_unique<case_multi_seq_split_replay>());
    cases.push_back(std::make_unique<case_shared_seq_reserve>());
    cases.push_back(std::make_unique<case_layer_inp>());

    return cases;
}
