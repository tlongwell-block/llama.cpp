#include "testing.h"

#include "mtmd-image.h"
#include "mtmd-vad.h"
#include "mtmd-side.h"
#include "mtmd-ear.h"
#include "mtmd-audio.h"
#include "clip-impl.h"
#include "gguf.h"
#include "mtmd-helper.h"
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <limits>
#include "mtmd-internal.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// this test file contains:
// 1. test cases for mtmd helpers
// 2. test cases for internal mtmd components
// internal headers can be included here

struct test_registry {
    using fn_t = void (*)(testing &);

    struct entry {
        std::string name;
        fn_t fn;
    };

    static std::vector<entry> & all() {
        static std::vector<entry> entries;
        return entries;
    }

    test_registry(const char * name, fn_t fn) {
        all().push_back({ name, fn });
    }
};

#define MAKE_TEST(name)                                               \
    static void name(testing & t);                                    \
    static const test_registry test_registry_ ## name(#name, &name);  \
    static void name(testing & t)


//
// mtmd_image
//

MAKE_TEST(test_image_preprocessor_lfm2) {
    clip_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(64, 256);

    // { image size, expected tiling }
    const std::vector<std::pair<clip_image_size, bool>> cases = {
        { {  704, 704 }, false },
        // 720 / (patch_size * n_merge) is exactly 22.5, so this only matches HF
        // if round_by_factor rounds half to even (22) instead of away from zero (23)
        { {  720, 720 }, false },
        { {  736, 736 }, true  },
        { { 1024, 977 }, true  },
        { { 1056, 384 }, false },
    };

    for (const auto & [size, expected] : cases) {
        const bool actual = mtmd_image_preprocessor_lfm2::should_tile(hparams, size);

        t.assert_equal(
            "tiling for " + std::to_string(size.width) + "x" + std::to_string(size.height),
            std::string(expected ? "tiled" : "single"),
            std::string(actual   ? "tiled" : "single"));
    }
}

//
// mtmd temporal merge
//

MAKE_TEST(test_temporal_merge_grouping) {
    std::vector<mtmd::bitmap_ptr> pool; // keeps the bitmaps alive until the end of the test

    // spec chars:
    //   v = video frame, w = video frame of another size, a = audio, i = plain image, t = text
    auto make_parts = [&pool](const std::string & spec) {
        std::vector<mtmd_internal_part> parts;
        for (char c : spec) {
            if (c == 't') {
                parts.push_back({ "hello", nullptr });
                continue;
            }
            mtmd_bitmap * bm = nullptr;
            switch (c) {
                case 'v': bm = mtmd_bitmap_init(100, 100, nullptr);   break;
                case 'w': bm = mtmd_bitmap_init(200, 200, nullptr);   break;
                case 'a': bm = mtmd_bitmap_init_from_audio(100, nullptr); break;
                case 'i': bm = mtmd_bitmap_init(100, 100, nullptr);   break;
                default: throw std::runtime_error(std::string("unknown spec char: ") + c);
            }
            mtmd_bitmap_set_mergeable(bm, c != 'i');
            pool.emplace_back(bm);
            parts.push_back({ "", bm });
        }
        return parts;
    };

    // { parts, n_merge, expected size of each group }
    const std::vector<std::tuple<std::string, int, std::string>> cases = {
        { "vv",   2, "2"    },
        { "vvv",  2, "21"   },
        { "vvvv", 2, "22"   },
        { "vvi",  2, "21"   },
        { "tvvt", 2, "2"    },
        { "vtv",  2, "11"   }, // text in between breaks the merge
        { "vw",   2, "11"   }, // different sizes cannot be merged
        { "aa",   2, "11"   }, // audio is never merged
        { "ii",   2, "11"   }, // two unrelated images must stay separated
        { "iv",   2, "11"   },
        { "vi",   2, "11"   },
        { "vv",   1, "11"   }, // model without temporal merge
    };

    for (const auto & [spec, n_merge, expected] : cases) {
        auto parts  = make_parts(spec);
        auto groups = mtmd_group_mergeable_bitmaps(parts, n_merge);

        std::string actual;
        for (const auto & group : groups) {
            actual += std::to_string(group.size());
        }

        const std::string name = "\"" + spec + "\" with n_merge=" + std::to_string(n_merge);
        t.assert_equal("groups for " + name, expected, actual);

        size_t n_bitmap_parts = 0;
        for (const auto & p : parts) {
            n_bitmap_parts += p.bitmap != nullptr ? 1 : 0;
        }
        t.assert_equal("remaining bitmap parts for " + name, groups.size(), n_bitmap_parts);
    }
}

MAKE_TEST(test_vad_reference) {
    const char * root = std::getenv("MTMD_VAD_FIXTURE");
    if (!root) {
        std::cout << "SKIP VAD parity: set MTMD_VAD_FIXTURE to the frozen fixture directory\n";
        return;
    }
    const std::string dir(root);
    mtmd_vad vad(dir + "/vad-f32.gguf");
    mtmd_vad other(dir + "/vad-f32.gguf");
    std::ifstream audio(dir + "/chunks.f32", std::ios::binary);
    std::ifstream expected(dir + "/probabilities.f32", std::ios::binary);
    if (!audio || !expected) {
        throw std::runtime_error("missing VAD parity inputs");
    }
    std::array<float, 512> samples{};
    size_t count = 0;
    float worst = 0;
    while (audio.read(reinterpret_cast<char *>(samples.data()), sizeof(samples))) {
        if (count >= 320) { throw std::runtime_error("too many fixture chunks"); }
        if (count == 32 || count == 64) { vad.reset(); other.reset(); }
        float target = 0;
        if (!expected.read(reinterpret_cast<char *>(&target), sizeof(target))) {
            throw std::runtime_error("missing reference probability");
        }
        const float p = vad.process(samples);
        worst = std::max(worst, std::abs(p - target));
        t.assert_equal("reference chunk " + std::to_string(count), true, std::abs(p - target) < 1e-4f);
        t.assert_equal("speech decision " + std::to_string(count), target > 0.5f, p > 0.5f);
        t.assert_equal("stream isolation " + std::to_string(count), p, other.process(samples));
        ++count;
    }
    if (audio.gcount() != 0 || count <= 64 || expected.peek() != EOF) {
        throw std::runtime_error("invalid fixture length");
    }
    vad.reset();
    other.reset();
    samples.fill(0);
    const float first = vad.process(samples);
    for (int i = 0; i < 1024; ++i) { vad.process(samples); }
    vad.reset();
    t.assert_equal("reset replay", first, vad.process(samples));
    samples[0] = std::numeric_limits<float>::quiet_NaN();
    bool rejected = false;
    try { other.process(samples); } catch (const std::runtime_error &) { rejected = true; }
    t.assert_equal("non-finite rejected", true, rejected);
    samples[0] = 0;
    t.assert_equal("invalid input preserves state", first, other.process(samples));
    std::cout << "VAD chunks=" << count << " max_abs_error=" << worst << "\n";
}

MAKE_TEST(test_ear_reference) {
    const char * root = std::getenv("MTMD_EAR_FIXTURE");
    if (!root) {
        std::cout << "SKIP ear parity: set MTMD_EAR_FIXTURE to the frozen fixture directory\n";
        return;
    }
    const std::string dir(root);
    ggml_context * raw = nullptr;
    auto * meta = gguf_init_from_file((dir + "/bridge-f32.gguf").c_str(), {false, &raw});
    const auto weights = std::unique_ptr<ggml_context, decltype(&ggml_free)>(raw, ggml_free);
    const auto metadata = std::unique_ptr<gguf_context, decltype(&gguf_free)>(meta, gguf_free);
    if (!meta || !raw) { throw std::runtime_error("missing ear weights fixture"); }
    mtmd_ear ear(raw);
    auto read = [&](const char * name, size_t count) {
        std::ifstream file(dir + "/" + name + ".f32", std::ios::binary);
        std::vector<float> out(count);
        if (!file.read(reinterpret_cast<char *>(out.data()), count * sizeof(float)) || file.peek() != EOF) {
            throw std::runtime_error("invalid ear fixture");
        }
        return out;
    };
    clip_hparams hp;
    hp.audio_n_fft = 512; hp.audio_window_len = 400; hp.audio_hop_len = 160;
    hp.audio_sample_rate = 16000; hp.n_mel_bins = 80; hp.parakeet_mlx_frontend = true;
    hp.mel_filters = read("filters", 80 * 257); hp.window = read("window", 400);
    mtmd_audio_preprocessor_parakeet frontend(hp);
    frontend.initialize();
    const auto pcm = read("audio16", 32000);
    const auto mel_ref = read("mel", 201 * 80);
    std::vector<mtmd_audio_mel> mels;
    t.assert_equal("ear frontend processes PCM", true, frontend.preprocess(pcm.data(), pcm.size(), mels));
    t.assert_equal("ear frontend frame count", int64_t(201), mels.at(0).n_len);
    float mel_error = 0;
    for (size_t i = 0; i < 201; ++i) {
        for (size_t j = 0; j < 80; ++j) { mel_error = std::max(mel_error, std::abs(mels[0].data[j * 201 + i] - mel_ref[i * 80 + j])); }
    }
    std::cout << "ear frontend max abs error: " << mel_error << "\n";
    t.assert_equal("ear frozen frontend", true, mel_error < 1e-4f);
    t.assert_equal("ear frontend rejects short PCM", false, frontend.preprocess(pcm.data(), 256, mels));
    const auto input = read("frames", 26 * 512);
    if (std::getenv("MTMD_EAR_ENCODER_TEST")) {
        clip_context_params params{};
        params.use_gpu = std::getenv("MTMD_EAR_GPU") != nullptr;
        params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
        params.cb_eval_user_data = const_cast<std::string *>(&dir);
        params.cb_eval = [](ggml_tensor * tensor, bool ask, void * user) {
            const std::string name(tensor->name);
            const bool capture = name == "encoder_out" || (name.find("enc_") == 0 && name.size() > 4 && name.substr(name.size() - 4) == "_res");
            if (ask) { return capture; }
            if (capture && tensor->type == GGML_TYPE_F32) {
                std::vector<float> values(ggml_nelements(tensor));
                ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
                std::ofstream file(*static_cast<std::string *>(user) + "/NATIVE_" + name + ".f32", std::ios::binary);
                file.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
            }
            return true;
        };
        struct reader_state {
            std::ifstream file;
            uint64_t readable;
            size_t largest = 0;
        } source{std::ifstream(dir + "/encoder-f32.gguf", std::ios::binary | std::ios::ate), 0};
        if (!source.file || source.file.tellg() <= 0) { throw std::runtime_error("reader fixture missing"); }
        source.readable = uint64_t(source.file.tellg());
        const uint64_t full_size = source.readable;
        params.model_reader_user_data = &source;
        params.model_reader_size = full_size;
        params.model_reader = [](void * data, void * out, uint64_t offset, size_t n) -> size_t {
            auto & s = *static_cast<reader_state *>(data);
            s.largest = std::max(s.largest, n);
            if (offset > s.readable || n > s.readable - offset) { return 0; }
            s.file.clear(); s.file.seekg(offset);
            if (!s.file.read(static_cast<char *>(out), n)) { return 0; }
            return n;
        };
        const auto loaded = clip_init("not-a-file.gguf", params);
        const auto encoder = std::unique_ptr<clip_ctx, decltype(&clip_free)>(loaded.ctx_a, clip_free);
        if (!encoder) { throw std::runtime_error("cannot load native ear encoder"); }
        t.assert_equal("model reader chunks bounded", true, source.largest <= 1024 * 1024);
        auto failed_load = [&](const char * label) {
            const auto failed = clip_init("not-a-file.gguf", params);
            t.assert_equal(label, true, !failed.ctx_v && !failed.ctx_a && !failed.ctx_gen_a);
            clip_free(failed.ctx_v); clip_free(failed.ctx_a); clip_free(failed.ctx_gen_a);
        };
        source.readable = 16;
        failed_load("model reader rejects short header");
        source.readable = full_size / 2;
        failed_load("model reader rejects truncated weights");
        source.readable = full_size;
        params.model_reader_size = 0;
        failed_load("model reader requires bounds");

        clip_image_f32 image;
        image.set_size({201, 80}, false, true);
        auto encoder_mel = mels[0].data;
        if (std::getenv("MTMD_EAR_FROZEN_MEL")) {
            for (size_t i = 0; i < 201; ++i) {
                for (size_t j = 0; j < 80; ++j) { encoder_mel[j * 201 + i] = mel_ref[i * 80 + j]; }
            }
        }
        image.cpy_buf(encoder_mel);
        std::vector<float> encoded(input.size());
        clip_image_f32_batch batch;
        batch.is_audio = true;
        batch.entries.push_back(std::move(image));
        if (!clip_image_batch_encode(encoder.get(), 4, &batch, encoded)) { throw std::runtime_error("ear encoder failed"); }
        t.assert_equal("ear encoder output size", input.size(), encoded.size());
        float encoder_error = 0;
        double squared_error = 0, power = 0;
        for (size_t i = 0; i < input.size(); ++i) {
            const double d = encoded.at(i) - input[i];
            encoder_error = std::max(encoder_error, float(std::abs(d)));
            squared_error += d * d; power += double(input[i]) * input[i];
        }
        std::ofstream native(dir + "/native_frames.f32", std::ios::binary);
        native.write(reinterpret_cast<const char *>(encoded.data()), encoded.size() * sizeof(float));
        std::cout << "ear encoder max error: " << encoder_error << " SNR: " << 10 * std::log10(power / squared_error) << "\n";
        const auto f32_reference = read("MLX_F32_encoder_out", input.size());
        double precision_error = 0;
        float f32_error = 0;
        for (size_t i = 0; i < input.size(); ++i) {
            const double d = f32_reference[i] - input[i];
            precision_error += d * d;
            f32_error = std::max(f32_error, std::abs(encoded[i] - f32_reference[i]));
        }
        std::cout << "ear encoder F32 reference error: " << f32_error << " precision budget ratio: " << squared_error / precision_error << "\n";
        t.assert_equal("ear encoder F32 math", true, f32_error < 1e-3f);
        t.assert_equal("ear encoder precision budget", true, squared_error < 1.1 * precision_error);

    }

    auto expected = read("ear_rows", 26 * 5120);
    const auto tone = read("tone_row", 5120);
    expected.insert(expected.end(), tone.begin(), tone.end());
    const auto output = ear.process(input.data(), 26);
    float worst = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) { worst = std::max(worst, std::abs(output[i] - expected[i])); }
    std::cout << "ear reference max abs error: " << worst << "\n";
    t.assert_equal("ear frozen reference", true, worst < 1e-4f);
    t.assert_equal("ear repeat", true, output == ear.process(input.data(), 26));
    const auto single = ear.process(input.data(), 1);
    const auto single_tone = read("tone_single", 5120);
    float single_worst = 0.0f;
    for (size_t i = 0; i < 5120; ++i) {
        single_worst = std::max(single_worst, std::abs(single[i] - expected[i]));
        single_worst = std::max(single_worst, std::abs(single[5120 + i] - single_tone[i]));
    }
    std::cout << "ear single-frame max abs error: " << single_worst << "\n";
    t.assert_equal("ear single frame", true, single_worst < 1e-4f);
    std::vector<float> long_input(1024 * 512);
    for (size_t i = 0; i < 1024; ++i) { std::copy_n(input.data(), 512, long_input.data() + i * 512); }
    const auto long_output = ear.process(long_input.data(), 1024);
    float long_worst = 0.0f;
    for (size_t i = 0; i < 1024 * 5120; ++i) { long_worst = std::max(long_worst, std::abs(long_output[i] - single[i % 5120])); }
    t.assert_equal("ear maximum frame batch", true, long_worst < 1e-4f);
    for (size_t n : {size_t(0), size_t(1025)}) {
        bool rejected = false;
        try { ear.process(input.data(), n); } catch (const std::runtime_error &) { rejected = true; }
        t.assert_equal("ear frame bounds", true, rejected);
    }
    auto bad = input;
    bad[0] = std::numeric_limits<float>::quiet_NaN();
    bool rejected = false;
    try { ear.process(bad.data(), 26); } catch (const std::runtime_error &) { rejected = true; }
    t.assert_equal("ear finite input", true, rejected);
    t.assert_equal("ear rejection preserves operation", true, output == ear.process(input.data(), 26));
}

MAKE_TEST(test_side_reference) {
    const char * root = std::getenv("MTMD_SIDE_FIXTURE");
    if (!root) {
        std::cout << "SKIP side parity: set MTMD_SIDE_FIXTURE to the frozen fixture directory\n";
        return;
    }
    const std::string dir(root);
    mtmd_side side(dir + "/side-f32.gguf");
    std::ifstream input(dir + "/inputs.f32", std::ios::binary);
    std::ifstream expected(dir + "/outputs.f32", std::ios::binary);
    if (!input || !expected) { throw std::runtime_error("missing side fixture"); }
    std::array<float, 5120> hidden{};
    std::array<float, 2048> target{};
    size_t rows = 0;
    float worst = 0;
    while (input.read(reinterpret_cast<char *>(hidden.data()), sizeof(hidden))) {
        if (++rows > 64 || !expected.read(reinterpret_cast<char *>(target.data()), sizeof(target))) {
            throw std::runtime_error("invalid side fixture");
        }
        const auto out = side.process(hidden);
        for (size_t i = 0; i < out.size(); ++i) { worst = std::max(worst, std::abs(out[i] - target[i])); }
        t.assert_equal("side row " + std::to_string(rows), true, worst < 1e-4f);
        t.assert_equal("side repeat row " + std::to_string(rows), true, out == side.process(hidden));
    }
    if (rows != 16 || input.gcount() != 0 || expected.peek() != EOF) { throw std::runtime_error("invalid side fixture length"); }
    hidden[0] = std::numeric_limits<float>::quiet_NaN();
    bool rejected = false;
    try { side.process(hidden); } catch (const std::runtime_error &) { rejected = true; }
    t.assert_equal("side rejects non-finite", true, rejected);
    std::cout << "Side rows=" << rows << " max_abs_error=" << worst << "\n";
}

MAKE_TEST(test_qwen3tts_icl_body) {
    const size_t width = 2;
    const std::vector<float> ref{1, 2}, target{3, 4}, codec{5, 6}, eos{7, 8}, pad{9, 10}, bos{11, 12}, cpad{13, 14};
    std::vector<float> out;
    t.assert_equal("ICL accepts valid rows", true, mtmd_helper_qwen3tts_body(
        width, 5, ref, target, codec.data(), 1, eos, pad, bos, cpad, out));
    t.assert_equal("ICL text then codes", true,
        out == std::vector<float>({14, 16, 16, 18, 20, 22, 20, 22, 14, 16}));
    const auto saved = out;
    t.assert_equal("ICL context bound", false, mtmd_helper_qwen3tts_body(
        width, 4, ref, target, codec.data(), 1, eos, pad, bos, cpad, out));
    t.assert_equal("ICL rejects missing codes", false, mtmd_helper_qwen3tts_body(
        width, 5, ref, target, nullptr, 1, eos, pad, bos, cpad, out));
    auto invalid = target;
    invalid[0] = std::numeric_limits<float>::quiet_NaN();
    t.assert_equal("ICL rejects nonfinite rows", false, mtmd_helper_qwen3tts_body(
        width, 5, ref, invalid, codec.data(), 1, eos, pad, bos, cpad, out));
    t.assert_equal("ICL failure preserves output", true, out == saved);
    t.assert_equal("ICL baseline without reference", true, mtmd_helper_qwen3tts_body(
        width, 3, {}, target, nullptr, 0, eos, pad, bos, cpad, out));
    t.assert_equal("ICL baseline rows", true, out == std::vector<float>({16, 18, 20, 22, 20, 22}));

    const char * fixture = std::getenv("MTMD_ICL_FIXTURE");
    if (!fixture) {
        std::cout << "SKIP ICL reference parity: set MTMD_ICL_FIXTURE\n";
        return;
    }
    auto read = [&](const std::string & name) {
        std::ifstream file(std::string(fixture) + "/" + name + ".f32", std::ios::binary | std::ios::ate);
        if (!file || file.tellg() <= 0 || file.tellg() > 32 * 1024 * 1024) {
            throw std::runtime_error("invalid ICL fixture size");
        }
        const size_t size = (size_t) file.tellg();
        if (size % (2048 * sizeof(float))) { throw std::runtime_error("invalid ICL fixture rows"); }
        std::vector<float> data(size / sizeof(float));
        file.seekg(0);
        if (!file.read(reinterpret_cast<char *>(data.data()), size)) { throw std::runtime_error("ICL read failed"); }
        return data;
    };
    const auto reference = read("ref_text"), text = read("target_text"), codes = read("ref_codec");
    const auto expected = read("expected");
    t.assert_equal("frozen MLX ICL packing", true, mtmd_helper_qwen3tts_body(
        2048, 32768, reference, text, codes.data(), codes.size() / 2048,
        read("tts_eos"), read("tts_pad"), read("codec_bos"), read("codec_pad"), out));
    t.assert_equal("frozen ICL length", expected.size(), out.size());
    float worst = 0, rounded_worst = 0;
    if (out.size() != expected.size()) { throw std::runtime_error("ICL size mismatch"); }
    for (size_t j = 0; j < out.size(); ++j) {
        worst = std::max(worst, std::abs(out[j] - expected[j]));
        // MLX adds these rows in BF16; the native prefill retains F32 sums.
        const float rounded = ggml_bf16_to_fp32(ggml_fp32_to_bf16(out[j]));
        rounded_worst = std::max(rounded_worst, std::abs(rounded - expected[j]));
    }
    t.assert_equal("frozen ICL parity at reference precision", 0.0f, rounded_worst);
    std::cout << "ICL body rows=" << out.size() / 2048 << " F32 max_abs_error=" << worst
              << " BF16 rounded max_abs_error=" << rounded_worst << "\n";
}

//
// main
//

int main(int argc, char ** argv) {
    testing t(std::cout);
    t.verbose = true;

    // usage: test-mtmd-impl [filter_regex]
    for (int i = 1; i < argc; i++) {
        t.set_filter(argv[i]);
    }

    for (const auto & e : test_registry::all()) {
        t.test(e.name, e.fn);
    }

    return t.summary();
}
