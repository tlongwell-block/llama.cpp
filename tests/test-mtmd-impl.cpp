#include "testing.h"
#ifdef LLAMA_TEST_FRANKIE
#include "../tools/frankie/text-alignment.h"
#include "../vendor/cpp-httplib/httplib.h"
namespace httplib::ws::impl {
    bool read_websocket_frame(Stream &, Opcode &, std::string &, bool &, bool, size_t);
}
namespace httplib::detail {
    bool write_websocket_frame(Stream &, ws::Opcode, const char *, size_t, bool, bool);
}
#include "../tools/frankie/speech-boundary.h"
#include "../tools/frankie/brain-output.h"
#include "../tools/frankie/turn-session.h"
#include "../tools/frankie/token-boundary.h"
#include "../tools/frankie/image-input.h"
#include "llama-cpp.h"
#endif

#include "mtmd-image.h"
#include "mtmd-vad.h"
#include "mtmd-side.h"
#include "mtmd-turn.h"
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


#ifdef LLAMA_TEST_FRANKIE
MAKE_TEST(test_frankie_context_pool) {
    for (const auto & [total, slots, per_http, voice, expected_http] : std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>>{
            {131072, 0, 0, 131072, 0},
            {300000, 2, 0, 100000, 100000},
            {300001, 2, 0, 100001, 100000},
            {300000, 2, 110000, 80000, 110000},
            {356000, 2, 128000, 100000, 128000},
            {16384, 2, 6144, 4096, 6144},
            {2359296, 8, 0, 262144, 262144}}) {
        frankie_options options;
        options.context_tokens = total;
        options.http_slots = slots;
        options.http_context_tokens = per_http;
        options.resolve_context();
        t.assert_equal("voice receives the pool remainder", voice, options.voice_context_tokens());
        t.assert_equal("HTTP capacity is per slot", expected_http, options.http_context_tokens);
        t.assert_equal("slots fit the total pool", total, options.voice_context_tokens() + slots * options.http_context_tokens);
        options.resolve_context();
        t.assert_equal("resolution is idempotent", voice, options.voice_context_tokens());
    }
    for (const auto & [total, slots, per_http, output] : std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>>{
            {0, 0, 0, 4096}, {4096, 2, 0, 4096}, {300000, 0, 0, 4096},
            {300000, 2, 150000, 4096}, {300000, 2, 200000, 4096},
            {300000, 2, 127, 4096}, {300000, 2, UINT32_MAX, 4096},
            {UINT32_MAX, 8, 0, 4096}, {131072, UINT32_MAX, 0, 4096},
            {300000, 2, 110000, 80001}}) {
        frankie_options options;
        options.context_tokens = total;
        options.http_slots = slots;
        options.http_context_tokens = per_http;
        options.max_output_tokens = output;
        bool rejected = false;
        try { options.resolve_context(); } catch (const std::runtime_error &) { rejected = true; }
        t.assert_true("invalid or overflowing allocation rejected", rejected);
    }
}

MAKE_TEST(test_frankie_image_transport) {
    for (const auto * detail : {"auto", "low", "high"}) { frankie_image_detail(detail); }
    bool rejected = false;
    try { frankie_image_detail("invalid"); } catch (const std::invalid_argument &) { rejected = true; }
    t.assert_true("invalid detail rejected", rejected);
    for (const auto * mime : {"png", "jpeg"}) {
        const auto data = frankie_image_bytes(std::string("data:image/") + mime + ";base64,aGVsbG8=", true);
        t.assert_equal("standard inline transport", std::string("hello"), std::string(data.begin(), data.end()));
    }
    for (const auto * url : {"https://example.com/image.png", "file:///image.png",
                            "data:image/png,abc", "data:image/png;base64,???", "data:image/png;base64,"}) {
        rejected = false;
        try { frankie_image_bytes(url, true); } catch (const std::exception &) { rejected = true; }
        t.assert_true("invalid realtime image transport rejected", rejected);
    }
}

MAKE_TEST(test_frankie_long_token_boundary) {
    const char * package = std::getenv("FRANKIE_PACKAGE_FIXTURE");
    if (!package) { std::cout << "SKIP long token boundary: set FRANKIE_PACKAGE_FIXTURE\n"; return; }
    component source(package, "brain");
    auto params = llama_model_default_params(); params.vocab_only = true;
    llama_model_ptr model(llama_model_init_from_user(source.metadata(), component::set_tensor, &source, params));
    if (!model) { throw std::runtime_error("missing brain vocabulary fixture"); }
    const auto * vocab = llama_model_get_vocab(model.get());
    std::string prefix = "<|im_start|>user\n";
    for (int i = 0; i < 6000; ++i) { prefix += "abcdefghijklmnopqrstuvwxyz\n"; }
    prefix += "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nhello";
    const std::string prompt = prefix + "<|im_end|>\n<|im_start|>user\nContinue.<|im_end|>\n";
    t.assert_equal("long prompt crosses old byte limit", true, prompt.size() > 128 * 1024);
    t.assert_equal("long cached token boundary is reusable", true, frankie_token_boundary(vocab, prompt, prefix.size()));
    t.assert_equal("split inside a token remains rejected", false, frankie_token_boundary(vocab, prompt, prefix.size() - 2));
    t.assert_equal("oversized formatted prompt remains bounded", false,
        frankie_token_boundary(vocab, std::string(4 * 1024 * 1024 + 1, 'x'), 1));
}

MAKE_TEST(test_frankie_turn_queue_recovery) {
    const char * package = std::getenv("FRANKIE_PACKAGE_FIXTURE");
    if (!package) { std::cout << "SKIP queue recovery: set FRANKIE_PACKAGE_FIXTURE\n"; return; }
    component source(package, "vad");
    frankie_options options;
    options.use_gpu = std::getenv("MTMD_COMPONENT_GPU") != nullptr;
    frankie_turn_session turns(source, options);
    t.assert_equal("package has VAP", true, turns.has_vap());
    t.assert_equal("package has backchannels", true, turns.has_bc());
    std::array<float, 512> silence{};
    for (int i = 0; i < 1024; ++i) { turns.append(silence, silence); }
    constexpr uint64_t end_ms = 1024 * 32;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (!turns.current(end_ms).valid && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    t.assert_equal("burst backlog recovers current predictions", true, turns.current(end_ms).valid);
    t.assert_equal("stale predictions fail open", false, turns.current(end_ms + 600).valid);
    t.assert_equal("stale VAP releases", true, turns.release(end_ms + 600, 240));
}

struct frankie_fragmented_stream : httplib::Stream {
    std::string data; size_t offset=0, chunk;
    explicit frankie_fragmented_stream(size_t n):chunk(n){}
    bool is_readable() const override{return offset<data.size();}
    bool wait_readable() const override{return is_readable();}
    bool wait_writable() const override{return true;}
    ssize_t read(char *p,size_t n) override{n=std::min({n,chunk,data.size()-offset}); std::memcpy(p,data.data()+offset,n);offset+=n;return n;}
    ssize_t write(const char *p,size_t n) override{n=std::min(n,chunk);data.append(p,n);return n;}
    void get_remote_ip_and_port(std::string &,int &)const override{}
    void get_local_ip_and_port(std::string &,int &)const override{}
    socket_t socket()const override{return -1;}
    time_t duration()const override{return 0;}
};
MAKE_TEST(test_frankie_websocket_fragments) {
    // A microphone stream crossed a buffered-read boundary after ~43 seconds.
    // Force every header/length/mask to split, independent of TCP packet timing.
    for (size_t chunk : {size_t(1), size_t(3), size_t(4096)}) {
        for (bool mask : {false, true}) {
            for (size_t n : {size_t(0), size_t(1), size_t(125), size_t(126), size_t(65535), size_t(65536)}) {
                std::string expected(n, 'x');
                for (size_t i = 0; i < n; ++i) { expected[i] = char(32 + i % 95); }
                frankie_fragmented_stream stream(chunk);
                t.assert_equal("fragmented frame written", true, httplib::detail::write_websocket_frame(
                    stream, httplib::ws::Opcode::Text, expected.data(), expected.size(), true, mask));
                httplib::ws::Opcode opcode; std::string actual; bool fin = false;
                const bool read = httplib::ws::impl::read_websocket_frame(stream, opcode, actual, fin, mask, 100000);
                t.assert_equal("fragmented frame read", true, read);
                t.assert_equal("fragmented payload unchanged", expected, actual);
                t.assert_equal("final frame preserved", true, fin);
            }
        }
    }
    frankie_fragmented_stream eof(1);
    eof.data = "\x81";
    httplib::ws::Opcode opcode; std::string actual; bool fin;
    t.assert_equal("truncated header rejected", false,
        httplib::ws::impl::read_websocket_frame(eof, opcode, actual, fin, false, 1000));
    frankie_fragmented_stream stalled(0);
    t.assert_equal("zero-byte writer fails", false, httplib::detail::write_websocket_frame(
        stalled, httplib::ws::Opcode::Text, "hello", 5, true, false));
}

MAKE_TEST(test_frankie_speech_boundary) {
    t.assert_equal("no three-word opener", size_t(0), frankie_speech_boundary("Let me check the machine", 0, false));
    t.assert_equal("no clause cut", size_t(0), frankie_speech_boundary("Let me check, then answer", 0, false));
    t.assert_equal("join short interjection", size_t(0), frankie_speech_boundary("Ha. That sounds", 0, false));
    const std::string joined = "Ha. That sounds good. Next";
    t.assert_equal("natural sentence", joined.find(" Next"), frankie_speech_boundary(joined, 0, false));
    t.assert_equal("short final answer", size_t(4), frankie_speech_boundary("Yes.", 0, true));
    t.assert_equal("tool flush", size_t(9), frankie_speech_boundary("Checking.", 0, true));
    std::string long_sentence;
    for (int i = 0; i < 50; ++i) { long_sentence += "word "; }
    t.assert_equal("fifty word bound", long_sentence.size() - 1, frankie_speech_boundary(long_sentence, 0, false));
    const std::string prefix = "This is done.";
    t.assert_equal("next fragment has own count", prefix.size(), frankie_speech_boundary(prefix + " And three words ", prefix.size(), false));

    brain_output raw;
    raw.text = "<think>private</think>Hello. Next phrase.";
    raw.ends = {27, 28, 33, 41};
    raw.hidden.resize(raw.ends.size());
    for (size_t i = 0; i < raw.hidden.size(); ++i) { raw.hidden[i].fill(float(i)); }
    const auto phrase = raw.slice(28, 41);
    t.assert_equal("only phrase text is queued", std::string(" Next phrase."), phrase.text);
    t.assert_equal("prior phrases and reasoning are not copied", size_t(2), phrase.hidden.size());
    t.assert_equal("phrase rows retain their states", 2.0f, phrase.hidden.front()[0]);
    t.assert_true("phrase offsets are rebased", phrase.ends == std::vector<size_t>({5, 13}));
    const auto partial = raw.slice(30, 37);
    t.assert_equal("phrase may clip a token", std::string("ext phr"), partial.text);
    t.assert_true("clipped token offsets", partial.ends == std::vector<size_t>({3, 7}));
}

MAKE_TEST(test_frankie_unicode_alignment) {
    t.assert_equal("bare spoken digit", std::string("four"), frankie_spoken_numbers("4"));
    t.assert_equal("spoken number in text", std::string("It is fifty-six."), frankie_spoken_numbers("It is 56."));
    t.assert_equal("preserve identifiers and long digit strings", std::string("Q4 model 123456789012345"), frankie_spoken_numbers("Q4 model 123456789012345"));
    struct fixture {
        std::string text, normalized;
        std::vector<size_t> brain_ends, talker_ends, indices;
        std::vector<std::pair<size_t, size_t>> offsets;
    };
    // Frozen HF Qwen brain/talker token offsets, checked against actual converted vocabularies.
    const std::vector<fixture> cases = {
        {"It\342\200\231s a lovely day.","It\342\200\231s a lovely day.",{2,6,8,15,19,20},{2,6,8,15,19,20},{0,1,2,3,4,5},{{0,2},{2,4},{4,6},{6,13},{13,17},{17,18}}},
        {"  Caf\303\251 \342\200\224 d\303\251j\303\240 vu!","Caf\303\251 \342\200\224 d\303\251j\303\240 vu!",{1,7,11,18,21,22},{1,3,5,9,16,19,20},{1,1,1,2,3,4,5},{{0,1},{1,3},{3,4},{4,6},{6,11},{11,14},{14,15}}},
        {"\344\275\240\345\245\275\357\274\214\344\270\226\347\225\214\343\200\202","\344\275\240\345\245\275\357\274\214\344\270\226\347\225\214\343\200\202",{6,9,15,18},{6,9,15,18},{0,1,2,3},{{0,2},{2,3},{3,5},{5,6}}},
        {"An emoji \360\237\246\212 says hi.","An emoji \360\237\246\212 says hi.",{2,8,11,12,13,18,21,22},{2,8,11,12,13,18,21,22},{0,1,2,2,2,5,6,7},{{0,2},{2,8},{8,10},{9,10},{9,10},{10,15},{15,18},{18,19}}},
        {"e\314\201 is accented.","\303\251 is accented.",{1,3,6,10,15,16},{2,5,9,14,15},{0,2,3,4,5},{{0,1},{2,5},{5,9},{9,14},{14,15}}},
        {"\302\240Hello\342\200\203world.","Hello\342\200\203world.",{2,7,9,10,15,16},{5,7,8,13,14},{1,2,2,4,5},{{0,5},{5,6},{5,6},{6,11},{11,12}}},
        {"e\314\201 is accented.","\303\251 is accented.",{1,3,6,10,15,16},{2,5,9,14,15},{0,2,3,4,5},{{0,1},{2,5},{5,9},{9,14},{14,15}}},
        {"\341\204\200\341\205\241\341\206\250 test","\352\260\201 test",{1,2,3,4,5,6,7,8,9,14},{3,8},{0,9},{{0,1},{3,8}}},
        {"a\314\201\314\243 test","\341\272\241\314\201 test",{1,3,4,5,10},{3,5,10},{0,2,4},{{0,1},{2,3},{3,8}}},
        {"\342\204\253 test","\303\205 test",{2,3,8},{2,7},{0,2},{{0,1},{1,6}}},
        {"a\314\201b","\303\241b",{1,3,4},{3},{0},{{0,3}}},
        {"\315\204","\314\210\314\201",{1,2},{1,2,4},{0,0,0},{{0,1},{0,1},{0,1}}},
        {"\341\270\212\314\243","\341\270\214\314\207",{1,2,3,4,5},{3,4,5},{0,3,3},{{0,1},{1,2},{1,2}}},
        {"\342\200\203Hello\302\240","Hello",{2,3,8,10},{5},{2},{{0,5}}},
        {"A \360\237\221\251\360\237\217\275\342\200\215\360\237\222\273 works.","A \360\237\221\251\360\237\217\275\342\200\215\360\237\222\273 works.",{1,5,6,8,9,10,12,13,16,17,23,24},{1,5,6,10,12,13,17,23,24},{0,1,1,3,6,6,8,10,11},{{0,1},{1,3},{2,3},{3,4},{4,5},{4,5},{5,6},{6,12},{12,13}}},
        {"\314\201\314\243 starts.","\314\243\314\201 starts.",{2,3,4,11,12},{1,2,4,11,12},{0,0,1,3,4},{{0,1},{0,1},{1,2},{2,9},{9,10}}},
        {"\341\270\212\314\243","\341\270\214\314\207",{1,2,3,4,5},{3,4,5},{0,3,3},{{0,1},{1,2},{1,2}}},
        {"I\342\200\231m here. What\342\200\231s next?","I\342\200\231m here. What\342\200\231s next?",{1,5,10,11,16,20,25,26},{1,5,10,11,16,20,25,26},{0,1,2,3,4,5,6,7},{{0,1},{1,3},{3,8},{8,9},{9,14},{14,16},{16,21},{21,22}}},
        {"\012\012Sure \342\200\224 let\342\200\231s go!","Sure \342\200\224 let\342\200\231s go!",{2,6,10,14,18,21,22},{4,8,12,16,19,20},{1,2,3,4,5,6},{{0,4},{4,6},{6,10},{10,12},{12,15},{15,16}}},
    };
    for (const auto & c : cases) {
        const frankie_text_offsets map(c.text);
        size_t lead = 0, tail = c.text.size();
        while (lead < tail && map.whitespace[map.floor[lead]]) { ++lead; }
        while (tail > lead && map.whitespace[map.floor[tail - 1]]) { --tail; }
        const auto spoken = c.text.substr(lead, tail - lead);
        const frankie_normalized_text normalized(spoken);
        t.assert_true("NFC text", normalized.text == c.normalized);
        const auto spans = normalized.spans(c.talker_ends);
        t.assert_true("original character offsets", spans == c.offsets);
        t.assert_true("hidden row alignment", frankie_align_text(c.text, c.brain_ends, lead, spoken, c.talker_ends, spans) == c.indices);
        brain_output raw;
        raw.text = c.text; raw.ends = c.brain_ends; raw.hidden.resize(c.brain_ends.size());
        for (size_t i = 0; i < raw.hidden.size(); ++i) { raw.hidden[i].fill(float(i)); }
        const auto phrase = raw.slice(lead, tail);
        const auto aligned = frankie_align_text(phrase.text, phrase.ends, 0, spoken, c.talker_ends, spans);
        for (size_t i = 0; i < aligned.size(); ++i) {
            t.assert_equal("sliced Unicode preserves hidden row", float(c.indices[i]), phrase.hidden[aligned[i]][0]);
        }
    }
    for (const std::string & invalid : {std::string("\300\200"), std::string("\355\240\200"), std::string("\360\237"), std::string("\364\220\200\200")}) {
        bool rejected = false;
        try { const frankie_normalized_text normalized(invalid); } catch (const std::runtime_error &) { rejected = true; }
        t.assert_true("invalid UTF-8 rejected", rejected);
    }
}
#endif

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

static std::unique_ptr<ggml_context, decltype(&ggml_free)> load_component_fixture(const std::string & path) {
    ggml_context * raw = nullptr;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
        gguf_init_from_file(path.c_str(), {false, &raw}), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> weights(raw, ggml_free);
    if (!metadata || !weights) { throw std::runtime_error("missing component fixture: " + path); }
    return weights;
}

MAKE_TEST(test_vad_reference) {
    const char * root = std::getenv("MTMD_VAD_FIXTURE");
    if (!root) {
        std::cout << "SKIP VAD parity: set MTMD_VAD_FIXTURE to the frozen fixture directory\n";
        return;
    }
    const std::string dir(root);
    auto weights = load_component_fixture(dir + "/vad-f32.gguf");
    mtmd_vad vad(weights.get(), std::getenv("MTMD_COMPONENT_GPU"));
    mtmd_vad other(weights.get(), std::getenv("MTMD_COMPONENT_GPU"));
    weights.reset();
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
    const auto weights = load_component_fixture(dir + "/bridge-f32.gguf");
    mtmd_ear ear(weights.get(), std::getenv("MTMD_COMPONENT_GPU"));
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
            const bool capture = name == "encoder_out" || name.find("pre_") == 0 || (name.find("enc_") == 0 && name.size() > 4 && name.substr(name.size() - 4) == "_res");
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
    std::vector<int32_t> ids;
    t.assert_equal("ear CTC preserves embeddings", true, output == ear.process(input.data(), 26, &ids));
    t.assert_equal("ear CTC frame count", size_t(26), ids.size());
    t.assert_equal("ear CTC ids", true, std::all_of(ids.begin(), ids.end(), [](int32_t id) { return id >= 0 && id < 1025; }));
    const auto single = ear.process(input.data(), 1);
    const auto single_tone = read("tone_single", 5120);
    float single_worst = 0.0f;
    for (size_t i = 0; i < 5120; ++i) {
        single_worst = std::max(single_worst, std::abs(single[i] - expected[i]));
        single_worst = std::max(single_worst, std::abs(single[5120 + i] - single_tone[i]));
    }
    std::cout << "ear single-frame max abs error: " << single_worst << "\n";
    t.assert_equal("ear single frame", true, single_worst < 1e-4f);
    std::vector<float> long_input(1502 * 512);
    for (size_t i = 0; i < 1502; ++i) { std::copy_n(input.data(), 512, long_input.data() + i * 512); }
    const auto long_output = ear.process(long_input.data(), 1502);
    float long_worst = 0.0f;
    for (size_t i = 0; i < 1502 * 5120; ++i) { long_worst = std::max(long_worst, std::abs(long_output[i] - single[i % 5120])); }
    t.assert_equal("ear maximum frame batch", true, long_worst < 1e-4f);
    t.assert_equal("ear short replay after long input", true, output == ear.process(input.data(), 26, &ids));
    for (size_t n : {size_t(0), size_t(1503)}) {
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
    auto weights = load_component_fixture(dir + "/side-f32.gguf");
    mtmd_side side(weights.get(), std::getenv("MTMD_COMPONENT_GPU"));
    weights.reset();
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

MAKE_TEST(test_turn_reference) {
    const char * root = std::getenv("MTMD_TURN_FIXTURE");
    if (!root) { std::cout << "SKIP turn parity: set MTMD_TURN_FIXTURE\n"; return; }
    const std::string dir(root);
    for (const std::string mode : {"vap", "bc", "duplex"}) {
        if (mode == "duplex" && !std::ifstream(dir + "/duplex.gguf")) { continue; }
        const auto weights = load_component_fixture(dir + "/" + mode + ".gguf");
        mtmd_turn model(weights.get(), mode == "vap" ? mtmd_turn::mode::vap :
                                      mode == "bc" ? mtmd_turn::mode::backchannel : mtmd_turn::mode::duplex,
                        std::getenv("MTMD_COMPONENT_GPU"));
        std::ifstream input(dir + "/input.f32", std::ios::binary);
        std::ifstream output(dir + "/" + mode + "-output.f32", std::ios::binary);
        std::ifstream encoder(dir + "/" + mode + "-encoder.f32", std::ios::binary);
        std::ifstream hidden(dir + "/" + mode + "-hidden.f32", std::ios::binary);
        if (!input || !output || !encoder || !hidden) { throw std::runtime_error("missing turn reference data"); }
        float worst_p = 0, worst_e = 0, worst_h = 0;
        std::array<float, 1600> user{}, system{};
        auto compare = [](std::ifstream & in, const std::vector<float> & actual, float & worst) {
            std::vector<float> expected(actual.size());
            if (!in.read(reinterpret_cast<char *>(expected.data()), expected.size() * sizeof(float))) {
                throw std::runtime_error("short turn fixture");
            }
            for (size_t i = 0; i < expected.size(); ++i) { worst = std::max(worst, std::abs(expected[i] - actual[i])); }
        };
        mtmd_turn::result first{};
        for (int i = 0; i < 250; ++i) {
            if (!input.read(reinterpret_cast<char *>(user.data()), sizeof(user)) ||
                !input.read(reinterpret_cast<char *>(system.data()), sizeof(system))) { throw std::runtime_error("short turn audio"); }
            const auto r = model.process(user, system);
            if (!i) { first = r; }
            compare(output, mode == "duplex" ? std::vector<float>{r.next_speaker[0], r.next_speaker[1], r.backchannel} :
                            mode == "vap" ? std::vector<float>{r.next_speaker[0], r.next_speaker[1]} :
                                            std::vector<float>{r.backchannel}, worst_p);
            compare(encoder, model.encoded(), worst_e);
            compare(hidden, model.hidden(), worst_h);
            if (i == 0 || i == 69 || i == 200) {
                std::cout << mode << " frame=" << i << " error p=" << worst_p << " encoder=" << worst_e << " hidden=" << worst_h << "\n";
            }
        }
        std::cout << mode << " reference max error p=" << worst_p << " encoder=" << worst_e << " hidden=" << worst_h << "\n";
        t.assert_equal(mode + " probability", true, worst_p < 2e-4f);
        t.assert_equal(mode + " encoder", true, worst_e < 2e-4f);
        t.assert_equal(mode + " hidden", true, worst_h < 2e-3f);
        model.reset();
        user.fill(0); system.fill(0);
        const auto again = model.process(user, system);
        t.assert_equal(mode + " reset VAP", true, std::abs(again.next_speaker[0] - first.next_speaker[0]) < 1e-6f);
        t.assert_equal(mode + " reset system", true, std::abs(again.next_speaker[1] - first.next_speaker[1]) < 1e-6f);
        t.assert_equal(mode + " reset BC", true, std::abs(again.backchannel - first.backchannel) < 1e-6f);
    }
}

MAKE_TEST(test_expression_reference) {
    const char * root = std::getenv("MTMD_EXPRESSION_FIXTURE");
    if (!root) {
        std::cout << "SKIP expression parity: set MTMD_EXPRESSION_FIXTURE\n";
        return;
    }
    const std::string dir(root);
    const auto weights = load_component_fixture(dir + "/expression.gguf");
    mtmd_expression head(weights.get(), std::getenv("MTMD_COMPONENT_GPU"));
    std::ifstream input(dir + "/expression-inputs.f32", std::ios::binary);
    std::ifstream expected(dir + "/expression-outputs.f32", std::ios::binary);
    if (!input || !expected) { throw std::runtime_error("missing expression outputs"); }
    std::array<float, 5120> hidden{};
    mtmd_expression::result target;
    float worst = 0;
    for (int row = 0; row < 16; ++row) {
        if (!input.read(reinterpret_cast<char *>(hidden.data()), sizeof(hidden)) ||
            !expected.read(reinterpret_cast<char *>(target.probabilities.data()), sizeof(target.probabilities)) ||
            !expected.read(reinterpret_cast<char *>(target.offset.data()), sizeof(target.offset))) {
            throw std::runtime_error("short expression fixture");
        }
        const auto actual = head.process(hidden);
        for (size_t i = 0; i < 4; ++i) { worst = std::max(worst, std::abs(actual.probabilities[i] - target.probabilities[i])); }
        for (size_t i = 0; i < 2048; ++i) { worst = std::max(worst, std::abs(actual.offset[i] - target.offset[i])); }
        t.assert_equal("expression reference row " + std::to_string(row), true, worst < 1e-4f);
    }
    t.assert_equal("expression fixture extent", true, input.peek() == EOF && expected.peek() == EOF);
    hidden[0] = std::numeric_limits<float>::quiet_NaN();
    bool rejected = false;
    try { head.process(hidden); } catch (const std::runtime_error &) { rejected = true; }
    t.assert_equal("expression rejects nonfinite", true, rejected);
    std::cout << "Expression max_abs_error=" << worst << "\n";
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
