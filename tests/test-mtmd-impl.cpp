#include "testing.h"
#ifdef LLAMA_TEST_FRANKIE
#include "../tools/frankie/text-alignment.h"
#include "../vendor/cpp-httplib/httplib.h"
namespace httplib::detail {
    bool write_websocket_frame(Stream &, ws::Opcode, const char *, size_t, bool, bool);
}
#include "../tools/frankie/speech-boundary.h"
#include "../tools/frankie/backchannel-gate.h"
#include "../tools/frankie/brain-output.h"
#include "../tools/frankie/brain-session.h"
#include "../tools/frankie/turn-session.h"
#include "../tools/frankie/token-boundary.h"
#include "../tools/frankie/image-input.h"
#include "../tools/frankie/watermark.h"
#include "llama-cpp.h"
#endif

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "mtmd-audio.h"
#include "mtmd-subcontext.h"
#include "mtmd-helper.h"
#include "mtmd-image.h"
#include "mtmd-vad.h"
#include "mtmd-side.h"
#include "mtmd-turn.h"
#include "mtmd-watermark.h"
#include "mtmd-ear.h"
#include "mtmd-audio.h"
#include "mtmd-model.h"
#include "gguf.h"
#include "mtmd-helper.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <limits>
#include "mtmd-internal.h"
#include "models/models.h"

#include "chat-tts-norm.h"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
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
MAKE_TEST(test_frankie_prompt_counts) {
    frankie_prompt_counts cache(1024, 2);
    size_t calls = 0;
    auto count = [&](const std::string & text) {
        return cache.get(text, [&] { ++calls; return text.size(); });
    };
    t.assert_equal("empty text needs no tokenizer", size_t(0), count(""));
    t.assert_equal("new segment counted", size_t(3), count("one"));
    count("two"); count("one"); count("six"); count("one");
    t.assert_equal("recent segment survives entry eviction", size_t(3), calls);
    count("two");
    t.assert_equal("old segment must be recounted", size_t(4), calls);
    const std::string oversized(1024, 'x');
    count(oversized); count(oversized);
    t.assert_equal("oversized segments are not retained", size_t(6), calls);
    count("two");
    t.assert_equal("oversized miss preserves useful entries", size_t(6), calls);
    cache.clear(); count("two");
    t.assert_equal("reset drops prior session text", size_t(7), calls);
    frankie_prompt_counts bounded(1024);
    for (const auto & text : {std::string(600, 'a'), std::string(600, 'b'), std::string(600, 'a')}) {
        bounded.get(text, [&] { ++calls; return size_t(1); });
    }
    t.assert_equal("byte cap evicts independently of entry cap", size_t(10), calls);
    frankie_prompt_counts exact;
    for (const auto & text : {"tool result: one", "tool result: two", "tool result: one ", "tool result: one"}) {
        exact.get(text, [&] { ++calls; return size_t(1); });
    }
    t.assert_equal("changed tool content and whitespace cannot alias", size_t(13), calls);
}

MAKE_TEST(test_frankie_backchannel_gate) {
    frankie_backchannel_gate gate;
    t.assert_true("one low observation waits", !gate.observe(0.1f, 80, true));
    t.assert_true("duplicate observation cannot trigger", !gate.observe(0.1f, 80, true));
    t.assert_true("ordinary speech yields on second observation", gate.observe(0.1f, 160, true));
    gate = {};
    t.assert_true("first nod observation holds", !gate.observe(0.3f, 80, true));
    t.assert_true("confirmed nod holds", !gate.observe(0.3f, 160, true));
    t.assert_true("nod release uses lower threshold", !gate.observe(0.1f, 240, true));
    t.assert_true("one low nod tail holds", !gate.observe(0.01f, 320, true));
    t.assert_true("silence clears consecutive evidence", !gate.observe(0, 400, false));
    t.assert_true("resumed speech requires fresh evidence", !gate.observe(0.01f, 480, true));
    t.assert_true("nod followed by real speech yields", gate.observe(0.01f, 560, true));
    t.assert_true("out-of-order observation ignored", !gate.observe(0.01f, 520, true));
}

MAKE_TEST(test_frankie_context_pool) {
    t.assert_equal("output ceiling fits", size_t(65536), frankie_http_output_budget(47000, 65536, 128000));
    t.assert_equal("output ceiling uses remaining capacity", size_t(81000), frankie_http_output_budget(47000, 100000, 128000));
    t.assert_equal("large requested ceiling stays within the slot", size_t(81000), frankie_http_output_budget(47000, INT32_MAX, 128000));
    t.assert_equal("last available output token", size_t(1), frankie_http_output_budget(127999, 512, 128000));
    for (size_t prompt : {128000u, 128001u}) {
        bool rejected = false;
        try { frankie_http_output_budget(prompt, 1, 128000); }
        catch (const std::runtime_error & e) {
            rejected = std::string(e.what()).find("prompt_tokens=" + std::to_string(prompt)) != std::string::npos;
        }
        t.assert_true("full prompt reports its token count", rejected);
    }
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
        options.max_output_tokens = voice + 1;
        options.resolve_context();
        t.assert_equal("voice output ceiling uses remaining capacity", voice, options.max_output_tokens);
    }
    for (const auto & [total, slots, per_http, output] : std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>>{
            {0, 0, 0, 4096}, {4096, 2, 0, 4096}, {300000, 0, 0, 4096},
            {300000, 2, 150000, 4096}, {300000, 2, 200000, 4096},
            {300000, 2, 127, 4096}, {300000, 2, UINT32_MAX, 4096},
            {UINT32_MAX, 8, 0, 4096}, {131072, UINT32_MAX, 0, 4096}}) {
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
    frankie_prompt_counts counts;
    for (int round = 0; round < 3; ++round) {
        for (const std::string text : {"<|im_start|>user\nHello.<|im_end|>\n", "Tool result: one", "Tool result: two", "", "\xc3\xa9\xe4\xb8\xad"}) {
            auto tokenize = [&] {
                const int n = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, false, true);
                return size_t(n < 0 ? -n : n);
            };
            t.assert_equal("memoized count matches model vocabulary", tokenize(), counts.get(text, tokenize));
        }
    }
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
                const bool read = httplib::ws::impl::read_websocket_frame(stream, opcode, actual, fin, mask, 100000) == httplib::ws::impl::FrameRead::Ok;
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
        httplib::ws::impl::read_websocket_frame(eof, opcode, actual, fin, false, 1000) == httplib::ws::impl::FrameRead::Ok);
    frankie_fragmented_stream stalled(0);
    t.assert_equal("zero-byte writer fails", false, httplib::detail::write_websocket_frame(
        stalled, httplib::ws::Opcode::Text, "hello", 5, true, false));
}

MAKE_TEST(test_frankie_realtime_history) {
    brain_session::request request;
    common_chat_msg user, empty, heard, call;
    user.role = "user"; user.content = "question";
    empty.role = heard.role = call.role = "assistant";
    heard.content = "First sentence.";
    call.tool_calls.push_back({"lookup", "{}", "call_one"});
    request.chat.messages = {user, empty, heard, call};
    request.playback_cutoffs.insert(2);
    t.assert_equal("HTTP history untouched", size_t(4), request.chat_input().messages.size());
    request.realtime_history = true;
    auto input = request.chat_input();
    t.assert_equal("empty assistant omitted, notice inserted", size_t(4), input.messages.size());
    t.assert_equal("heard text retained", std::string("First sentence."), input.messages[1].content);
    t.assert_equal("notice is not spoken assistant text", std::string("user"), input.messages[2].role);
    t.assert_true("notice names confirmed heard phrases",
        input.messages[2].content.find("contains only confirmed heard phrases") != std::string::npos);
    t.assert_equal("empty tool call retained", size_t(1), input.messages[3].tool_calls.size());
    t.assert_equal("authoritative item positions untouched", size_t(4), request.chat.messages.size());
    request.chat.messages[1].content = " \t\r\n";
    t.assert_equal("whitespace-only assistant omitted", size_t(4), request.chat_input().messages.size());
    auto unheard = request;
    unheard.playback_cutoffs = {1};
    t.assert_true("notice without heard text",
        unheard.chat_input().messages[1].content.find("before any complete phrase") != std::string::npos);
    request.playback_cutoffs = {3};
    t.assert_equal("unfinished call has no intervening notice", size_t(3), request.chat_input().messages.size());
    common_chat_msg result;
    result.role = "tool"; result.tool_call_id = "call_one"; result.content = "result";
    request.chat.messages.push_back(result);
    input = request.chat_input();
    t.assert_equal("tool result precedes playback notice", std::string("tool"), input.messages[3].role);
    t.assert_equal("notice follows complete call group", std::string("user"), input.messages[4].role);
    common_chat_msg system;
    system.role = "system"; system.content = "Be brief.";
    request.chat.messages = {system, user};
    request.playback_cutoffs.clear();
    t.assert_equal("no listener, no cutoff: instructions unchanged", std::string("Be brief."), request.chat_input().messages[0].content);
    request.streaming_listener = true;
    const auto listened = request.chat_input().messages[0].content;
    t.assert_true("listener appends regeneration instructions",
        listened.rfind("Be brief.\n\nAutomatic playback notices are engine-provided history data", 0) == 0 &&
        listened.find("Respect a request for silence by ending without speech.") != std::string::npos);
    t.assert_equal("authoritative instructions untouched", std::string("Be brief."), request.chat.messages[0].content);
}

MAKE_TEST(test_frankie_speech_boundary) {
    t.assert_equal("confirmed lookahead releases same phrase", size_t(0), frankie_speech_boundary("This is a sentence", 0, false, true));
    t.assert_equal("lookahead after punctuation", size_t(19), frankie_speech_boundary("This is a sentence.", 0, false, true));
    t.assert_equal("pending decimal does not cut", size_t(0), frankie_speech_boundary("It costs about 1.", 0, false, false));
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

MAKE_TEST(test_image_preprocessor_longest_edge) {
    mtmd_hparams hparams;
    hparams.image_longest_edge = 1540;
    hparams.patch_size = 14;
    hparams.n_merge = 2;

    const std::vector<std::pair<mtmd_image_size, mtmd_image_size>> cases = {
        { {  463,  463 }, {  476,  476 } },
        { {  463,  300 }, {  476,  308 } },
        { {  448,  280 }, {  448,  280 } },
        { { 2000, 1000 }, { 1540,  784 } },
    };

    for (const auto & [input, expected] : cases) {
        const mtmd_image_size actual = mtmd_image_preprocessor_longest_edge::get_target_size(hparams, input);
        const std::string name = std::to_string(input.width) + "x" + std::to_string(input.height);
        t.assert_equal(name + " width", expected.width, actual.width);
        t.assert_equal(name + " height", expected.height, actual.height);
    }
}

MAKE_TEST(test_image_preprocessor_dyn_size) {
    // { patch_size, image size, expected size }
    // a side that is an even number plus .5 merged patches rounds down, like Python round() in smart_resize
    const std::vector<std::tuple<int, mtmd_image_size, mtmd_image_size>> cases = {
        { 16, {  768, 400 }, {  768, 384 } },
        { 16, {  800, 336 }, {  800, 320 } },
        { 16, {  400, 400 }, {  384, 384 } },
        { 16, {  640, 240 }, {  640, 256 } },
        { 14, {  966, 560 }, {  952, 560 } },
    };

    for (const auto & [patch_size, input, expected] : cases) {
        mtmd_hparams hparams;
        hparams.patch_size = patch_size;
        hparams.n_merge = 2;
        hparams.image_round_half_even = true;
        hparams.set_limit_image_tokens(8, 4096);

        const mtmd_image_size actual = mtmd_image_preprocessor_dyn_size::get_target_size(hparams, input);
        const std::string name = std::to_string(input.width) + "x" + std::to_string(input.height);
        t.assert_equal(name + " width", expected.width, actual.width);
        t.assert_equal(name + " height", expected.height, actual.height);
    }
}

MAKE_TEST(test_image_preprocessor_qwen3vl_video) {
    mtmd_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    const mtmd_image_preprocessor_qwen3vl preproc(hparams);

    // { n_frames, frame size, expected size from Qwen3VLVideoProcessor smart_resize }
    const std::vector<std::tuple<int, mtmd_image_size, mtmd_image_size>> cases = {
        {  20, {  720,  358 }, {  704,  352 } }, // 720 / 32 = 22.5 rounds half to even
        { 768, { 1920, 1080 }, {  224,  128 } }, // over the longest_edge budget
        {   4, {   16,   40 }, {   32,   64 } }, // edge below the factor is upscaled first
        {   2, {   32,   32 }, {   64,   64 } }, // under the shortest_edge budget
        {   5, { 2760, 1552 }, { 2752, 1536 } }, // 5 / 2 = 2.5 rounds half to even
        {   4, {  250, 6000 }, {  256, 6016 } },
    };

    for (const auto & [n_frames, input, expected] : cases) {
        mtmd_image_size actual;
        const std::string name = std::to_string(n_frames) + " frames of " + std::to_string(input.width) + "x" + std::to_string(input.height);
        if (!t.assert_true(name + " has a frame size", preproc.get_video_frame_size(n_frames, input.width, input.height, actual))) {
            continue;
        }
        t.assert_equal(name + " width", expected.width, actual.width);
        t.assert_equal(name + " height", expected.height, actual.height);
    }
}

static std::string image_size_to_str(const mtmd_image_size & size) {
    return std::to_string(size.width) + "x" + std::to_string(size.height);
}

static std::string resize_algo_to_str(resize_algo algo) {
    std::string str;
    switch (algo & ~RESIZE_ALGO_FLAG_TORCH) {
        case RESIZE_ALGO_BILINEAR: str = "bilinear"; break;
        case RESIZE_ALGO_BICUBIC:  str = "bicubic";  break;
        case RESIZE_ALGO_LANCZOS:  str = "lanczos";  break;
        default:                   str = "unknown";  break;
    }
    if (algo & RESIZE_ALGO_FLAG_TORCH) {
        str += "_torch";
    }
    return str;
}

// pseudo-random pixels: s = s * 1103515245 + 12345 from s = 1, one byte (s >> 16) & 255 per channel
static mtmd_image_u8 make_test_image(const mtmd_image_size & size) {
    mtmd_image_u8 img;
    img.set_size(size, false);
    uint32_t s = 1;
    auto next = [&s]() {
        s = s * 1103515245u + 12345u;
        return static_cast<uint8_t>(s >> 16);
    };
    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {
            img.set_pixel(x, y, { next(), next(), next() });
        }
    }
    return img;
}

// FNV-1a of the pixels
static uint32_t hash_pixels(const mtmd_image_u8 & img, uint32_t hash = 2166136261u) {
    for (uint8_t v : img.get_ro_buf()) {
        hash = (hash ^ v) * 16777619u;
    }
    return hash;
}

MAKE_TEST(test_image_preprocessor_lfm2) {
    mtmd_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(64, 256);
    hparams.preproc_min_tiles            = 2;
    hparams.preproc_max_tiles            = 10;
    hparams.preproc_tile_size            = 512;
    hparams.preproc_max_pixels_tolerance = 2.0f;

    // { image size, expected tiling }
    const std::vector<std::pair<mtmd_image_size, bool>> cases = {
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

    // { image size, expected size from HF smart_resize }
    const std::vector<std::pair<mtmd_image_size, mtmd_image_size>> resize_cases = {
        { {  400, 600 }, { 384, 608 } }, // 400 / 32 = 12.5 rounds half to even
        { {  480, 336 }, { 480, 320 } },
        { {   64, 169 }, { 160, 416 } }, // 169 * beta / 32 is exactly 13 in double
        { { 1024, 768 }, { 576, 416 } },
    };

    for (const auto & [size, expected] : resize_cases) {
        t.assert_equal("smart_resize for " + image_size_to_str(size), image_size_to_str(expected),
            image_size_to_str(mtmd_image_preprocessor_lfm2::smart_resize(hparams, size)));
    }

    // { image size, expected grid from HF }
    const std::vector<std::pair<mtmd_image_size, mtmd_image_size>> grid_cases = {
        { {  720, 1600 }, { 2, 4 } }, // 0.45 is an exact tie between 2/4 and 2/5 in double, not in float
        { {  420, 1440 }, { 1, 3 } },
        { {  255, 2160 }, { 1, 8 } },
        { { 1024,  768 }, { 3, 2 } },
    };

    for (const auto & [size, expected] : grid_cases) {
        t.assert_equal("grid for " + image_size_to_str(size), image_size_to_str(expected),
            image_size_to_str(mtmd_image_preprocessor_lfm2::get_grid_layout(hparams, size.height, size.width)));
    }

    // do_image_splitting=false
    hparams.preproc_min_tiles = 1;
    hparams.preproc_max_tiles = 1;
    t.assert_equal("tiling for 1024x977 with max_tiles=1", std::string("single"),
            std::string(mtmd_image_preprocessor_lfm2::should_tile(hparams, { 1024, 977 }) ? "tiled" : "single"));
}

MAKE_TEST(test_image_preprocessor_lfm2_torch_resize) {
    mtmd_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(64, 256);
    hparams.preproc_min_tiles            = 2;
    hparams.preproc_max_tiles            = 10;
    hparams.preproc_tile_size            = 512;
    hparams.preproc_max_pixels_tolerance = 2.0f;
    // Lfm2VlImageProcessor stretches the image to the tile grid
    hparams.image_pad_rf = PAD_NONE;

    // { resize algo, image size, expected hash of the tiles and the thumbnail from Lfm2VlImageProcessor (torchvision 0.29, native uint8 resize: arm64, or x86 with AVX2/AVX512) }
    const std::vector<std::tuple<resize_algo, mtmd_image_size, uint32_t>> cases = {
        { RESIZE_ALGO_BILINEAR, {  620,  410 }, 0xf5e9c941 }, // 608x416
        { RESIZE_ALGO_BILINEAR, { 2000, 1500 }, 0x3331f56c }, // 3x2 tiles, 576x416 thumbnail
        { RESIZE_ALGO_BICUBIC,  {  620,  410 }, 0x2b34eac9 },
        { RESIZE_ALGO_BICUBIC,  { 2000, 1500 }, 0xb035d942 },
    };

    for (const auto & [base_algo, size, expected] : cases) {
        const resize_algo algo = static_cast<resize_algo>(base_algo | RESIZE_ALGO_FLAG_TORCH);
        hparams.image_resize_algo_rf = algo;
        hparams.image_resize_algo_ov = algo;
        const mtmd_image_preprocessor_lfm2 preproc(hparams);
        const auto sliced = preproc.slice_image(make_test_image(size), preproc.get_slice_instructions(size));

        uint32_t hash = 2166136261u;
        for (const auto & slice : sliced.slices) {
            hash = hash_pixels(slice, hash);
        }
        hash = hash_pixels(sliced.overview, hash);
        t.assert_equal("pixels of " + std::to_string(size.width) + "x" + std::to_string(size.height) + " with resize algo " + resize_algo_to_str(algo), expected, hash);
    }
}

MAKE_TEST(test_image_preprocessor_inkling2v) {
    mtmd_hparams hparams;
    hparams.patch_size = 20;
    hparams.n_merge = 2;
    for (int c = 0; c < 3; ++c) {
        hparams.image_mean[c] = hparams.image_std[c] = 0.5f;
    }
    mtmd_image_preprocessor_inkling2v preproc(hparams);
    for (const auto & size : std::vector<mtmd_image_size>{{1, 1}, {40, 40}, {61, 43}, {83, 45}, {161, 21}}) {
        mtmd_image_u8 image;
        image.set_size(size, false);
        for (int y = 0; y < size.height; ++y) {
            for (int x = 0; x < size.width; ++x) {
                image.set_pixel(x, y, {uint8_t(x % 256), uint8_t(y % 256), 255});
            }
        }
        const auto result = preproc.preprocess(image);
        t.assert_equal("one image", size_t(1), result.entries.size());
        const auto & output = result.entries[0];
        t.assert_equal("width ceil", ((size.width + 39) / 40) * 40, output.nx());
        t.assert_equal("height ceil", ((size.height + 39) / 40) * 40, output.ny());
        const auto & pixels = output.get_ro_buf();
        bool exact = true;
        for (int y = 0; y < output.ny(); ++y) {
            for (int x = 0; x < output.nx(); ++x) {
                const bool pad = x >= size.width || y >= size.height;
                const float rgb[3] = {float(x % 256), float(y % 256), 255.0f};
                for (int c = 0; c < 3; ++c) {
                    const float expected = ((pad ? -1.0f : rgb[c]) - 127.5f) / 127.5f;
                    exact = exact && pixels[3 * (size_t(y) * output.nx() + x) + c] == expected;
                }
            }
        }
        t.assert_true("pixels and negative padding", exact);
        image.set_size(size, true);
        const auto placeholder = preproc.preprocess(image);
        t.assert_true("placeholder", placeholder.entries[0].is_placeholder());
        t.assert_equal("placeholder width", output.nx(), placeholder.entries[0].nx());
        t.assert_equal("placeholder height", output.ny(), placeholder.entries[0].ny());
    }
    // an image above image_max_pixels is resized down, the output stays aligned
    hparams.image_max_pixels = 16 * 40 * 40;
    mtmd_image_u8 large;
    large.set_size({1000, 300}, false);
    large.cpy_buf(std::vector<uint8_t>(1000 * 300 * 3, 17));
    const auto capped = preproc.preprocess(large);
    t.assert_true("capped area", capped.entries[0].nx() * capped.entries[0].ny() <= hparams.image_max_pixels);
    t.assert_equal("capped width aligned", 0, capped.entries[0].nx() % 40);
    t.assert_equal("capped height aligned", 0, capped.entries[0].ny() % 40);
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
    mtmd_hparams hp;
    hp.audio_n_fft = 512; hp.audio_window_len = 400; hp.audio_hop_len = 160;
    hp.audio_sample_rate = 16000; hp.n_mel_bins = 80; hp.parakeet_mlx_frontend = true;
    hp.mel_filters = read("filters", 80 * 257); hp.window = read("window", 400);
    mtmd_audio_preprocessor_parakeet frontend(hp);
    frontend.initialize();
    const auto pcm = read("audio16", 32000);
    const auto mel_ref = read("mel", 201 * 80);
    auto preprocessed = frontend.preprocess(pcm.data(), pcm.size());
    t.assert_equal("ear frontend processes PCM", true, !preprocessed.entries.empty());
    t.assert_equal("ear frontend computes mel", true, frontend.compute(preprocessed));
    const auto & mel = preprocessed.entries.at(0);
    t.assert_equal("ear frontend frame count", 201, mel.nx());
    float mel_error = 0;
    for (size_t i = 0; i < 201; ++i) {
        for (size_t j = 0; j < 80; ++j) { mel_error = std::max(mel_error, std::abs(mel.get_ro_buf()[j * 201 + i] - mel_ref[i * 80 + j])); }
    }
    std::cout << "ear frontend max abs error: " << mel_error << "\n";
    t.assert_equal("ear frozen frontend", true, mel_error < 1e-4f);
    t.assert_equal("ear frontend rejects short PCM", false, !frontend.preprocess(pcm.data(), 256).entries.empty());
    const auto input = read("frames", 26 * 512);
    if (std::getenv("MTMD_EAR_ENCODER_TEST")) {
        mtmd_model_params params;
        auto device = mtmd_select_device(std::getenv("MTMD_EAR_GPU") != nullptr, nullptr);
        auto weight_device = device ? device : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!weight_device) { throw std::runtime_error("ear backend unavailable"); }
        auto buft = ggml_backend_dev_buffer_type(weight_device);
        auto cb_eval = [](ggml_tensor * tensor, bool ask, void * user) {
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
        auto loaded = mtmd_model_load("not-a-file.gguf", params, buft);
        if (!loaded.a) { throw std::runtime_error("cannot load native ear encoder"); }
        mtmd_backends backends(device, 4, loaded.a->max_nodes, cb_eval, const_cast<std::string *>(&dir));
        mtmd_subcontext encoder(*loaded.a, backends, { MTMD_FLASH_ATTN_TYPE_DISABLED, false });
        t.assert_equal("model reader chunks bounded", true, source.largest <= 1024 * 1024);
        auto failed_load = [&](const char * label) {
            bool rejected = false;
            try { mtmd_model_load("not-a-file.gguf", params, buft); }
            catch (const std::runtime_error &) { rejected = true; }
            t.assert_equal(label, true, rejected);
        };
        source.readable = 16;
        failed_load("model reader rejects short header");
        source.readable = full_size / 2;
        failed_load("model reader rejects truncated weights");
        source.readable = full_size;
        params.model_reader_size = 0;
        failed_load("model reader requires bounds");

        mtmd_image_f32 image;
        image.set_size({201, 80}, false, true);
        auto encoder_mel = mel.get_ro_buf();
        if (std::getenv("MTMD_EAR_FROZEN_MEL")) {
            for (size_t i = 0; i < 201; ++i) {
                for (size_t j = 0; j < 80; ++j) { encoder_mel[j * 201 + i] = mel_ref[i * 80 + j]; }
            }
        }
        image.cpy_buf(encoder_mel);
        std::vector<float> encoded(input.size());
        mtmd_image_f32_batch batch;
        batch.is_audio = true;
        batch.entries.push_back(std::move(image));
        if (!encoder.encode_batch(4, &batch, encoded)) { throw std::runtime_error("ear encoder failed"); }
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

MAKE_TEST(test_backchannel_reference) {
    const char * root = std::getenv("MTMD_BACKCHANNEL_FIXTURE");
    if (!root) { std::cout << "SKIP BC-Det parity: set MTMD_BACKCHANNEL_FIXTURE\n"; return; }
    const std::string dir(root);
    auto weights = load_component_fixture(dir + "/bc-det.gguf");
    auto params = mtmd_context_params_default();
    params.use_gpu = std::getenv("MTMD_COMPONENT_GPU");
    params.n_threads = 4;
    mtmd_backchannel detector(weights.get(), (dir + "/bc-det.gguf").c_str(), params);
    std::ifstream input(dir + "/input.f32", std::ios::binary);
    std::ifstream output(dir + "/output.f32", std::ios::binary);
    std::ifstream encoder(dir + "/encoder.f32", std::ios::binary);
    if (!input || !output || !encoder) { throw std::runtime_error("missing BC-Det fixture"); }
    std::array<float, 1280> user{}, system{};
    float worst_p = 0, worst_e = 0;
    size_t steps = 0;
    const auto start = std::chrono::steady_clock::now();
    while (input.read(reinterpret_cast<char *>(user.data()), sizeof(user))) {
        float expected;
        std::array<float, 1024> features;
        if (!input.read(reinterpret_cast<char *>(system.data()), sizeof(system)) ||
            !output.read(reinterpret_cast<char *>(&expected), sizeof(expected)) ||
            !encoder.read(reinterpret_cast<char *>(features.data()), sizeof(features))) {
            throw std::runtime_error("short BC-Det reference");
        }
        const float actual = detector.process(user, system);
        worst_p = std::max(worst_p, std::abs(expected - actual));
        const auto encoded = detector.encoded();
        for (size_t i = 0; i < features.size(); ++i) { worst_e = std::max(worst_e, std::abs(features[i] - encoded[i])); }
        if (steps < 3 || steps % 40 == 0) {
            std::cout << "bc-det step=" << steps << " p=" << actual << " expected=" << expected
                      << " max_error=" << worst_p << " encoder=" << worst_e << "\n";
        }
        ++steps;
    }
    const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cout << "bc-det steps=" << steps << " mean_pair_ms=" << ms / std::max(size_t(1), steps)
              << " max_probability_error=" << worst_p << " max_encoder_error=" << worst_e << "\n";
    t.assert_true("cross both rolling windows", steps > 250);
    t.assert_true("BC-Det probability parity", worst_p < 2e-4f);
    t.assert_true("Mimi feature parity", worst_e < 2e-3f);
    detector.reset();
    user.fill(0); system.fill(0);
    t.assert_equal("BC-Det reset skips first frame", -1.0f, detector.process(user, system));
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

#ifdef LLAMA_TEST_FRANKIE
static std::vector<char> read_fixture_bytes(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { throw std::runtime_error("missing fixture: " + path); }
    std::vector<char> bytes(size_t(file.tellg()));
    file.seekg(0);
    file.read(bytes.data(), bytes.size());
    return bytes;
}

template <typename T> static std::vector<T> read_fixture(const std::string & path) {
    const auto bytes = read_fixture_bytes(path);
    std::vector<T> values(bytes.size() / sizeof(T));
    std::memcpy(values.data(), bytes.data(), values.size() * sizeof(T));
    return values;
}

// Signal-to-error ratio of the added watermark: how closely native reproduces the reference's residual.
static double watermark_snr(const std::vector<float> & base, const std::vector<float> & expected, const std::vector<float> & actual) {
    double signal = 0, error = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        signal += double(expected[i] - base[i]) * double(expected[i] - base[i]);
        error += double(actual[i] - expected[i]) * double(actual[i] - expected[i]);
    }
    return 10 * std::log10(signal / std::max(error, 1e-30));
}

MAKE_TEST(test_watermark_reference) {
    const char * root = std::getenv("MTMD_WATERMARK_FIXTURE");
    if (!root) { std::cout << "SKIP AudioSeal parity: set MTMD_WATERMARK_FIXTURE\n"; return; }
    const std::string dir(root);
    frankie_watermark owner(dir + "/audioseal-wm16.gguf");
    const auto input = read_fixture<float>(dir + "/input.f32");
    const auto expected = read_fixture<float>(dir + "/output.f32");
    const auto marked16 = read_fixture<float>(dir + "/marked16.f32");
    const auto fir = read_fixture<double>(dir + "/fir.f64");
    auto low = read_fixture<float>(dir + "/low.f32");
    low.resize(marked16.size());  // the fixture also logs the final pending frame twice

    // The FIR design matches scipy.signal.firwin.
    frankie_causal_resampler resampler(2, 3);
    double fir_error = 0;
    for (size_t i = 0; i < fir.size(); ++i) { fir_error = std::max(fir_error, std::abs(fir[i] - resampler.filter()[i])); }
    t.assert_true("FIR taps match firwin", fir.size() == 61 && fir_error < 1e-15);

    // The 16 kHz generator alone, frame by frame, against the reference model's output.
    auto generate = [&](uint16_t payload) {
        auto ggml = load_component_fixture(dir + "/audioseal-wm16.gguf");
        mtmd_watermark model(ggml.get(), false);
        const auto message = model.message(payload);
        auto state = model.initial_state();
        std::vector<float> out(low.size()), frame(mtmd_watermark::frame);
        for (size_t i = 0; i < low.size(); i += mtmd_watermark::frame) {
            const size_t n = std::min<size_t>(mtmd_watermark::frame, low.size() - i);
            std::fill(frame.begin(), frame.end(), 0.0f);
            std::copy(low.begin() + i, low.begin() + i + n, frame.begin());
            model.process(state, message, frame.data(), frame.data());
            std::copy(frame.begin(), frame.begin() + n, out.begin() + i);
        }
        return out;
    };
    const auto start = std::chrono::steady_clock::now();
    const auto native16 = generate(frankie_watermark_payload);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const double generator_snr = watermark_snr(low, marked16, native16);
    const double wrong_snr = watermark_snr(low, marked16, generate(0x0000));
    std::cout << "audioseal frames=" << (low.size() + 319) / 320 << " ms_per_frame=" << ms / ((low.size() + 319) / 320)
              << " generator_snr_db=" << generator_snr << " wrong_payload_snr_db=" << wrong_snr << "\n";
    t.assert_true("generator parity", generator_snr > 60);
    t.assert_true("wrong payload is caught", wrong_snr < 20);

    // The whole residual stream on 24 kHz speech, in the reference's 1920-sample chunks
    // and in irregular ones: same output, same sample count.
    auto run = [&](const std::vector<size_t> & sizes) {
        frankie_watermark::stream stream(owner);
        std::vector<float> out;
        size_t i = 0, k = 0;
        while (i < input.size()) {
            const size_t n = std::min(sizes[k++ % sizes.size()], input.size() - i);
            const auto y = stream.push(input.data() + i, n);
            out.insert(out.end(), y.begin(), y.end());
            t.assert_true("output never leads input", stream.output_samples() <= stream.input_samples());
            i += n;
        }
        const auto tail = stream.finish();
        out.insert(out.end(), tail.begin(), tail.end());
        return out;
    };
    const auto native = run({1920});
    t.assert_equal("residual conserves samples", input.size(), native.size());
    const double residual_snr = watermark_snr(input, expected, native);
    std::cout << "audioseal residual samples=" << native.size() << " residual_snr_db=" << residual_snr << "\n";
    t.assert_true("residual parity", residual_snr > 60);
    const auto irregular = run({1, 319, 1921, 7, 480, 4096});
    t.assert_true("chunking invariant", irregular == native);
    if (const char * dump = std::getenv("MTMD_WATERMARK_DUMP")) {
        std::ofstream(dump, std::ios::binary).write(reinterpret_cast<const char *>(native.data()), native.size() * sizeof(float));
    }

    // Cancellation is destruction: a fresh stream starts from zero history.
    { frankie_watermark::stream abandoned(owner); abandoned.push(input.data(), 5000); }
    t.assert_true("fresh stream after abandon", run({1920}) == native);
    frankie_watermark::stream closed(owner);
    closed.finish();
    bool rejected = false;
    try { closed.push(input.data(), 1); } catch (const std::runtime_error &) { rejected = true; }
    t.assert_true("finished stream rejects audio", rejected);
}
#endif

//
// mtmd_helper video
//

MAKE_TEST(test_video_config) {
    mtmd_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    const mtmd_image_preprocessor_qwen3vl preproc(hparams);

    // { template, n_frames and fps of the source, expected fps, n_frames and frame size }
    // the 1920x1080 sizes are from Qwen3VLVideoProcessor smart_resize
    const std::vector<std::tuple<mtmd_video_template, int32_t, float, float, int32_t, mtmd_image_size>> cases = {
        { MTMD_VIDEO_TEMPLATE_QWEN3VL,   36000, 30.0f, 2.0f, 768, { 224, 128 } },
        { MTMD_VIDEO_TEMPLATE_LFM3,      36000, 30.0f, 2.0f, 768, { 224, 128 } },
        { MTMD_VIDEO_TEMPLATE_LFM3_OMNI, 36000, 30.0f, 2.0f, 384, { 320, 192 } },
        { MTMD_VIDEO_TEMPLATE_QWEN3VL,      -1, 30.0f, 2.0f,   0, {   0,   0 } }, // frame count unknown
        { MTMD_VIDEO_TEMPLATE_DEFAULT,   36000, 30.0f, 0.0f,   0, {   0,   0 } },
    };

    for (const auto & [tmpl, n_frames, fps, expected_fps, expected_n_frames, expected_size] : cases) {
        const mtmd_video_config cfg = mtmd_video_config_init(tmpl, &preproc, 1920, 1080, n_frames, fps);
        const std::string name = "template " + std::to_string(tmpl) + " with " + std::to_string(n_frames) + " frames";
        t.assert_equal(name + " template", (int) tmpl, (int) cfg.tmpl);
        t.assert_equal(name + " fps", expected_fps, cfg.fps);
        t.assert_equal(name + " n_frames", expected_n_frames, cfg.n_frames);
        t.assert_equal(name + " frame size", image_size_to_str(expected_size), image_size_to_str({ cfg.frame_width, cfg.frame_height }));
    }
}

// n_frames source frames, each one starts with its index
struct test_video_source : mtmd_helper_video_seq {
    int32_t n_frames;
    int32_t n_read    = 0;
    int32_t n_skipped = 0;

    test_video_source(int32_t n_frames) : n_frames(n_frames) {}

    bool skip_raw_frame() override {
        if (n_read >= n_frames) {
            return false;
        }
        n_read++;
        n_skipped++;
        return true;
    }

    bool read_raw_frame() override {
        if (n_read >= n_frames) {
            return false;
        }
        frame_buf[0] = static_cast<uint8_t>(n_read & 255);
        frame_buf[1] = static_cast<uint8_t>(n_read >> 8);
        n_read++;
        return true;
    }
};

// text as is, a frame as [source index], an audio bitmap as {n_samples}
static std::string video_sequence_to_str(testing & t, mtmd_helper_video_seq & video) {
    std::string out;
    mtmd_bitmap * bitmap;
    char * text;
    int32_t ret;
    while ((ret = video.read_next(&bitmap, &text)) == 0) {
        if (text) {
            out += text;
            free(text);
        } else if (mtmd_bitmap_is_audio(bitmap)) {
            out += "{" + std::to_string(mtmd_bitmap_get_n_bytes(bitmap) / sizeof(float)) + "}";
            mtmd_bitmap_free(bitmap);
        } else {
            const unsigned char * data = mtmd_bitmap_get_data(bitmap);
            out += "[" + std::to_string(data[0] | data[1] << 8) + "]";
            mtmd_bitmap_free(bitmap);
        }
    }
    t.assert_equal("end of stream", -1, ret);
    return out;
}

MAKE_TEST(test_video_sequence_qwen3vl) {
    mtmd_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    const mtmd_image_preprocessor_qwen3vl preproc(hparams);

    // { source video, expected frame size and sequence from Qwen3VLProcessor (transformers 4.57.1) }
    const std::vector<std::tuple<mtmd_helper_video_stream_info, mtmd_image_size, std::string>> cases = {
        { {  720, 358, 24.0f, 240, -1.0f }, { 704, 352 },
            "<0.3 seconds>[0][13]<1.3 seconds>[25][38]<2.4 seconds>[50][63]<3.4 seconds>[75][88]<4.5 seconds>[101][113]"
            "<5.5 seconds>[126][138]<6.6 seconds>[151][164]<7.6 seconds>[176][189]<8.6 seconds>[201][214]<9.7 seconds>[226][239]" },
        // odd frame count: the last group repeats its last frame
        { {  640, 480, 10.0f,  35, -1.0f }, { 640, 480 },
            "<0.3 seconds>[0][6]<1.4 seconds>[11][17]<2.5 seconds>[23][28]<3.4 seconds>[34]" },
        // fewer frames than the minimum of 4
        { {  320, 240, 30.0f,   3, -1.0f }, { 320, 256 },
            "<0.0 seconds>[0][1]<0.1 seconds>[2]" },
        { { 1280, 720, 30000.0f / 1001.0f, 300, -1.0f }, { 1280, 704 },
            "<0.3 seconds>[0][16]<1.3 seconds>[31][47]<2.4 seconds>[63][79]<3.4 seconds>[94][110]<4.5 seconds>[126][142]"
            "<5.5 seconds>[157][173]<6.6 seconds>[189][205]<7.6 seconds>[220][236]<8.7 seconds>[252][268]<9.7 seconds>[283][299]" },
    };

    for (const auto & [src, expected_size, expected] : cases) {
        const std::string name = std::to_string(src.n_frames) + " frames of " + std::to_string(src.width) + "x" + std::to_string(src.height);
        const mtmd_video_config cfg = mtmd_video_config_init(MTMD_VIDEO_TEMPLATE_QWEN3VL, &preproc, src.width, src.height, src.n_frames, src.fps);
        test_video_source video(src.n_frames);
        video.configure(cfg, src, 0.0f);
        video.src_alive = true;
        t.assert_equal(name + " frame size", image_size_to_str(expected_size), image_size_to_str({ (int) video.info.width, (int) video.info.height }));
        t.assert_equal(name + " sequence", expected, video_sequence_to_str(t, video));
    }
}

// the soundtrack of each group comes after its frames
MAKE_TEST(test_video_sequence_lfm3_omni) {
    mtmd_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    const mtmd_image_preprocessor_lfm3vl preproc(hparams);

    const mtmd_helper_video_stream_info src = { 64, 64, 2.0f, 9, -1.0f };
    const mtmd_video_config cfg = mtmd_video_config_init(MTMD_VIDEO_TEMPLATE_LFM3_OMNI, &preproc, src.width, src.height, src.n_frames, src.fps);
    test_video_source video(src.n_frames);
    video.configure(cfg, src, 0.0f);
    video.src_alive = true;
    // the video is 4.5 seconds long, the rest of the soundtrack is dropped
    const std::vector<float> pcm(500, 0.0f);
    video.set_soundtrack(pcm.data(), pcm.size(), 100);

    t.assert_equal("sequence",
        std::string("<0.2 seconds>[0][1]{100}<1.2 seconds>[2][3]{100}<2.2 seconds>[4][5]{100}<3.2 seconds>[6][7]{100}<4.0 seconds>[8]{50}"),
        video_sequence_to_str(t, video));
}

// a source that gives every frame: the frames of fps_target are picked by index
MAKE_TEST(test_video_sequence_sample_by_index) {
    const mtmd_video_config cfg = { MTMD_VIDEO_TEMPLATE_DEFAULT, 0.0f, 0, 0, 0 };

    // { source video, frames in the source, expected sequence, expected number of skipped frames }
    const std::vector<std::tuple<mtmd_helper_video_stream_info, int32_t, std::string, int32_t>> cases = {
        { { 64, 64, 8.0f, 16, -1.0f }, 16, "Video:[0][0m0.00s][4][8][0m1.00s][12]", 9 },
        // unknown length: every frame is kept
        { { 64, 64, 8.0f, -1, -1.0f },  3, "Video:[0][0m0.00s][1][2]", 0 },
    };

    for (const auto & [src, n_frames, expected, expected_skipped] : cases) {
        const std::string name = std::to_string(src.n_frames) + " frames";
        test_video_source video(n_frames);
        video.timestamp_interval_ms = 1000;
        video.configure(cfg, src, 2.0f);
        video.sample_by_index(src);
        video.src_alive = true;
        t.assert_equal(name + " sequence", expected, video_sequence_to_str(t, video));
        t.assert_equal(name + " skipped", expected_skipped, video.n_skipped);
    }
}

//
// mtmd_audio
//

MAKE_TEST(test_audio_preprocessor_whisper_chunks) {
    mtmd_hparams hparams;
    hparams.n_mel_bins        = 128;
    hparams.audio_sample_rate = 16000;
    hparams.audio_n_fft       = 400;
    hparams.audio_window_len  = 400;
    hparams.audio_hop_len     = 160;

    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    ggml_backend_t backends[] = { backend.get() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 256, false, true));

    mtmd_audio_preprocessor_whisper preproc(hparams, sched.get());
    preproc.initialize();

    // { sample count, expected chunks }, one chunk of 3000 frames covers 480000 samples
    const std::vector<std::pair<size_t, size_t>> cases = {
        {       0, 0 },
        {       1, 1 },
        {     200, 1 },
        {     201, 1 },
        {   32000, 1 },
        {  479834, 1 },
        {  479839, 1 },
        {  479840, 1 },
        {  479999, 1 },
        {  480000, 1 },
        {  480001, 2 },
        {  480159, 2 },
        {  480160, 2 },
        {  959840, 2 },
        {  959999, 2 },
        {  960000, 2 },
        {  960001, 3 },
        { 1440000, 3 },
    };

    for (const auto & [n_samples, expected] : cases) {
        const std::string name = std::to_string(n_samples) + " samples";
        std::vector<float> samples(n_samples, 0.0f);
        if (n_samples > 0) {
            samples.back() = 0.5f;
        }
        mtmd_audio_preproc_out out = preproc.preprocess(samples.data(), samples.size());
        t.assert_equal("chunks for " + name, expected, out.entries.size());
        for (const auto & mel : out.entries) {
            t.assert_equal("frames for " + name, 3000, mel.nx());
            t.assert_equal("mel bins for " + name, 128, mel.ny());
        }
    }
}

MAKE_TEST(test_audio_preprocessor_qwen3a_tail) {
    mtmd_hparams hparams;
    hparams.n_mel_bins        = 128;
    hparams.audio_sample_rate = 16000;
    hparams.audio_n_fft       = 400;
    hparams.audio_window_len  = 400;
    hparams.audio_hop_len     = 160;

    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    ggml_backend_t backends[] = { backend.get() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 256, false, true));

    mtmd_audio_preprocessor_qwen3a preproc(hparams, sched.get());
    preproc.initialize();

    // one 800-frame window covers 128000 samples (hop 160), so longer audio is split;
    // the last window is emitted with its exact frame count, the graph pads the final 100-frame chunk
    // { sample count, expected n_len of each output window }
    const std::vector<std::pair<size_t, std::vector<int64_t>>> cases = {
        {       0, {}          },
        {       1, {1}         },
        {   32000, {200}       },
        {  128000, {800}       },
        {  128160, {800, 1}    },
        {  168000, {800, 250}  },
        {  170080, {800, 263}  },
        {  176000, {800, 300}  },
    };

    for (const auto & [n_samples, expected] : cases) {
        const std::string name = std::to_string(n_samples) + " samples";
        std::vector<float> samples(n_samples, 0.0f);
        if (n_samples > 0) {
            samples.back() = 0.5f;
        }
        mtmd_audio_preproc_out out = preproc.preprocess(samples.data(), samples.size());
        t.assert_equal("windows for " + name, expected.size(), out.entries.size());
        for (size_t i = 0; i < out.entries.size(); i++) {
            t.assert_equal("frames for " + name, expected[i], out.entries[i].nx());
            t.assert_equal("mel bins for " + name, 128, out.entries[i].ny());
        }
    }
}

MAKE_TEST(test_audio_preprocessor_conformer) {
    mtmd_hparams hparams;
    hparams.n_mel_bins        = 128;
    hparams.audio_sample_rate = 16000;
    hparams.audio_n_fft       = 512;
    hparams.audio_window_len  = 400;
    hparams.audio_hop_len     = 160;

    // 0.4 s of tones, 0.2 s of silence, 0.4 s of a quiet tone
    const double pi = 3.14159265358979323846;
    const int sr = hparams.audio_sample_rate;
    std::vector<float> samples(sr, 0.0f);
    for (int i = 0; i < sr; i++) {
        const double ts = (double) i / sr;
        if (i < 0.4 * sr) {
            samples[i] = (float) (0.3 * std::sin(2 * pi * 300 * ts) + 0.2 * std::sin(2 * pi * 1200 * ts) + 0.1 * std::sin(2 * pi * 3500 * ts));
        } else if (i >= 0.6 * sr) {
            samples[i] = (float) (1e-3 * std::sin(2 * pi * 800 * ts));
        }
    }

    // reference values from NeMo AudioToMelSpectrogramPreprocessor (liquid-audio 1.3.0)
    // { mel bin, frame, value }
    const std::vector<std::tuple<int, int, float>> cases = {
        {   8,  0,  3.6104f }, // tones
        {  24, 50, -0.6371f }, // silence
        {  24, 60, -0.0476f }, // silence -> quiet tone
        { 100, 20, -0.0913f }, // tones, high bin
    };

    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    ggml_backend_t backends[] = { backend.get() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 256, false, true));

    // the result must not depend on the number of threads
    for (int n_threads : { 1, 4, 7 }) {
        mtmd_audio_preprocessor_conformer preproc(hparams, sched.get());
        preproc.mel_config.n_threads = n_threads;
        preproc.initialize();
        mtmd_audio_preproc_out out = preproc.preprocess(samples.data(), samples.size());
        if (!t.assert_true("preprocess", !out.entries.empty() && preproc.compute(out))) {
            return;
        }
        const mtmd_image_f32 & mel = out.entries[0];
        t.assert_equal("n_len", 101, mel.nx());

        for (const auto & [bin, frame, expected] : cases) {
            const float actual = mel.get_ro_buf()[bin * mel.nx() + frame];
            t.assert_true("mel[" + std::to_string(bin) + "][" + std::to_string(frame) + "] = " + std::to_string(actual) + ", expected " + std::to_string(expected), std::fabs(actual - expected) < 1e-3f);
        }
    }
}

MAKE_TEST(test_audio_preprocessor_threads) {
    mtmd_hparams hparams;
    hparams.n_mel_bins        = 128;
    hparams.audio_sample_rate = 16000;
    hparams.audio_n_fft       = 512;
    hparams.audio_window_len  = 400;
    hparams.audio_hop_len     = 160;

    // chirp: every frame has a different content
    std::vector<float> samples(hparams.audio_sample_rate * 5);
    for (size_t i = 0; i < samples.size(); i++) {
        samples[i] = (float) (0.3 * std::sin(0.05 * i + 2e-6 * i * i));
    }

    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    ggml_backend_t backends[] = { backend.get() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 256, false, true));

    mtmd_audio_preprocessor_conformer preproc(hparams, sched.get());
    preproc.initialize();
    // tokenize only gets the shapes, encode must get the same ones
    mtmd_audio_preproc_out ref = preproc.preprocess(samples.data(), samples.size());
    if (!t.assert_true("preprocess", ref.entries.size() == 1 && ref.needs_compute())) {
        return;
    }
    const mtmd_image_size shape = ref.entries[0].get_size();
    if (!t.assert_true("compute", preproc.compute(ref))) {
        return;
    }
    t.assert_true("shape", ref.entries.size() == 1 && ref.entries[0].get_size() == shape);
    t.assert_true("computed", !ref.needs_compute());

    // frames are independent, so the whole output must be bit-exact with the single-threaded one
    for (int n_threads : { 4, 12 }) {
        preproc.mel_config.n_threads = n_threads;
        mtmd_audio_preproc_out out = preproc.preprocess(samples.data(), samples.size());
        const bool ok = preproc.compute(out);
        t.assert_true("n_threads = " + std::to_string(n_threads), ok && out.entries[0].get_ro_buf() == ref.entries[0].get_ro_buf());
    }
}

MAKE_TEST(test_audio_log_mel_precision) {
    const int n_mel    = 80;
    const int n_fft    = 400;
    const int hop      = 160;
    const int n_frames = 200;

    mtmd_audio_cache cache;
    cache.fill_sin_cos_table(n_fft);
    cache.fill_hann_window(n_fft, true);
    cache.fill_mel_filterbank_matrix(n_mel, n_fft, 16000);

    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    ggml_backend_t backends[] = { backend.get() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 256, false, true));

    mtmd_audio_mel_config config;
    config.sched      = sched.get();
    config.n_threads  = 2;
    config.n_mel      = n_mel;
    config.n_fft_bins = 1 + n_fft / 2;
    config.hop_length = hop;
    config.cache      = &cache;

    // a loud tone next to a 74 dB weaker one: rounding the DFT inputs to F16 or TF32 loses the weak one
    const double pi = 3.14159265358979323846;
    std::vector<float> samples((size_t) (n_frames - 1) * hop + n_fft);
    for (size_t i = 0; i < samples.size(); i++) {
        samples[i] = (float) (0.5 * std::sin(2 * pi * 200 * i / 16000) + 1e-4 * std::sin(2 * pi * 6000 * i / 16000));
    }

    std::vector<float> out((size_t) n_frames * n_mel);
    if (!t.assert_true("log mel", mtmd_audio_log_mel(config, samples.data(), cache.hann_window.data(), n_frames, out.data()))) {
        return;
    }

    const int n_bins = n_fft / 2 + 1;
    double max_diff = 0.0;
    std::vector<double> power(n_bins);
    for (int i = 0; i < n_frames; i++) {
        for (int k = 0; k < n_bins; k++) {
            double re = 0.0, im = 0.0;
            for (int j = 0; j < n_fft; j++) {
                const double x = (double) cache.hann_window[j] * samples[(size_t) i * hop + j];
                re += x * std::cos(2 * pi * j * k / n_fft);
                im -= x * std::sin(2 * pi * j * k / n_fft);
            }
            power[k] = re * re + im * im;
        }
        for (int m = 0; m < n_mel; m++) {
            double sum = 0.0;
            for (int k = 0; k < n_bins; k++) {
                sum += power[k] * cache.filters.data[(size_t) m * n_bins + k];
            }
            const double expected = std::log10(std::max(sum, (double) config.mel_floor));
            max_diff = std::max(max_diff, std::fabs(expected - out[(size_t) m * n_frames + i]));
        }
    }
    t.assert_true("max_diff = " + std::to_string(max_diff), max_diff < 0.05);
}

MAKE_TEST(test_audio_log_mel_blocks) {
    const int n_mel    = 80;
    const int n_fft    = 400;
    const int hop      = 160;
    const int n_frames = 8192 + 100; // long audio is split into blocks of 8192 frames

    mtmd_audio_cache cache;
    cache.fill_sin_cos_table(n_fft);
    cache.fill_hann_window(n_fft, true);
    cache.fill_mel_filterbank_matrix(n_mel, n_fft, 16000);

    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    ggml_backend_t backends[] = { backend.get() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 256, false, true));

    mtmd_audio_mel_config config;
    config.sched       = sched.get();
    config.n_threads   = 2;
    config.n_mel       = n_mel;
    config.n_fft_bins  = 1 + n_fft / 2;
    config.hop_length  = hop;
    config.cache       = &cache;

    std::vector<float> samples((size_t) (n_frames - 1) * hop + n_fft);
    for (size_t i = 0; i < samples.size(); i++) {
        samples[i] = (float) (0.3 * std::sin(0.05 * i + 1e-6 * i * i));
    }

    // frames are independent: the frames around the seam must not change when computed on their own
    // not bit-exact, matmul kernels may round differently depending on the shape
    const int off   = 8192 - 50;
    const int n_sub = n_frames - off;
    std::vector<float> all((size_t) n_frames * n_mel);
    std::vector<float> sub((size_t) n_sub * n_mel);
    const float * window = cache.hann_window.data();
    if (!t.assert_true("all", mtmd_audio_log_mel(config, samples.data(), window, n_frames, all.data())) ||
        !t.assert_true("sub", mtmd_audio_log_mel(config, samples.data() + (size_t) off * hop, window, n_sub, sub.data()))) {
        return;
    }

    float max_diff = 0.0f;
    for (int j = 0; j < n_mel; j++) {
        for (int i = 0; i < n_sub; i++) {
            max_diff = std::max(max_diff, std::fabs(all[(size_t) j * n_frames + off + i] - sub[(size_t) j * n_sub + i]));
        }
    }
    t.assert_true("max_diff = " + std::to_string(max_diff), max_diff < 1e-5f);
}

//
// mtmd_helper_resampler
//

static std::vector<float> make_tone(double freq, int rate, size_t n) {
    const double pi = 3.14159265358979323846;
    std::vector<float> x(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = (float) (0.5 * std::sin(2 * pi * freq * i / rate));
    }
    return x;
}

// amplitude of the freq component, skipping the edges
static double tone_amplitude(const std::vector<float> & x, double freq, int rate) {
    const double pi = 3.14159265358979323846;
    const size_t skip = rate / 10;
    double re = 0.0, im = 0.0;
    for (size_t i = skip; i < x.size() - skip; i++) {
        re += x[i] * std::cos(2 * pi * freq * i / rate);
        im += x[i] * std::sin(2 * pi * freq * i / rate);
    }
    return 2.0 * std::sqrt(re * re + im * im) / (x.size() - 2 * skip);
}

MAKE_TEST(test_audio_fft) {
    // the sizes in use: powers of two, and an odd part (400 = 16 * 25, 736 = 32 * 23, 1920 = 128 * 15)
    // plus small edge cases: 4 = gather + one stage, 6 = n_odd 3 with n_sub 2, 15 = odd composite with n_sub 1
    const int sizes[] = { 1, 2, 3, 4, 6, 12, 15, 256, 400, 512, 736, 1920 };
    const double pi = 3.14159265358979323846;
    for (int n : sizes) {
        mtmd_audio_cache cache;
        cache.fill_sin_cos_table(n);
        // phase offset so that x[0] != 0 (n = 1 would otherwise compare all zeros)
        std::vector<float> x(2 * n);
        for (int i = 0; i < 2 * n; i++) {
            x[i] = (float) std::sin(0.37 * i + 0.001 * i * i + 0.5);
        }
        std::vector<float> out(8 * n);
        for (bool inv : { false, true }) {
            // input sized exactly as documented so that sanitizers catch any overread
            const std::vector<float> in(x.begin(), x.begin() + (inv ? 2 * n : n));
            if (inv) {
                mtmd_audio_ifft(cache, in.data(), n, out.data());
            } else {
                mtmd_audio_fft(cache, in.data(), n, out.data());
            }
            double max_err = 0.0, max_ref = 0.0;
            for (int k = 0; k < n; k++) {
                double re = 0.0, im = 0.0;
                for (int j = 0; j < n; j++) {
                    const double theta = (inv ? 2 : -2) * pi * (double) ((int64_t) j * k % n) / n;
                    const double a = inv ? x[2 * j] : x[j];
                    const double b = inv ? x[2 * j + 1] : 0.0;
                    re += a * std::cos(theta) - b * std::sin(theta);
                    im += a * std::sin(theta) + b * std::cos(theta);
                }
                if (inv) {
                    re /= n;
                    im /= n;
                }
                max_err = std::max({ max_err, std::fabs(re - out[2 * k]), std::fabs(im - out[2 * k + 1]) });
                max_ref = std::max({ max_ref, std::fabs(re), std::fabs(im) });
            }
            t.assert_true(std::string(inv ? "ifft" : "fft") + " n = " + std::to_string(n) + ", relative error " + std::to_string(max_err / max_ref),
                          max_ref > 0.0 && max_err <= 1e-5 * max_ref);
        }

        // round trip: ifft(fft(x)) == x for real x
        std::vector<float> spec(8 * n);
        mtmd_audio_fft(cache, x.data(), n, spec.data());
        const std::vector<float> spec_in(spec.begin(), spec.begin() + 2 * n);
        mtmd_audio_ifft(cache, spec_in.data(), n, out.data());
        double max_err = 0.0, max_ref = 0.0;
        for (int k = 0; k < n; k++) {
            max_err = std::max({ max_err, (double) std::fabs(out[2 * k] - x[k]), (double) std::fabs(out[2 * k + 1]) });
            max_ref = std::max(max_ref, (double) std::fabs(x[k]));
        }
        t.assert_true("fft round trip n = " + std::to_string(n) + ", relative error " + std::to_string(max_err / max_ref),
                      max_ref > 0.0 && max_err <= 1e-5 * max_ref);
    }
}

static std::vector<float> resample_chunked(mtmd_helper_resampler * rs, const std::vector<float> & in, size_t chunk) {
    std::vector<float> res;
    for (size_t off = 0; off < in.size(); off += chunk) {
        const float * out = nullptr;
        const int64_t n = mtmd_helper_resampler_process(rs, in.data() + off, std::min(chunk, in.size() - off), &out);
        if (n < 0) {
            return {};
        }
        res.insert(res.end(), out, out + n);
    }
    return res;
}

MAKE_TEST(test_helper_resampler) {
    const int rate_out = 16000;
    for (int rate_in : { 8000, 24000, 44100, 48000 }) {
        const std::string name = std::to_string(rate_in) + " -> " + std::to_string(rate_out);
        const std::vector<float> in = make_tone(440, rate_in, (size_t) rate_in * 2);

        mtmd_helper::resampler_ptr rs_all(mtmd_helper_resampler_init(rate_in, rate_out));
        mtmd_helper::resampler_ptr rs_chk(mtmd_helper_resampler_init(rate_in, rate_out));
        if (!t.assert_true(name + " init", rs_all && rs_chk)) {
            continue;
        }
        const std::vector<float> all = resample_chunked(rs_all.get(), in, in.size());
        const std::vector<float> chk = resample_chunked(rs_chk.get(), in, 1234); // odd size, not a multiple of the ratio

        t.assert_equal(name + " n_samples", (size_t) rate_out * 2, all.size());
        t.assert_true(name + " chunked == one-shot", all == chk);

        const double amp = tone_amplitude(all, 440, rate_out);
        t.assert_true(name + " amplitude = " + std::to_string(amp), std::fabs(amp - 0.5) < 0.02);
    }

    // above the output Nyquist: 10 kHz at 48 kHz aliases to 6 kHz at 16 kHz, the low-pass filter must cut it
    {
        mtmd_helper::resampler_ptr rs(mtmd_helper_resampler_init(48000, rate_out));
        const std::vector<float> out = resample_chunked(rs.get(), make_tone(10000, 48000, 48000 * 2), 4800);
        const double db = 20 * std::log10(tone_amplitude(out, 6000, rate_out) / 0.5);
        t.assert_true("aliasing = " + std::to_string(db) + " dB", db < -15); // about -20 dB with lpfOrder = 8, -10 dB with the default
    }

    // same rate: no copy
    {
        mtmd_helper::resampler_ptr rs(mtmd_helper_resampler_init(rate_out, rate_out));
        const std::vector<float> in = make_tone(440, rate_out, 1000);
        const float * out = nullptr;
        const int64_t n = rs ? mtmd_helper_resampler_process(rs.get(), in.data(), in.size(), &out) : -1;
        t.assert_true("passthrough", n == (int64_t) in.size() && out == in.data());
    }

    t.assert_true("invalid rate", mtmd_helper_resampler_init(0, rate_out) == nullptr);
}

//
// kokoro
//

// a dictionary for tts_dict(): entries in any order, sorted here
static common_tts_dict kokoro_dict(std::vector<std::pair<std::string, std::string>> entries) {
    std::sort(entries.begin(), entries.end());
    common_tts_dict dict;
    for (auto & e : entries) {
        dict.keys.push_back(e.first);
        dict.values.push_back(e.second);
    }
    return dict;
}

// the "tts" template with small dictionaries: a few lexicon words, fixed tags, and a model for unknown words over the letters z a b i n g
MAKE_TEST(test_kokoro_tts_template) {
    std::string path = __FILE__;
    const size_t pos = path.find_last_of("/\\");
    path = (pos == std::string::npos ? std::string(".") : path.substr(0, pos)) + "/../models/templates/hexgrad-Kokoro-82M-tts.jinja";
    std::ifstream fs(path, std::ios_base::binary);
    if (!t.assert_true("template file is readable: " + path, fs.is_open())) {
        return;
    }
    const std::string tmpl((std::istreambuf_iterator<char>(fs)), std::istreambuf_iterator<char>());

    const auto dicts = std::make_shared<const common_tts_dicts>(common_tts_dicts{
        {"gold", kokoro_dict({
            {"hello", "\"h\xc9\x99l\xcb\x88O\""},
            {"world", "\"w\xcb\x88\xc9\x9c\xc9\xb9ld\""},
            {"I", "\"\xcb\x88I\""},
            {"read", "{\"ADJ\":\"\xc9\xb9\xcb\x88\xc9\x9b" "d\",\"DEFAULT\":\"\xc9\xb9\xcb\x88id\",\"VBD\":\"\xc9\xb9\xcb\x88\xc9\x9b" "d\",\"VBN\":\"\xc9\xb9\xcb\x88\xc9\x9b" "d\",\"VBP\":\"\xc9\xb9\xcb\x88\xc9\x9b" "d\"}"},
            {"it", "\"\xc9\xaat\""},
            {"three", "\"\xce\xb8\xc9\xb9\xcb\x88i\""},
            {"cat", "\"k\xcb\x88\xc3\xa6t\""},
            {"walk", "\"w\xcb\x88\xc9\x94k\""},
            {"the", "\"\xc3\xb0i\""},
            {"Z", "\"z\xcb\x88i\""},
            {"'s", "\"s\""},
            {"a", "\"A\""},
            {"and", "\"\xc3\xa6nd\""},
            {"is", "\"\xc9\xaaz\""},
        })},
        {"silver", kokoro_dict({
        })},
        {"gold_gb", kokoro_dict({
            {"hello", "\"h\xc9\x99l\xcb\x88Q\""},
            {"world", "\"w\xcb\x88\xc9\x9c\xcb\x90ld\""},
            {"walk", "\"w\xcb\x88\xc9\x94\xcb\x90k\""},
        })},
        {"silver_gb", kokoro_dict({
        })},
        {"pos_classes", kokoro_dict({
            {"", "[\"NN\",\"NNP\",\"PRP\",\"VBD\"]"},
        })},
        {"pos_tagdict", kokoro_dict({
            {"I", "\"PRP\""},
            {"read", "\"VBD\""},
            {"it", "\"PRP\""},
            {".", "\".\""},
            {",", "\",\""},
        })},
        {"pos_weights", kokoro_dict({
            {"bias", "{\"c0\":1.0}"},
        })},
        {"lts", kokoro_dict({
            {"z", "{\"z\":0.1}"},
            {"b", "{\"b\":0.1}"},
            {"i", "{\"\xc9\xaa\":0.1}"},
            {"n", "{\"\xc5\x8b\":0.1}"},
            {"g", "{\"\":0.1}"},
            {"a", "{\"\xcb\x88\xc3\xa6\":0.1,\"\xc3\xa6\":0.06}"},
            {"z:z a", "{\"\xcb\x88\xc3\xa6\":0.2,\"\xc3\xa6\":0.01}"},
            {"$", "{\"\":0.2}"},
        })},
        {"lts_gb", kokoro_dict({
            {"z", "{\"z\":0.1}"},
            {"b", "{\"b\":0.1}"},
            {"i", "{\"\xc9\xaa\":0.1}"},
            {"n", "{\"\xc5\x8b\":0.1}"},
            {"g", "{\"\":0.1}"},
            {"a", "{\"\xcb\x88\xc9\x91\xcb\x90\":0.1,\"\xc9\x91\xcb\x90\":0.06}"},
            {"z:z a", "{\"\xcb\x88\xc9\x91\xcb\x90\":0.2,\"\xc9\x91\xcb\x90\":0.01}"},
            {"$", "{\"\":0.2}"},
        })},
    });

    // voice, text, expected phonemes
    const std::vector<std::array<std::string, 3>> cases = {
        // words of the lexicon
        { "af_heart", "Hello world.", "h\xc9\x99l\xcb\x88O w\xcb\x88\xc9\x9c\xc9\xb9ld." },
        // the pronunciation depends on the part of speech
        { "af_heart", "I read it.", "\xcb\x8cI \xc9\xb9\xcb\x88\xc9\x9b" "d \xc9\xaat." },
        // a number, and the -s suffix rule
        { "af_heart", "3 cats", "\xce\xb8\xc9\xb9\xcb\x88i k\xcb\x88\xc3\xa6ts" },
        // the -ed and -ing suffix rules
        { "af_heart", "walked, walking", "w\xcb\x88\xc9\x94kt, w\xcb\x88\xc9\x94k\xc9\xaa\xc5\x8b" },
        // not in the lexicon: predicted from the spelling
        { "af_heart", "zab", "z\xcb\x88\xc3\xa6" "b" },
        // the free search gives two primary stresses, the stress pass keeps one
        { "af_heart", "zaba", "z\xcb\x88\xc3\xa6" "b\xc3\xa6" },
        // a doubled consonant says one sound
        { "af_heart", "zabbing", "z\xcb\x88\xc3\xa6" "b\xc9\xaa\xc5\x8b" },
        // an unknown word with 's
        { "af_heart", "The zab's cat", "\xc3\xb0\xc9\x99 z\xcb\x88\xc3\xa6" "bs k\xcb\x88\xc3\xa6t" },
        // no vowel: spelled out
        { "af_heart", "zz", "z\xcb\x8ciz\xcb\x88i" },
        // an accented letter is read as its base letter
        { "af_heart", "H\xc3\xa9ll\xc3\xb3 world.", "h\xc9\x99l\xcb\x88O w\xcb\x88\xc9\x9c\xc9\xb9ld." },
        // digits glued to letters are read as a number
        { "af_heart", "3Z cats", "\xce\xb8\xc9\xb9\xcb\x88i z\xcb\x88i k\xcb\x88\xc3\xa6ts" },
        // an unknown word in mixed case is read part by part
        { "af_heart", "catWalk", "k\xcb\x88\xc3\xa6tw\xcb\x88\xc9\x94k" },
        // a british voice uses the british lexicon
        { "bf_emma", "Hello world.", "h\xc9\x99l\xcb\x88Q w\xcb\x88\xc9\x9c\xcb\x90ld." },
        // and the british model for unknown words
        { "bf_emma", "zaba", "z\xcb\x88\xc9\x91\xcb\x90" "b\xc9\x91\xcb\x90" },
        // a voice of another language takes phonemes, they pass through
        { "ef_dora", "\xcb\x88ola, m\xcb\x88undo.", "\xcb\x88ola, m\xcb\x88undo." },
    };

    jinja::lexer lexer;
    const jinja::program ast = jinja::parse_from_tokens(lexer.tokenize(tmpl));
    for (const auto & c : cases) {
        std::string out;
        try {
            jinja::context ctx(tmpl);
            common_chat_tts_norm_add_builtins(ctx, dicts);
            auto msg = jinja::mk_val<jinja::value_object>();
            msg->insert("role", jinja::mk_val<jinja::value_string>("user"));
            msg->insert("content", jinja::mk_val<jinja::value_string>(c[1]));
            auto msgs = jinja::mk_val<jinja::value_array>();
            msgs->push_back(msg);
            ctx.set_val("messages", msgs);
            ctx.set_val("language", jinja::mk_val<jinja::value_string>(""));
            ctx.set_val("voice", jinja::mk_val<jinja::value_string>(c[0]));
            jinja::runtime runtime(ctx);
            out = jinja::runtime::gather_string_parts(runtime.execute(ast))->as_string().str();
        } catch (const std::exception & e) {
            out = std::string("exception: ") + e.what();
        }
        if (!t.assert_true(c[0] + ": " + c[1], out == c[2])) {
            t.log("Expected: " + c[2]);
            t.log("Actual  : " + out);
        }
    }
}

// every frame is the constant 0.25 (only the DC bin, of magnitude n_fft * 0.25)
// the overlap-add of hann windows at a quarter hop sums to 2, their squares to 1.5: the output is 0.25 * 2 / 1.5
MAKE_TEST(test_kokoro_istft) {
    const int n_fft = 20, hop = 5, n_frames = 9;
    std::vector<float> spec((size_t) n_frames * (n_fft + 2), 0.0f);
    for (int fr = 0; fr < n_frames; fr++) {
        float * row = spec.data() + (size_t) fr * (n_fft + 2);
        // the first half of a row is log magnitudes, the second half phases
        for (int k = 0; k <= n_fft / 2; k++) {
            row[k] = k == 0 ? logf(n_fft * 0.25f) : -100.0f;
        }
    }
    std::vector<float> audio;
    mtmd_graph_kokoro::istft(spec, audio);
    t.assert_equal("samples", (size_t) (n_frames - 1) * hop, audio.size());
    // the first and last hop are covered by fewer frames
    float max_err = 0.0f;
    for (size_t i = hop; i + hop < audio.size(); i++) {
        max_err = std::max(max_err, std::fabs(audio[i] - 1.0f / 3.0f));
    }
    t.assert_true("constant signal, max error " + std::to_string(max_err), max_err < 1e-4f);
}

// 13 tokens per 100-frame chunk; the tail is ceil^3 (one halving per stride-2 conv layer)
MAKE_TEST(test_qwen3a_tokens_for_frames) {
    const std::vector<std::pair<int64_t, int64_t>> cases = {
        {    0,   0 },
        {    1,   1 },
        {   99,  13 },
        {  100,  13 },
        {  101,  14 },
        {  110,  15 },
        {  200,  26 },
        {  210,  28 },
        { 1000, 130 },
        { 1024, 133 },
    };

    for (const auto & [frames, expected] : cases) {
        const std::string name = std::to_string(frames) + " frames";
        t.assert_equal(name, expected, mtmd_graph_qwen3a::qwen3a_tokens_for_frames(frames));
    }
}

//
// main
//

int main(int argc, char ** argv) {
    testing t(std::cout);
    t.verbose = true;

    // the audio preprocessors run on the CPU backend
    ggml_backend_load_all();

    // the video helper logs every frame at debug level
    mtmd_helper_log_set([](ggml_log_level level, const char * text, void *) {
        if (level != GGML_LOG_LEVEL_DEBUG) {
            fputs(text, stderr);
        }
    }, nullptr);

    // usage: test-mtmd-impl [filter_regex]
    for (int i = 1; i < argc; i++) {
        t.set_filter(argv[i]);
    }

    for (const auto & e : test_registry::all()) {
        t.test(e.name, e.fn);
    }

    return t.summary();
}
