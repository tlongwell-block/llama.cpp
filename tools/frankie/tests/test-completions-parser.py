#!/usr/bin/env python3
"""Exercise completions.h's parser wiring without starting a brain/server.

Usage: python3 tools/frankie/tests/test-completions-parser.py BUILD_DIR MODEL_OR_VOCAB_GGUF [COMPLETIONS_HEADER]
Requires a built llama-common and a Qwen3.5 vocabulary with think tokens. The tiny
C++ harness extracts the production statements, so reverting the HTTP wiring
also reverts the code under test (rather than testing a second implementation).
Only vocabulary metadata is loaded; no inference or model tensors are used.
"""
import pathlib
import re
import subprocess
import sys
import tempfile

repo = pathlib.Path(__file__).resolve().parents[3]
build = pathlib.Path(sys.argv[1]).resolve()
vocab_path = pathlib.Path(sys.argv[2]).resolve()
header = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else repo / 'tools/frankie/completions.h'
source = header.read_text()

def one(pattern):
    matches = re.findall(pattern, source)
    if len(matches) != 1:
        raise RuntimeError(f'Expected one production expression: {pattern}: {len(matches)}')
    return matches[0]

raw_type = 'common_chat_input' if 'common_chat_input raw;' in source else 'std::string'
setup = one(r'(j.parser = common_chat_parser_params\(formatted\);\s*j.parser.parser = formatted.parser;(?:\s*j.parser.generation_prompt = [^;]+;)?)')
append = one(r'(j.raw(?: \+= [^;]+|\.append\([^;]+\));)')
final_parse = one(r'message = (common_chat_parse\([^;]+false, j.parser\));')
partial_parse = one(r'auto parsed = j.chat \? (common_chat_parse\([^;]+true, j.parser\)) :')
plain_final = one(r'json\{\{"index", 0\}, \{"text", (j.raw(?:.text)?)\}, \{"finish_reason", finish_reason\}\}')
plain_partial = one(r'const auto & text = j.chat \? parsed.content : ([^;]+);')
program = r'''
#include "chat.h"
#include "common.h"
#include "llama.h"
#include <fstream>
#include <iostream>
#include <sstream>
struct job { RAW_TYPE raw; common_chat_parser_params parser; };
bool ids_match(const common_chat_input & raw, const std::vector<llama_token> & ids) { return raw.tokens == ids; }
bool ids_match(const std::string &, const std::vector<llama_token> &) { return false; }
int main(int argc, char ** argv) {
    auto params = llama_model_default_params(); params.vocab_only = true;
    llama_model_ptr model(llama_model_load_from_file(argv[1], params));
    if (!model) { return 2; }
    auto * vocab = llama_model_get_vocab(model.get());
    auto templates = common_chat_templates_init(model.get(), "");
    int failures = 0;
    for (bool thinking : {false, true}) {
        common_chat_templates_inputs input;
        common_chat_msg user; user.role = "user"; user.content = "What colors?";
        input.messages = {user}; input.enable_thinking = thinking;
        input.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        auto formatted = common_chat_templates_apply(templates.get(), input);
        job j;
        SETUP
        const std::string output = thinking ? "Hmm.\n</think>\n\nRed and blue."
                                           : "Red and blue.";
        const auto ids = common_tokenize(vocab, output, false, true);
        std::string streamed, reasoning;
        std::vector<llama_token> expected_ids;
        for (const auto token : ids) {
            const auto piece = common_token_to_piece(vocab, token, true);
            if (!piece.empty()) {
                expected_ids.push_back(token);
                expected_ids.resize(expected_ids.size() + piece.size() - 1, LLAMA_TOKEN_NULL);
            }
            APPEND
            auto parsed = PARTIAL_PARSE;
            if (parsed.content.compare(0, streamed.size(), streamed) != 0) { ++failures; }
            streamed = parsed.content;
            reasoning = parsed.reasoning_content;
        }
        bool ids_ok = ids_match(j.raw, expected_ids);
        std::cout << (ids_ok ? "PASS" : "FAIL") << " sampled token IDs thinking=" << thinking << "\n";
        failures += !ids_ok;
        auto message = FINAL_PARSE;
        for (bool partial : {false, true}) {
            const auto & content = partial ? streamed : message.content;
            const auto & reason = partial ? reasoning : message.reasoning_content;
            bool ok = content == "Red and blue." && (thinking ? reason == "Hmm." : reason.empty());
            std::cout << (ok ? "PASS" : "FAIL") << " thinking=" << thinking << " partial=" << partial
                      << " content=[" << content << "] reasoning=[" << reason << "]\n";
            failures += !ok;
        }
        auto parsed = common_chat_msg{};
        const auto & text = PLAIN_PARTIAL;
        bool plain_ok = text == output && PLAIN_FINAL == output;
        std::cout << (plain_ok ? "PASS" : "FAIL") << " plain completion thinking=" << thinking << "\n";
        failures += !plain_ok;
    }
    return failures ? 1 : 0;
}
'''
for key, value in {'RAW_TYPE': raw_type, 'SETUP': setup, 'APPEND': append,
                   'FINAL_PARSE': final_parse, 'PARTIAL_PARSE': partial_parse,
                   'PLAIN_FINAL': plain_final, 'PLAIN_PARTIAL': plain_partial}.items():
    program = program.replace(key, value)
with tempfile.TemporaryDirectory(prefix='frankie-parser-') as tmp:
    cpp = pathlib.Path(tmp) / 'test.cpp'
    exe = pathlib.Path(tmp) / 'test'
    cpp.write_text(program)
    subprocess.run(['c++', '-std=c++17', str(cpp), '-o', str(exe),
                    *[f'-I{repo / p}' for p in ['common', 'include', 'ggml/include', 'vendor']],
                    f'-L{build / "bin"}', f'-Wl,-rpath,{build / "bin"}',
                    '-lllama-common', '-lllama', '-lggml', '-lggml-base'], check=True)
    sys.exit(subprocess.run([str(exe), str(vocab_path)]).returncode)
