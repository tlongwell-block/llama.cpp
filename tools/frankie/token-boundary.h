#pragma once

#include "llama.h"
#include <string>
#include <vector>
#include <stdexcept>

// Media markers delimit independent text-tokenization segments.
inline bool frankie_token_boundary(const llama_vocab * vocab, const std::string & prompt, size_t cut) {
    // Match the formatted-prompt limit. A byte limit of 128 KiB forced valid
    // long contexts to replay even when their cached token prefix was identical.
    if (cut > prompt.size() || prompt.size() > 4 * 1024 * 1024) { return false; }
    size_t begin = 0;
    for (size_t marker = prompt.find("[FRANKIE_"); marker != std::string::npos && marker < cut;
         marker = prompt.find("[FRANKIE_", begin)) {
        const size_t end = prompt.find(']', marker);
        if (end == std::string::npos || end >= cut) { return false; }
        begin = end + 1;
    }
    const size_t next = prompt.find("[FRANKIE_", cut);
    const size_t end = next == std::string::npos ? prompt.size() : next;
    auto tokens = [&](size_t start, size_t count) {
        std::vector<llama_token> out(count + 32);
        const int n = llama_tokenize(vocab, prompt.data() + start, count, out.data(), out.size(), false, true);
        if (n < 0) { throw std::runtime_error("token boundary bounds"); }
        out.resize(n);
        return out;
    };
    auto split = tokens(begin, cut - begin);
    const auto rest = tokens(cut, end - cut);
    split.insert(split.end(), rest.begin(), rest.end());
    return split == tokens(begin, end - begin);
}
