#pragma once

#include <cstddef>
#include <string>

// Studio sentence policy: keep short interjections with the next sentence.
inline size_t frankie_speech_boundary(const std::string & text, size_t begin, bool last) {
    if (last) { return text.size(); }
    size_t words = 0;
    bool in_word = false;
    for (size_t i = begin; i < text.size(); ++i) {
        const bool space = text[i] == ' ' || text[i] == '\n' || text[i] == '\r' || text[i] == '\t';
        if (!space) { in_word = true; continue; }
        if (!in_word) { continue; }
        in_word = false;
        ++words;
        const char previous = text[i - 1];
        const bool sentence = previous == '.' || previous == '!' || previous == '?';
        if ((sentence && words >= 3) || words >= 50) { return i; }
    }
    return begin;
}
