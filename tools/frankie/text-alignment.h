#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct frankie_text_offsets {
    std::vector<size_t> floor, ceil;
    std::vector<bool> whitespace;
    std::vector<uint32_t> codepoints;

    explicit frankie_text_offsets(const std::string & text) : floor(text.size() + 1), ceil(text.size() + 1) {
        for (size_t i = 0; i < text.size();) {
            const size_t start = i;
            const auto first = static_cast<uint8_t>(text[i++]);
            uint32_t cp = first;
            size_t extra = 0;
            if (first >= 0xc2 && first <= 0xdf) { cp = first & 0x1f; extra = 1; }
            else if (first >= 0xe0 && first <= 0xef) { cp = first & 0x0f; extra = 2; }
            else if (first >= 0xf0 && first <= 0xf4) { cp = first & 7; extra = 3; }
            else if (first >= 0x80) { throw std::runtime_error("invalid UTF-8 alignment input"); }
            if (extra > text.size() - i) { throw std::runtime_error("incomplete UTF-8 alignment input"); }
            for (size_t j = 0; j < extra; ++j) {
                const auto byte = static_cast<uint8_t>(text[i++]);
                if ((byte & 0xc0) != 0x80) { throw std::runtime_error("invalid UTF-8 continuation"); }
                cp = (cp << 6) | (byte & 0x3f);
            }
            if ((extra == 1 && cp < 0x80) || (extra == 2 && cp < 0x800) ||
                (extra == 3 && cp < 0x10000) || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) {
                throw std::runtime_error("invalid UTF-8 code point");
            }
            codepoints.push_back(cp);
            const size_t n = whitespace.size();
            for (size_t byte = start; byte < i; ++byte) {
                floor[byte] = n;
                ceil[byte] = n + (byte != start);
            }
            whitespace.push_back((cp >= 9 && cp <= 13) || (cp >= 0x1c && cp <= 0x20) || cp == 0x85 ||
                                 cp == 0xa0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200a) ||
                                 cp == 0x2028 || cp == 0x2029 || cp == 0x202f || cp == 0x205f || cp == 0x3000);
            floor[i] = ceil[i] = n + 1;
        }
    }

    bool has_text(size_t start, size_t end) const {
        // An isolated partial token decodes as a replacement character in HF.
        if (floor.at(start) != ceil.at(start) || floor.at(end) != ceil.at(end)) { return true; }
        for (size_t i = floor.at(start); i < ceil.at(end); ++i) {
            if (!whitespace.at(i)) { return true; }
        }
        return false;
    }
};

struct frankie_normalized_text {
    std::string text;
    // Original character range for each normalized UTF-8 byte.
    std::vector<std::pair<size_t, size_t>> offsets;
    explicit frankie_normalized_text(const std::string & original);
    std::vector<std::pair<size_t, size_t>> spans(const std::vector<size_t> & token_ends) const;
};

// Match HF character offsets, including tokens that split a UTF-8 code point.
inline std::vector<size_t> frankie_align_text(const std::string & brain, const std::vector<size_t> & brain_ends,
                                            size_t lead, const std::string & spoken, const std::vector<size_t> & talker_ends,
                                            const std::vector<std::pair<size_t, size_t>> & normalized_spans = {}) {
    if (brain.compare(lead, spoken.size(), spoken) != 0 || brain_ends.empty() || brain_ends.back() != brain.size() ||
        talker_ends.empty() || (normalized_spans.empty() && talker_ends.back() != spoken.size()) ||
        (!normalized_spans.empty() && normalized_spans.size() != talker_ends.size())) { throw std::runtime_error("alignment span bounds"); }
    const frankie_text_offsets b(brain), q(spoken);
    if (b.floor.at(lead) != b.ceil.at(lead)) { throw std::runtime_error("alignment starts inside code point"); }
    std::vector<size_t> starts, ends, keep;
    size_t previous = 0;
    for (size_t i = 0; i < brain_ends.size(); ++i) {
        const size_t end = brain_ends[i];
        if (end < previous || end > brain.size()) { throw std::runtime_error("brain token bounds"); }
        starts.push_back(b.ceil.at(previous)); ends.push_back(b.ceil.at(end));
        if (b.has_text(previous, end)) { keep.push_back(i); }
        previous = end;
    }
    if (keep.empty()) { throw std::runtime_error("no brain text for alignment"); }
    std::vector<size_t> result;
    previous = 0;
    size_t token = 0;
    for (size_t end : talker_ends) {
        if (end < previous || (normalized_spans.empty() && end > spoken.size())) { throw std::runtime_error("talker token bounds"); }
        const auto span = normalized_spans.empty() ? std::make_pair(q.floor[previous], q.ceil[end]) : normalized_spans[token++];
        if (span.first > span.second || span.second > q.codepoints.size()) { throw std::runtime_error("normalized span bounds"); }
        const size_t a = b.floor[lead] + span.first, z = b.floor[lead] + span.second;
        size_t chosen = brain_ends.size();
        for (size_t i : keep) { if (ends[i] >= z) { chosen = i; break; } }
        if (chosen == brain_ends.size() || starts[chosen] > a) {
            int64_t best = std::numeric_limits<int64_t>::min();
            for (size_t i : keep) {
                const int64_t overlap = int64_t(std::min(ends[i], z)) - int64_t(std::max(starts[i], a));
                if (overlap > best) { best = overlap; chosen = i; }
            }
        }
        result.push_back(chosen); previous = end;
    }
    return result;
}
