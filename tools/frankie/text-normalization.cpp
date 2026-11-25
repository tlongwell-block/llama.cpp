#include "text-alignment.h"

#include <unicode/normalizer2.h>
#include <unicode/rbnf.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>
#include <memory>
#include <regex>

std::string frankie_spoken_numbers(const std::string & text) {
    static const std::regex digits("\\b[0-9]{1,14}\\b");
    std::string result;
    size_t previous = 0;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), digits); it != std::sregex_iterator(); ++it) {
        static thread_local const auto numbers = [] {
            UErrorCode error = U_ZERO_ERROR;
            auto value = std::make_unique<icu::RuleBasedNumberFormat>(icu::URBNF_SPELLOUT, icu::Locale::getEnglish(), error);
            if (U_FAILURE(error)) { throw std::runtime_error("speech number formatter initialization failed"); }
            return value;
        }();
        result.append(text, previous, it->position() - previous);
        icu::UnicodeString words;
        icu::FieldPosition position(0);
        numbers->format(int64_t(std::stoll(it->str())), words, position);
        words.toUTF8String(result);
        previous = it->position() + it->length();
    }
    result.append(text, previous, text.size() - previous);
    return result;
}

frankie_normalized_text::frankie_normalized_text(const std::string & original) {
    const frankie_text_offsets source(original);
    if (original.size() > 8192) { throw std::runtime_error("normalization input bounds"); }
    UErrorCode error = U_ZERO_ERROR;
    const auto * nfc = icu::Normalizer2::getNFCInstance(error);
    if (U_FAILURE(error) || !nfc) { throw std::runtime_error("NFC initialization failed"); }
    struct unit { UChar32 cp; int change; uint8_t ccc; };
    std::vector<unit> decomposed;
    for (uint32_t cp : source.codepoints) {
        icu::UnicodeString part;
        if (!nfc->getDecomposition(cp, part)) { part.append(UChar32(cp)); }
        bool first = true;
        for (int32_t i = 0; i < part.length();) {
            const UChar32 value = part.char32At(i);
            i += U16_LENGTH(value);
            decomposed.push_back({value, first ? 0 : 1, u_getCombiningClass(value)});
            first = false;
        }
    }
    // Stable canonical order, with starters delimiting combining sequences.
    size_t start = 0;
    for (size_t i = 0; i <= decomposed.size(); ++i) {
        if (i == decomposed.size() || decomposed[i].ccc == 0) {
            std::stable_sort(decomposed.begin() + start, decomposed.begin() + i,
                             [](const unit & a, const unit & b) { return a.ccc < b.ccc; });
            start = i;
        }
    }
    std::vector<unit> composed;
    size_t starter = size_t(-1);
    uint8_t last_class = 0;
    for (const auto & next : decomposed) {
        const UChar32 merged = starter == size_t(-1) ? -1 : nfc->composePair(composed[starter].cp, next.cp);
        if (merged >= 0 && (last_class == 0 || last_class < next.ccc)) {
            composed[starter].cp = merged;
            composed[starter].change += next.change - 1;
        } else {
            if (next.ccc == 0) { starter = composed.size(); }
            composed.push_back(next);
            last_class = next.ccc;
        }
    }
    // HF inserted characters retain the previous source range; composition consumes following characters.
    size_t cursor = 0;
    for (const auto & value : composed) {
        const size_t index = value.change > 0 ? (cursor ? cursor - 1 : 0) : cursor;
        if (index >= source.codepoints.size()) { throw std::runtime_error("normalization offset bounds"); }
        std::string bytes;
        icu::UnicodeString(value.cp).toUTF8String(bytes);
        text += bytes;
        offsets.insert(offsets.end(), bytes.size(), {index, index + 1});
        if (value.change <= 0) { cursor += 1 + size_t(-value.change); }
        if (cursor > source.codepoints.size()) { throw std::runtime_error("normalization consumed bounds"); }
    }
    if (cursor != source.codepoints.size()) { throw std::runtime_error("normalization incomplete"); }
}

std::vector<std::pair<size_t, size_t>> frankie_normalized_text::spans(const std::vector<size_t> & token_ends) const {
    std::vector<std::pair<size_t, size_t>> result;
    size_t start = 0;
    for (size_t end : token_ends) {
        if (end <= start || end > offsets.size()) { throw std::runtime_error("normalized token bounds"); }
        result.emplace_back(offsets[start].first, offsets[end - 1].second);
        start = end;
    }
    if (start != text.size()) { throw std::runtime_error("normalized token coverage"); }
    return result;
}
