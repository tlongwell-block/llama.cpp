#pragma once
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

struct brain_output {
    std::string                          text;
    std::vector<size_t>                  ends;
    std::vector<std::array<float, 5120>> hidden;
    // The brain's final state for each hidden row, width floats each, for brain-led delivery.
    size_t                               width = 0;
    std::vector<float>                   final;

    brain_output slice(size_t begin, size_t end) const {
        if (begin >= end || end > text.size() || hidden.size() != ends.size() || ends.empty() || ends.back() < end) {
            throw std::runtime_error("speech span outside hidden rows");
        }
        const auto first = std::upper_bound(ends.begin(), ends.end(), begin);
        const auto last = std::lower_bound(first, ends.end(), end) + 1;
        brain_output result;
        result.text = text.substr(begin, end - begin);
        result.hidden.assign(hidden.begin() + (first - ends.begin()), hidden.begin() + (last - ends.begin()));
        for (auto it = first; it != last; ++it) { result.ends.push_back(std::min(*it, end) - begin); }
        if (width) {
            result.width = width;
            result.final.assign(final.begin() + (first - ends.begin()) * width, final.begin() + (last - ends.begin()) * width);
        }
        return result;
    }
};
