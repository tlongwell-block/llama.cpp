#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>

// Exact rendered text only. Media lengths and chat templates are not cached.
class frankie_prompt_counts {
    struct entry { size_t tokens; uint64_t used; };
    std::map<std::string, entry> entries;
    std::mutex mutex;
    size_t bytes = 0;
    uint64_t clock = 0;
    const size_t max_bytes, max_entries;
    static size_t cost(const std::string & text) {
        return text.size() + sizeof(std::pair<const std::string, entry>) + 4 * sizeof(void *) + 1;
    }

  public:
    explicit frankie_prompt_counts(size_t max_bytes = 4 * 1024 * 1024, size_t max_entries = 1024) :
        max_bytes(max_bytes), max_entries(max_entries) {}

    template<class Count>
    size_t get(const std::string & text, Count count) {
        if (text.empty()) { return 0; }
        std::lock_guard<std::mutex> lock(mutex);
        const auto found = entries.find(text);
        if (found != entries.end()) {
            found->second.used = ++clock;
            return found->second.tokens;
        }
        const size_t tokens = count();
        const size_t size = cost(text);
        if (!max_entries || size > max_bytes) { return tokens; }
        while (!entries.empty() && (entries.size() >= max_entries || bytes > max_bytes - size)) {
            const auto oldest = std::min_element(entries.begin(), entries.end(), [](const auto & a, const auto & b) {
                return a.second.used < b.second.used;
            });
            bytes -= cost(oldest->first);
            entries.erase(oldest);
        }
        entries.emplace(text, entry{tokens, ++clock});
        bytes += size;
        return tokens;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        entries.clear();
        bytes = 0;
        clock = 0;
    }
};
