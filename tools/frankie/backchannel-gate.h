#pragma once

#include <cstdint>

// MaAI BC-Det produces one observation per 80 ms. Confirm both entry into a
// listener nod and exit from it; repeated reads of one observation do not count.
struct frankie_backchannel_gate {
    uint64_t last_end = 0;
    unsigned high = 0, low = 0;
    bool seen_backchannel = false;

    bool observe(float probability, uint64_t end, bool voiced) {
        if (end <= last_end) { return false; }
        last_end = end;
        if (!voiced) { high = low = 0; return false; }
        high = probability >= 0.2f ? high + 1 : 0;
        if (high >= 2) { seen_backchannel = true; }
        low = probability < (seen_backchannel ? 0.05f : 0.2f) ? low + 1 : 0;
        return low >= 2;
    }
};
