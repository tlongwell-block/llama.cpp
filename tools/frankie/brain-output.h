#pragma once
#include <array>
#include <string>
#include <vector>

struct brain_output {
    std::string                          text;
    std::vector<size_t>                  ends;
    std::vector<std::array<float, 5120>> hidden;
};
