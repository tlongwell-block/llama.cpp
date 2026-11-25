#pragma once
#include "base64.hpp"
#include "download.h"

inline void frankie_image_detail(const std::string & detail) {
    if (detail != "auto" && detail != "low" && detail != "high") {
        throw std::invalid_argument("image detail must be auto, low, or high");
    }
}

inline std::vector<unsigned char> frankie_image_bytes(const std::string & url, bool realtime = false) {
    constexpr size_t limit = 10 * 1024 * 1024;
    std::vector<unsigned char> bytes;
    if (!realtime && (url.compare(0, 7, "http://") == 0 || url.compare(0, 8, "https://") == 0)) {
        common_remote_params params;
        params.max_size = limit;
        params.timeout = 10;
        auto result = common_remote_get_content(url, params);
        if (result.first < 200 || result.first >= 300) { throw std::invalid_argument("image download failed"); }
        bytes.assign(result.second.begin(), result.second.end());
    } else {
        const auto comma = url.find(',');
        const auto header = url.substr(0, comma);
        if (comma == std::string::npos || header.compare(0, 11, "data:image/") != 0 ||
            header.size() < 7 || header.compare(header.size() - 7, 7, ";base64") != 0 ||
            (realtime && header != "data:image/png;base64" && header != "data:image/jpeg;base64")) {
            throw std::invalid_argument(realtime ? "inline PNG or JPEG required" : "image_url must be an HTTP(S) or image data URL");
        }
        if (url.size() - comma - 1 > 4 * ((limit + 2) / 3)) { throw std::invalid_argument("image exceeds 10 MiB"); }
        auto decoded = base64::decode(url.substr(comma + 1));
        bytes.assign(decoded.begin(), decoded.end());
    }
    if (bytes.empty() || bytes.size() > limit) { throw std::invalid_argument("image is empty or exceeds 10 MiB"); }
    return bytes;
}
