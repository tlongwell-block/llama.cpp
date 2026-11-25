#pragma once
#include "common.h"

inline common_params_sampling frankie_brain_sampling(bool thinking) {
    common_params_sampling sampling;
    sampling.temp = thinking ? 1.0f : 0.7f;
    sampling.top_p = thinking ? 0.95f : 0.8f;
    sampling.top_k = 20;
    sampling.min_p = 0.0f;
    sampling.penalty_present = thinking ? 0.0f : 1.5f;
    sampling.penalty_repeat = 1.0f;
    return sampling;
}
