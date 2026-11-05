// strata/core/sm75_series.hpp - STRATA_SM75_PERF, the opt-in switch of the Turing (RTX 20, compute capability 7.5)
// kernel series.  Its one reader is qsa_prompt_attn.cu: where the prompt attention runs as on a Turing card
// (STRATA_QSA_WARP=1|attn), STRATA_SM75_PERF=1 takes the v1 kernel's prefetch (#1821; the same bits).
#pragma once

#include <cstdlib>

namespace strata {

/// STRATA_SM75_PERF=1 (read once)
inline bool sm75_series() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_SM75_PERF");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return on;
}

}  // namespace strata
