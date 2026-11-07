// include/strata/core/defaults_026.hpp - STRATA_DEFAULTS=026: the master switch of the Qwen engine's 0.2.7 defaults.
#pragma once

#include <cstdlib>
#include <cstring>

namespace strata {

/// "026", "0.2.6" and "v0.2.6" name version 0.2.6; anything else (unset, "", "0", "027") does not.
inline bool is_026_token(const char* v) {
    return v != nullptr && (std::strcmp(v, "026") == 0 || std::strcmp(v, "0.2.6") == 0 || std::strcmp(v, "v0.2.6") == 0);
}

/// STRATA_DEFAULTS=026 (read once).
inline bool qwen_defaults_026() {
    static const bool on = is_026_token(std::getenv("STRATA_DEFAULTS"));
    return on;
}

/// STRATA_PROMPT_ATTN_IMMA (the prompt attention on int8 tensor cores, another summation order) when the user set nothing: ON for every CUDA card of
/// compute capability 8.0 or newer (the int8 m16n8k32 MMA), not Turing / Volta / Pascal, not HIP.  `cc` = 10 * major + minor as the engine
/// resolves it (an emulated or forced Turing counts as 75).  Measured on sm_120 (RTX PRO 5000) and sm_86 (2x RTX A4000): +1.1 .. +1.8% prompt.
inline bool prompt_attn_imma_default(bool hip, int cc, bool v026 = false) { return !v026 && !hip && cc >= 80; }

}  // namespace strata
