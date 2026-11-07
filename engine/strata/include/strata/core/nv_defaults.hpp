// strata/core/nv_defaults.hpp - the per-architecture defaults of switches that are exact (bitwise) on NVIDIA sm_86.
//
// 148-nv measured every switch of the 0.1.45 bit-exact ON set, one at a time and in the stacked set, on an RTX 3060, an RTX A4000
// (sm_86) and a Tesla P100 (sm_60): a switch is in a table only when its token ids equal the default's and it was faster on that
// architecture (5+ interleaved pairs of fresh engines, noise arm, equal expert-cache slots; notes-148-nv.md).  The identity gate
// (10 prompts + exact_pp 4K/20K, fixed flags) is IDENTICAL with the whole table on.  Switches that change bits, hang or lost speed
// on a card (STRATA_EXCHANGE_ROTATE, STRATA_PCIE_BALANCE, STRATA_QSA_WARP, ...) are not in any table.
//
//   STRATA_SM86_PERF=0   turns the sm_86 table off (cards of compute capability 8.6: RTX 3050-3090, RTX A2000-A6000)
//
// A switch the user set (to anything, including 0) is never overridden.  Applied once at start, before any engine code reads its
// switches; the table is used only when every visible CUDA device has the same compute capability.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// The environment settings that are the defaults of a card with compute capability major.minor (raw, as the driver reports it);
/// empty for every other card and when its off switch is 0.
std::vector<std::pair<std::string, std::string>> nv_default_env(int major, int minor);
/// The same with the answer of STRATA_DEFAULTS=026 given (the sm_121 table is empty then), so a test can check both without the environment.
std::vector<std::pair<std::string, std::string>> nv_default_env(int major, int minor, bool defaults026);

/// Sets `nv_default_env(major, minor)` in this process's environment where the user has not set the variable, prints one line to
/// stderr when it set anything, and returns the names it set.
std::vector<std::string> apply_nv_defaults(int major, int minor);

}  // namespace strata::core
