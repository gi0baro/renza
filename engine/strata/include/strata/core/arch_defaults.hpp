// include/strata/core/arch_defaults.hpp - the per-architecture defaults of switches that are exact everywhere.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// The tensor-core GDN prompt scan and the one-pass conv (QUAL-GDN, l620: prompt +4.4% at 4K, +5.5% at 16K, +5.7% at 32K; first-token KL mean
/// 0.0019, max 0.0125; owner yes 2026-11-07): STRATA_GDN_TC=1 and STRATA_GDN_CONVL2=2 on Linux CUDA, what serve/server.py's GDN_STACK_CUDA set.
/// The engine still gates the scan at run time (compute capability 8 or more, one SM per value head, shared memory) and falls back to the old
/// kernels.  Empty when `linux_cuda` is false (HIP, Windows, Metal, SYCL).  =0 turns each off; a value the user set is kept.  Silent.
std::vector<std::pair<std::string, std::string>> gdn_default_env_cuda(bool linux_cuda);
std::vector<std::string> apply_gdn_defaults_cuda(bool linux_cuda);

}  // namespace strata::core
