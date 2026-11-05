// include/strata/kernels/pdl.hpp - launch_pdl, the launch of the verify window's quantization and dense projection
// kernels.
#pragma once

#include <cuda_runtime.h>

#include <utility>

namespace strata::kernels {

/// The probe left of programmatic dependent launch (sm_90+, opt-in STRATA_DF_PDL): always false here; its first call
/// per device reads STRATA_DF_PDL and clears the thread's pending CUDA error.
bool pdl_supported();

#if defined(__CUDACC__)

/// `kernel<<<grid, block, smem, stream>>>(args...)`.  A launch error stays the thread's last error for the caller's own
/// check (cudaGetLastError) to report.
template <typename... KArgs, typename... Args>
inline void launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                       Args&&... args) {
    kernel<<<grid, block, smem, stream>>>(std::forward<Args>(args)...);
}

#endif  // __CUDACC__

}  // namespace strata::kernels
