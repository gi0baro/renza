// include/strata/kernels/dp4a.hpp - the two instructions the kernels use that Pascal (sm_60) does not have.
#pragma once

#include <cstdint>

#define STRATA_DP4A(a, b, c) __dp4a((a), (b), (c))

/// Yield the thread while a doorbell flag is being polled.  The length of the pause is a backoff hint, not a
/// contract, and only pre-Volta CUDA has no `__nanosleep` at all.
__device__ __forceinline__ void strata_spin_pause() {
    __nanosleep(100);
}
