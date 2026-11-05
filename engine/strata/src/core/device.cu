// src/core/device.cu - P2.S1: the CUDA side of the runtime core.
#include "strata/core/device.hpp"

#include <cuda_runtime.h>

#include <cstring>

namespace strata::core {

namespace {

// A NaN pattern, not zero.  Zeros read from uninitialised memory are indistinguishable from real zeros in a
// dequantized weight or a masked attention score, which is exactly the kind of wrong-but-plausible value the
// Phase 1 harnesses kept catching.
__global__ void poison_kernel(float* p, uint64_t n_floats) {
    const uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_floats) p[i] = __int_as_float(0x7fc00000);
}

}  // namespace

std::string device_code_error() {
    // every .cu of the engine is compiled for the same CMAKE_CUDA_ARCHITECTURES, so this kernel stands for all
    cudaFuncAttributes a{};
    const cudaError_t e = cudaFuncGetAttributes(&a, poison_kernel);
    if (e == cudaSuccess) return {};
    cudaGetLastError();
    return cudaGetErrorString(e);
}

}  // namespace strata::core
