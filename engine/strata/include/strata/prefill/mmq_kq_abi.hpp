#pragma once

#define STRATA_KQ_ABI_VERSION 1

#define STRATA_KQ_LIB_NAME "libstrata_kq.so"

extern "C" {
typedef int (*strata_kq_abi_fn)();
typedef int (*strata_kq_init_fn)(int device);
typedef int (*strata_kq_run_fn)(int device, int ggml_type, const void* mmq_args, void* stream);
}

// the engine's side
namespace strata::prefill::mmq::kq {
/// STRATA_MMQ_KQ=1: the K-quant experts' prompt products through MMQ (opt-in: other rounding than the FP16 path).
bool wanted();
/// wanted() and the library is open and started on every GPU (the first call opens it and says what happened).
bool available();
/// One product on the current GPU: 0 = launched, else an error (-1: no library or not this type).
int run(int ggml_type, const void* mmq_args, void* stream);
/// sizeof(mmq_args) on the engine's side (moe_mmq.cu), checked against the library's.
int abi_args_size();
}  // namespace strata::prefill::mmq::kq
