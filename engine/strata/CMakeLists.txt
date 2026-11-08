# Strata Engine - top-level build (P1.S1).
#
# Targets (phase-1 Ă‚Â§P1.S1): `strata_artifact` (no CUDA dependency), `tests`.  `strata_plan` and the C++
# `strata-pack` are added when their sources exist - a target with no sources would make the build graph
# claim something the tree does not have, which is worse than an absent target.
#
# CUDA targets arrive in Phase 2.
cmake_minimum_required(VERSION 3.24)
project(strata VERSION 0.1.45 LANGUAGES CXX)
add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(STRATA_ENABLE_CUDA "Build the CUDA targets (Phase 2+)" OFF)
option(STRATA_MMQ_KQUANTS "Build the MMQ prompt kernels for Q4_K / Q5_K / Q5_1 / Q6_K: the experts of Unsloth's K-quant files" ON)
# Q6_K gate/up in the grouped expert kernels (Unsloth's UD-Q6_K_XL, and layer 2 of UD-Q5_K_XL; docs/QUANTS.md).  On by
# default since MODEL-UPDATE Phase 1: one more instance per grouped kernel in iq_kernels.cu, +19 KB of sm_60 code,
# measured to leave every expert-cache slot count as it was (P100, R9700).  -DSTRATA_Q6K_EXPERTS=OFF drops it.
option(STRATA_Q6K_EXPERTS "Build the Q6_K gate/up grouped expert kernels (UD-Q6_K_XL, UD-Q5_K_XL)" ON)
if(STRATA_Q6K_EXPERTS)
  add_compile_definitions(STRATA_Q6K_EXPERTS=1)
endif()
option(STRATA_ORCA_Q4KS_MMQ "Build the GGML Q5_0 MMQ instance for OrcaRouter Q4_K_S" OFF)
set(_strata_tests_default OFF)
option(STRATA_BUILD_TESTS "Build the test suite" ${_strata_tests_default})
option(STRATA_BUILD_CONVERSATION_TESTS "Build standalone conversation cache tests" OFF)
option(STRATA_WERROR "Treat compiler warnings as errors" OFF)
# #2027: defined here, before the GPU section reads it to register the native-expert tests (it was defined at the
# native-experts section far below, so a first configure registered four tests fewer than the second)
option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)

if(STRATA_BUILD_CONVERSATION_TESTS)
  enable_testing()
  add_executable(conversation_cache_test src/core/conversation_cache_test.cpp)
  target_include_directories(conversation_cache_test PRIVATE include)
  add_test(NAME conversation_cache_test COMMAND conversation_cache_test)
  add_executable(conversation_split_failure_test src/core/conversation_split_failure_test.cpp)
  target_include_directories(conversation_split_failure_test PRIVATE include)
  add_test(NAME conversation_split_failure_test COMMAND conversation_split_failure_test)
  add_executable(conversation_memory_test src/core/conversation_memory_test.cpp src/core/conversation_memory.cpp)
  target_include_directories(conversation_memory_test PRIVATE include)
  add_test(NAME conversation_memory_test COMMAND conversation_memory_test)
  add_executable(conversation_file_test src/core/conversation_file_test.cpp src/core/conversation_file.cpp)
  target_include_directories(conversation_file_test PRIVATE include)
  add_test(NAME conversation_file_test COMMAND conversation_file_test)
  add_executable(session_save_reclaim_test src/core/session_save_reclaim_test.cpp src/core/conversation_file.cpp)
  target_include_directories(session_save_reclaim_test PRIVATE include)
  add_test(NAME session_save_reclaim_test COMMAND session_save_reclaim_test)
endif()

if(STRATA_ENABLE_CUDA)
  enable_language(CUDA)
  # `enable_language(CUDA)` gives the .cu compile rules but NOT the imported targets, so `CUDA::cudart` has to
  # be brought in explicitly for the host-side tests that call cudaMalloc themselves.
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures" FORCE)
  endif()
  message(STATUS "Strata: CUDA enabled, arch ${CMAKE_CUDA_ARCHITECTURES}")
  set(_strata_gpu_runtime_target CUDA::cudart)
  set(_strata_gpu_blas_target CUDA::cublas)
  set(_strata_gpu_include_directories ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

# Warnings live on an INTERFACE target the project's own code links, NOT in add_compile_options: applied
# globally they would also hit FetchContent'd Catch2, whose -W4 warnings are not this project's to fix and
# whose /WX failures would be indistinguishable from our own.
add_library(strata_warnings INTERFACE)
target_compile_options(strata_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
  $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
if(STRATA_WERROR)
  target_compile_options(strata_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
endif()

# ---- strata_artifact -------------------------------------------------------
# Header-only, and that is a deliberate property rather than an unfinished split: the GGUF reader and the
# scalar dequantizers hold no state worth hiding, and a header-only boundary cannot introduce a
# duplicate-symbol or missing-declaration bug into code that is already validated against the real artifact.
add_library(strata_artifact INTERFACE)
target_include_directories(strata_artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_artifact INTERFACE strata_warnings)


# ---- strata_plan ----------------------------------------------------------
# Also header-only.  A "planner" that hides state would be one whose decision is not inspectable, and the
# whole point of P1.S9 is that the decision is arithmetic you can check by hand.
add_library(strata_plan INTERFACE)
target_include_directories(strata_plan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_plan INTERFACE strata_warnings)


# ---- strata_core (GPU runtime) ---------------------------------------------
# Only built when a GPU backend is on, so the Phase 1 targets keep working without a toolkit.
if(STRATA_ENABLE_CUDA)
  add_library(strata_core STATIC src/core/device.cu src/core/pinned.cu src/platform/memory.cpp src/core/graph.cpp src/core/weights.cpp
                               src/core/layout.cpp)
  target_include_directories(strata_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # The selected runtime is PUBLIC because graph.cpp is a HOST translation unit that calls cudaGraphLaunch and
  # cudaEventQuery: the .cu sources get cudart through the CUDA language rules, a .cpp does not, and the
  # linker error ("unresolved external symbol cudaGraphInstantiate") names the function rather than the cause.
  target_link_libraries(strata_core PUBLIC strata_plan strata_warnings ${_strata_gpu_runtime_target})

  # ---- strata_kernels (P2.S2) ----
  # Naive first, each with a parity test against a reference that was itself validated against ggml.  The
  # kernels link `strata_artifact` for the scalar dequantizers, because using the VALIDATED scalar path as the
  # reference is the whole point - a kernel checked against another copy of itself proves nothing.
  add_library(strata_kernels STATIC src/kernels/cuda/dequant_s2.cu src/kernels/cuda/s2_gemv.cu
                          src/kernels/cuda/s_gemv.cu
                          src/kernels/cuda/s2_gemv_quads.cu
                          src/kernels/cuda/s2_gemv_fast.cu
                          src/kernels/cuda/router_top10.cu
                          src/kernels/cuda/quantize_act.cu
                          src/kernels/cuda/rope.cu
                          src/kernels/cuda/sampler.cu
                          src/kernels/cuda/s2_gemv_q8.cu
    src/kernels/cuda/s2_expert_grouped.cu
                          src/kernels/cuda/shared_expert.cu
                          src/kernels/cuda/gr.cu
                          src/kernels/cuda/gdn.cu
                          src/kernels/cuda/qsa.cu
                          src/kernels/cuda/elementwise.cu
                          src/kernels/cuda/bf16_gemv.cu
                          src/kernels/cuda/native_mmvq.cu
                          src/kernels/cuda/native_bf16.cu
                          src/kernels/cuda/native_gr_norm.cu
                          src/kernels/cuda/native_moe.cu
                          src/kernels/cuda/native_gdn.cu
                          src/kernels/cuda/native_gr_postops.cu
                          src/kernels/cuda/native_router.cu
                          src/kernels/cuda/native_gdn_preprocess.cu
                          src/kernels/cuda/native_qsa.cu
                          src/kernels/cuda/native_rope.cu
                          src/kernels/cuda/native_ple_postops.cu
                          src/kernels/cuda/native_qsa_indexer.cu
                          src/kernels/cuda/native_qsa_score.cu
                          src/kernels/cuda/ple.cu
                          src/kernels/cuda/kv_q8.cu
                          src/kernels/cuda/kv_stream.cu
                          src/kernels/cuda/kv_q4.cu
                          src/kernels/cuda/fused_gr.cu
                          src/kernels/cuda/cvec.cu
                          src/kernels/cuda/qsa_decode_attn.cu
                          src/kernels/cuda/qsa_prompt_attn.cu
                          src/kernels/cuda/fused_gdn.cu
                          src/kernels/cuda/dequant_bf16.cu
                          src/kernels/cuda/qsa_select.cu
                          src/kernels/cuda/verify_kernels.cu
                          src/kernels/cuda/iq_kernels.cu
                          src/kernels/ngram.cpp
                          src/ngram/ple_reader.cpp
                          src/platform/direct_file.cpp)
  target_link_libraries(strata_kernels PUBLIC strata_core strata_artifact)
  # plan v0.3 P6: llama.cpp's block layouts and codebook grids for the i-quant formats (MIT, third_party/ggml)
  target_include_directories(strata_kernels PRIVATE third_party/ggml)
  # Match the pinned llama.cpp CUDA arithmetic contract for this adapter only.
  if(STRATA_ENABLE_CUDA)
    set_source_files_properties(src/kernels/cuda/native_mmvq.cu src/kernels/cuda/native_bf16.cu
      src/kernels/cuda/native_gr_norm.cu
      src/kernels/cuda/native_moe.cu
      src/kernels/cuda/native_gdn.cu
      src/kernels/cuda/native_gr_postops.cu
      src/kernels/cuda/native_router.cu
      src/kernels/cuda/native_gdn_preprocess.cu
      src/kernels/cuda/native_qsa.cu
      src/kernels/cuda/native_rope.cu
      src/kernels/cuda/native_ple_postops.cu
      src/kernels/cuda/native_qsa_indexer.cu
      src/kernels/cuda/native_qsa_score.cu
      PROPERTIES COMPILE_OPTIONS "--use_fast_math")
  endif()

  # ---- strata_engine (P2.S5) -----------------------------------------------------
  # The COMPOSED layer lives here and not in `strata_core`, because it calls kernels and `strata_kernels`
  # already links `strata_core` - putting `gdn_layer` in core would make the two mutually dependent.  A third
  # library is the honest shape of the dependency graph: core knows about devices and weights, kernels know
  # about arithmetic, and this one knows the ORDER the arithmetic happens in.
  add_library(strata_engine STATIC src/core/layer.cpp src/core/session.cpp src/core/expert_source.cpp
    src/core/expert_cache.cpp src/core/native_head.cpp src/core/native_dense.cpp src/core/verify.cpp src/core/mtp.cpp
    src/core/conversation_snapshot.cpp src/core/conversation_state.cpp src/core/conversation_memory.cpp
    src/core/nv_defaults.cpp src/core/vmm.cpp src/core/conversation_file.cpp)
  target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
  # `strata generate` (P2.S6).  The FIRST user-facing program: everything under it is a component, and this is
  # what composes them into a token.  No tokenizer - the prompt arrives as ids - because Gate C1 compares logits
  # against llama.cpp on the SAME ids and a tokenizer on one side of that comparison is a second variable.
  # plan v0.3 P5: batched prompt processing (tensor-core GEMMs through cuBLAS).
  add_library(strata_prefill STATIC src/prefill/gemm.cu src/prefill/kernels.cu src/prefill/prefill.cpp)
  target_include_directories(strata_prefill PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
  target_link_libraries(strata_prefill PUBLIC strata_engine ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})

  add_library(strata_serve STATIC src/program/engine.cpp)
  target_include_directories(strata_serve PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                             PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(strata_serve PUBLIC strata_engine strata_prefill strata_spec ${_strata_gpu_runtime_target})
endif()

# ---- strata_kernels_cpu (P2.S3) ----
#
# A SEPARATE LIBRARY from `strata_kernels`, and `/arch:AVX512` is set on the SOURCE rather than on the target
# so that the flag applies only to the TU that needs it.  Both parts matter:
#
#   * the CPU expert kernel uses `_mm256_dpbusd_epi32` (AVX512-VNNI) and `_mm256_multishift_epi64_epi8`
#     (AVX512-VBMI); without the flag MSVC reports undeclared intrinsics and never mentions AVX-512;
#   * a TU compiled WITH `/arch:AVX512` may use AVX-512 in ANY of its code, so the scalar fallback and the
#     test drivers must NOT be in it - otherwise the fallback is not a fallback, it is the same bet with
#     worse opt-out.  `cpu_require_expert_support()` therefore refuses at startup on a CPU without the
#     features, and the fallback exists for tests, exactly as P2.S3 specifies.
# Plan v0.3 P6: speculation components that need no GPU.
add_library(strata_spec STATIC src/spec/suffix_drafter.cpp src/spec/controller.cpp src/spec/draft_policy.cpp
            src/spec/draft_source.cpp)
target_include_directories(strata_spec PUBLIC include)
if(STRATA_BUILD_TESTS)
  add_executable(window_evict_test src/core/window_evict_test.cpp)
  target_include_directories(window_evict_test PRIVATE include)
  add_test(NAME window_evict_test COMMAND window_evict_test)
  add_executable(adapt_decay_test src/program/adapt_decay_test.cpp)
  target_include_directories(adapt_decay_test PRIVATE include)
  add_test(NAME adapt_decay_test COMMAND adapt_decay_test)
  add_executable(auto_cache_test src/core/auto_cache_test.cpp)
  target_include_directories(auto_cache_test PRIVATE include)
  add_test(NAME auto_cache_test COMMAND auto_cache_test)
  add_executable(pcie_share_rule_test src/core/pcie_share_rule_test.cpp)
  target_include_directories(pcie_share_rule_test PRIVATE include)
  add_test(NAME pcie_share_rule_test COMMAND pcie_share_rule_test)
  add_executable(adapt_async_default_test src/core/adapt_async_default_test.cpp)
  target_include_directories(adapt_async_default_test PRIVATE include)
  add_test(NAME adapt_async_default_test COMMAND adapt_async_default_test)
  add_executable(ple_residency_test src/core/ple_residency_test.cpp)
  target_include_directories(ple_residency_test PRIVATE include)
  add_test(NAME ple_residency_test COMMAND ple_residency_test)
  add_executable(nv_defaults_test src/core/nv_defaults_test.cpp src/core/nv_defaults.cpp)
  target_include_directories(nv_defaults_test PRIVATE include)
  add_test(NAME nv_defaults_test COMMAND nv_defaults_test)
  add_executable(suffix_drafter_test src/spec/suffix_drafter_test.cpp)
  target_link_libraries(suffix_drafter_test PRIVATE strata_spec)
  add_test(NAME suffix_drafter_test COMMAND suffix_drafter_test)
  add_executable(controller_test src/spec/controller_test.cpp)
  target_link_libraries(controller_test PRIVATE strata_spec)
  add_test(NAME controller_test COMMAND controller_test)
  add_executable(draft_policy_test src/spec/draft_policy_test.cpp)
  target_link_libraries(draft_policy_test PRIVATE strata_spec)
  add_test(NAME draft_policy_test COMMAND draft_policy_test)
endif()
if(STRATA_BUILD_TESTS OR STRATA_BUILD_CONVERSATION_TESTS)
  # the serve loop's conversation-cache retention policy (header-only, no library to link)
  add_executable(conv_cache_test src/program/conv_cache_test.cpp)
  target_include_directories(conv_cache_test PRIVATE include)
  add_test(NAME conv_cache_test COMMAND conv_cache_test)
  # which path computes the draft layer's prompt K/V (header-only, CPU)
  add_executable(draft_kv_plan_test src/program/draft_kv_plan_test.cpp)
  target_include_directories(draft_kv_plan_test PRIVATE include)
  add_test(NAME draft_kv_plan_test COMMAND draft_kv_plan_test)
  # coupled draft sampling's counter and penalty-ring arithmetic (header-only, CPU)
  add_executable(coupled_draft_test src/core/coupled_draft_test.cpp)
  target_include_directories(coupled_draft_test PRIVATE include)
  add_test(NAME coupled_draft_test COMMAND coupled_draft_test)
  # probabilistic draft acceptance (STRATA_SPEC_PROB): rejection sampling preserves p - chi-square over 2M trials per case (CPU)
  add_executable(spec_prob_test src/core/spec_prob_test.cpp)
  target_include_directories(spec_prob_test PRIVATE include)
  add_test(NAME spec_prob_test COMMAND spec_prob_test)
endif()

add_library(strata_kernels_cpu STATIC src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp
            src/kernels/cpu/expert_layout.cpp src/kernels/cpu/q2_avx2.cpp)
target_include_directories(strata_kernels_cpu PUBLIC include)
# Plan v0.3 P6: only the kernel files get the AVX-512 / AVX2 flags; the pool and the probe run on any x86-64,
# and the engine picks the kernels at run time (cpu_avx512_ok).
# GCC/Clang spell it differently, and `-mavx512vnni` alone is not enough: VBMI is a separate feature bit.
set_source_files_properties(src/kernels/cpu/expert.cpp
                            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
# AVX-VNNI (Alder Lake, Sapphire Rapids and later): the AVX-2 expert kernels' vpdpwssd / vpdpbusd copies (q2_avx2.cpp,
# iq_avx2.cpp; *_rows.inl), taken at run time where cpu_avxvnni_ok().  GCC and Clang compile only those functions with
# target("avxvnni") - never -mavxvnni on a file, which would let the compiler use it anywhere - and MSVC takes the
# intrinsics under /arch:AVX2.  A compiler without them (GCC < 11, Clang < 12) builds the files without the copies.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma -mf16c")
check_cxx_source_compiles([=[
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avxvnni")))
#endif
static __m256i dots(__m256i a, __m256i b) { return _mm256_dpbusd_avx_epi32(_mm256_dpwssd_avx_epi32(a, a, b), a, b); }
int main() {
    volatile int x = 1;
    return _mm256_extract_epi32(dots(_mm256_set1_epi32(x), _mm256_set1_epi32(x)), 0) == 0;
}
]=] STRATA_HAVE_AVXVNNI_INTRINSICS)
unset(CMAKE_REQUIRED_FLAGS)
if(STRATA_HAVE_AVXVNNI_INTRINSICS)
  set(_strata_avxvnni STRATA_AVXVNNI=1)
else()
  set(_strata_avxvnni STRATA_AVXVNNI=0)
endif()
set_property(SOURCE src/kernels/cpu/q2_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${_strata_avxvnni})
find_package(Threads REQUIRED)
target_link_libraries(strata_kernels_cpu PUBLIC Threads::Threads)

# ---- plan v0.3 P6: the i-quant models' CPU experts (IQ2_XS / IQ3_XXS files) ----
#
# The CPU computes a native expert with ggml-cpu's own dot products (`ggml_get_type_traits_cpu`), so an IQ
# weight means exactly what it means in llama.cpp.  ggml is built static from a llama.cpp checkout: a local one
# with -DSTRATA_GGML_DIR=..., otherwise the pinned upstream commit (third_party/ggml/VERSION.txt).
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")
if(STRATA_ORCA_Q4KS_MMQ AND NOT STRATA_NATIVE_EXPERTS)
  message(FATAL_ERROR "STRATA_ORCA_Q4KS_MMQ requires STRATA_NATIVE_EXPERTS=ON for GGML sources")
endif()
if(STRATA_NATIVE_EXPERTS)
  enable_language(C)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_STATIC ON CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  if(STRATA_GGML_DIR)
    add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  else()
    include(FetchContent)
    FetchContent_Declare(strata_llamacpp
      GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
      GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d
      GIT_SHALLOW FALSE)
    FetchContent_GetProperties(strata_llamacpp)
    if(NOT strata_llamacpp_POPULATED)
      FetchContent_Populate(strata_llamacpp)
    endif()
    add_subdirectory(${strata_llamacpp_SOURCE_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  endif()
  # part of strata_kernels_cpu (the pool calls these and they call its CPU probe: one library, so GNU ld's
  # one-pass static linking has no cycle to resolve)
  target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/native_expert.cpp src/kernels/cpu/iq_avx512.cpp
                                           src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/kq_avx2.cpp)
  target_include_directories(strata_kernels_cpu PRIVATE third_party/ggml)
  set_source_files_properties(src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS
                              "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  set_source_files_properties(src/kernels/cpu/kq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  set_property(SOURCE src/kernels/cpu/iq_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${_strata_avxvnni})
  target_link_libraries(strata_kernels_cpu PUBLIC ggml-cpu ggml-base)
  target_compile_definitions(strata_kernels_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)
  set(_strata_mmq_types q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0)
  # #296: OrcaRouter Q4_K_S's Q5_0 experts (Q4_K and Q5_1 come with STRATA_MMQ_KQUANTS on CUDA, S5's branch)
  if(STRATA_ORCA_Q4KS_MMQ)
    list(APPEND _strata_mmq_types q5_0)
  endif()
  if(STRATA_ENABLE_CUDA)
    # prompt-speed plan step 2b: llama.cpp's MMQ kernels (ggml-cuda, MIT) for the prompt path's experts, compiled
    # from the same checkout as ggml above - only mmq.cuh's instances for the packs' types and the quantizer, with
    # the few host symbols of ggml-cuda.cu they use in src/prefill/ggml_cuda_host.cu
    if(STRATA_GGML_DIR)
      set(_strata_gg ${STRATA_GGML_DIR}/ggml)
    else()
      set(_strata_gg ${strata_llamacpp_SOURCE_DIR}/ggml)
    endif()
    set(_strata_mmq ${_strata_gg}/src/ggml-cuda)
    # #136: the Q2_0 pack's prompt experts on Strata's own int8 kernels (moe_fused.cu, opt-in STRATA_PF_FUSED=1; CUDA
    # only: an sm_75 device or an sm_75-only build reports them unavailable)
    add_library(strata_mmq STATIC src/prefill/moe_mmq.cu src/prefill/moe_fused.cu src/prefill/moe_fused_iq.cu
                src/prefill/ggml_cuda_host.cu ${_strata_mmq}/quantize.cu)
    set(_strata_mmq_cuda_types ${_strata_mmq_types})
    foreach(_t IN LISTS _strata_mmq_cuda_types)
      target_sources(strata_mmq PRIVATE ${_strata_mmq}/template-instances/mmq-instance-${_t}.cu)
    endforeach()
    target_include_directories(strata_mmq PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq}
                               PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(strata_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
    # GGML's CUDA error reporting calls cuGetErrorString from the driver API.
    target_link_libraries(strata_mmq PUBLIC ggml-base ${_strata_gpu_runtime_target})
    target_link_libraries(strata_mmq PUBLIC CUDA::cuda_driver)
    if(STRATA_MMQ_KQUANTS)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_KQUANTS=1)
    endif()
    if(STRATA_ORCA_Q4KS_MMQ)
      target_compile_definitions(strata_mmq PRIVATE STRATA_ORCA_Q4KS_MMQ=1)
    endif()
    target_link_libraries(strata_prefill PUBLIC strata_mmq)
    target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_MMQ=1 STRATA_PREFILL_FUSED=1)
  endif()
endif()
