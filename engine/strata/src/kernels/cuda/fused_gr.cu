// src/kernels/cuda/fused_gr.cu - see include/strata/kernels/fused_gr.hpp.
#include "strata/kernels/q8_1_finite.hpp"   // #606: q8_1_ds
#include "strata/core/geometry.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <type_traits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace strata::kernels {
namespace {

constexpr int N = 2560;         // n_embd
constexpr int HC = 4;           // streams
constexpr int D = N * HC;       // 10240
constexpr int LR = 320;         // hc_lr
static_assert(N == strata::core::kCompiledGeometry.n_embd && HC == strata::core::kCompiledGeometry.hc && LR == strata::core::kCompiledGeometry.hc_lr, "fused_gr is compiled for core/geometry.hpp");
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int DOWN_BLOCKS = LR / WARPS;          // 40 blocks of 8 rows; one more for the inject rows
constexpr int UP_COLS = 32;                      // columns d per `up` block (x 4 streams = 128 rows)
constexpr int UP_BLOCKS = N / UP_COLS;           // 80

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// 8 bf16 packed in a uint4, unpacked to two float4 once when reused across tokens.
struct Bf16x8 { float4 a, b; };
__device__ __forceinline__ Bf16x8 unpack8(const uint4 w) {
    return {
        make_float4(__uint_as_float(w.x << 16), __uint_as_float(w.x & 0xffff0000u),
                    __uint_as_float(w.y << 16), __uint_as_float(w.y & 0xffff0000u)),
        make_float4(__uint_as_float(w.z << 16), __uint_as_float(w.z & 0xffff0000u),
                    __uint_as_float(w.w << 16), __uint_as_float(w.w & 0xffff0000u))
    };
}
__device__ __forceinline__ float dot8u(const Bf16x8& w, const float4 x0, const float4 x1) {
    float acc = 0.0f;
    acc = fmaf(w.a.x, x0.x, acc);
    acc = fmaf(w.a.y, x0.y, acc);
    acc = fmaf(w.a.z, x0.z, acc);
    acc = fmaf(w.a.w, x0.w, acc);
    acc = fmaf(w.b.x, x1.x, acc);
    acc = fmaf(w.b.y, x1.y, acc);
    acc = fmaf(w.b.z, x1.z, acc);
    acc = fmaf(w.b.w, x1.w, acc);
    return acc;
}
__device__ __forceinline__ float dot8u_ptr(const Bf16x8& w, const float* x) {
    const float4* x4 = reinterpret_cast<const float4*>(x);
    return dot8u(w, x4[0], x4[1]);
}

// 8 bf16 packed in a uint4 against 8 16-byte-aligned floats (same 8-FMA order).
__device__ __forceinline__ float dot8(const uint4 w, const float* x) {
    return dot8u_ptr(unpack8(w), x);
}

__device__ __forceinline__ float dot8q(const uint32_t a, const uint32_t b, const float* x) {
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) acc = fmaf((float) (int8_t) (a >> (8 * i)), x[i], acc);
#pragma unroll
    for (int i = 0; i < 4; ++i) acc = fmaf((float) (int8_t) (b >> (8 * i)), x[4 + i], acc);
    return acc;
}
template <bool Q8>
__device__ __forceinline__ uint4 ldw(const uint16_t* wb, const float* sq, size_t e) {
    if constexpr (Q8) {
        if (sq != nullptr) {
            const uint2 v = __ldg(reinterpret_cast<const uint2*>(reinterpret_cast<const int8_t*>(wb) + e));
            return make_uint4(v.x, v.y, __float_as_uint(__ldg(sq + e / 32)), 0u);
        }
    }
    return __ldg(reinterpret_cast<const uint4*>(wb + e));
}
template <bool Q8>
__device__ __forceinline__ float dotw(const bool q8, const uint4 w, const float* x) {
    if constexpr (Q8) {
        if (q8) return __uint_as_float(w.z) * dot8q(w.x, w.y, x);
    }
    return dot8(w, x);
}

template <bool Q8 = false>
__global__ void __launch_bounds__(THREADS) gr_down_kernel(FusedGrArgs a) {
    __shared__ __align__(16) float xn[D];
    __shared__ float part[WARPS][HC];
    __shared__ float s_rs[HC];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    // 1. R' * w_norm into shared memory, and the per-stream sums of squares of R'.
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw[c], r.x); r.y = fmaf(b.y, gw[c], r.y);
            r.z = fmaf(b.z, gw[c], r.z); r.w = fmaf(b.w, gw[c], r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<float4*>(xn + i) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    __syncthreads();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = rsqrtf(s / (float) N + a.eps);
        if (blockIdx.x == 0) a.rs[t] = s_rs[t];
    }
    __syncthreads();
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
    __syncthreads();
    // 2. one warp per output row: 10240 bf16 = 1280 chunks of 8, 40 per lane.
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    if (inject_block && (a.w_inject == nullptr || warp >= HC)) return;
    const uint16_t* wmat = inject_block ? a.w_inject : a.w_down;
    const float* sq = (Q8 && !inject_block) ? a.s_down : nullptr;   // the inject rows stay bf16
    const bool q8 = Q8 && !inject_block;
    float acc = 0.0f;
#pragma unroll 4
    for (int j = lane; j < D / 8; j += 32) acc += dotw<Q8>(q8, ldw<Q8>(wmat, sq, (size_t) row * D + (size_t) j * 8), xn + j * 8);
    acc = warp_sum(acc);
    if (lane != 0) return;
    if (inject_block) {
        a.inject_out[row] = acc;
    } else {
        const float x = acc / (float) HC;
        a.lo[row] = x / (1.0f + __expf(-x));
    }
}

template <bool Q8 = false>
__global__ void __launch_bounds__(THREADS) gr_up_kernel(FusedGrArgs a) {
    __shared__ __align__(16) float lo[LR];
    __shared__ float g[HC][UP_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * UP_COLS;
    for (int k = t; k < LR; k += THREADS) lo[k] = a.lo[k];
    __syncthreads();
    // 128 rows (4 streams x 32 columns), 16 per warp: 320 bf16 = 40 chunks of 8.
    for (int r = warp; r < HC * UP_COLS; r += WARPS) {
        const int c = r / UP_COLS, dd = r - c * UP_COLS, i = c * N + d0 + dd;
        const bool q8 = Q8;
        float acc = dotw<Q8>(q8, ldw<Q8>(a.w_up, a.s_up, (size_t) i * LR + (size_t) lane * 8), lo + lane * 8);
        if (lane < LR / 8 - 32)
            acc += dotw<Q8>(q8, ldw<Q8>(a.w_up, a.s_up, (size_t) i * LR + (size_t) (32 + lane) * 8), lo + (32 + lane) * 8);
        acc = warp_sum(acc);
        if (lane == 0) {
            float rv = a.R[i];
            if (a.apply) {
                rv = fmaf(a.bo_prev[d0 + dd], 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC), rv);
                a.R_out[i] = rv;                       // this block owns column d0+dd of every stream
            }
            const float x = rv * a.w_norm[i] * a.rs[c];
            g[c][dd] = x * sigmoidf_(acc);
        }
    }
    __syncthreads();
    if (t < UP_COLS) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[c][t];
        a.mixed[d0 + t] = s / (float) HC;
    }
}

// ================================ plan v0.3 P6: T tokens, one weight read ================================
struct GrMulti {
    FusedGrArgs a[kFusedGrMaxT];
    float* xn;
    int T;
};


// the q8_1 image of `mixed`, written by the up kernels below. A q8_1 block is 32 columns and an
// up block owns UPM_COLS = 16, so the second of the two blocks that own a 32-column group (a per-group counter,
// incremented after the block's writes are fenced, reset by that block for the next launch) reads the 32 values back
// and quantizes them with native_quantize_q8_1_kernel's quantizer: one warp per token, the same XOR-tree max and sum,
// d = amax / 127, roundf(x / d), ds = (d, sum) - the bytes the separate launch writes.
struct GrQ81 { half2 ds; int8_t qs[32]; };
static_assert(2 * 16 == 32, "two up blocks per q8_1 group");
__device__ __forceinline__ void gr_q8_tail(const GrMulti& m, int d0) {
    __shared__ int last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned* c = m.a[0].q8_cnt + d0 / 32;
        last = atomicAdd(c, 1u) == 1u;
        if (last) atomicExch(c, 0u);
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp >= m.T) return;
    const int c0 = (d0 / 32) * 32;
    const float xi = *reinterpret_cast<const volatile float*>(m.a[warp].mixed + c0 + lane);
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float d = q8_1_finite(amax / 127.0f);   // #606: as native_quantize_q8_1_kernel - finite blocks bit for bit
    const int8_t q = q8_1_quant(xi, d, amax);
    GrQ81* y = reinterpret_cast<GrQ81*>(m.a[warp].q8_mixed) + c0 / 32;
    y->qs[lane] = q;
    if (lane == 0) y->ds = q8_1_ds(d, sum);   // #606: clamped scale/sum - an unclamped pair NaN-poisons the dot path
}
// Step 1 of `gr_down_kernel`, one block per token, same threads and reduction order: rs[t] and xn[t] to global.
__global__ void __launch_bounds__(THREADS) gr_norm_multi_kernel(GrMulti m) {
    __shared__ float part[WARPS][HC];
    __shared__ float s_rs[HC];
    const FusedGrArgs& a = m.a[blockIdx.x];
    float* xn = m.xn + (size_t) blockIdx.x * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw[c], r.x); r.y = fmaf(b.y, gw[c], r.y);
            r.z = fmaf(b.z, gw[c], r.z); r.w = fmaf(b.w, gw[c], r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<float4*>(xn + i) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    __syncthreads();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = rsqrtf(s / (float) N + a.eps);
        a.rs[t] = s_rs[t];
    }
    __syncthreads();
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
}

// Step 2 of `gr_down_kernel` for T tokens.  One warp per row (so each lane accumulates the same chunks in the
// same order as the single-token kernel); per tile the lane's weight chunks are loaded BEFORE the activation
// tile is staged, so the DRAM and L2 traffic are in flight together.
//
// TILEV = xn floats per token staged at a time.  The tile only changes the staging granularity: the lane's chunk
// order (lane + 32*q within the tile, tiles ascending) is strictly increasing for either value, so the results are
// bitwise identical to `gr_down_kernel` for both.  2560 stages 320 chunks of 8 per tile (10 per lane).
template <int TILEV, int MAX_T = kFusedGrMaxT, bool Q8 = false>
__global__ void __launch_bounds__(THREADS) gr_down_multi_kernel(GrMulti m) {
    constexpr int TQ = TILEV / 8 / 32;      // uint4 weight chunks per lane per tile
    extern __shared__ __align__(16) float tile[];   // [T][TILEV]
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wmat = inject_block ? m.a[0].w_inject : m.a[0].w_down;
    const float* sq = (Q8 && !inject_block) ? m.a[0].s_down : nullptr;   // the inject rows stay bf16
    const bool q8 = Q8 && !inject_block;
    const size_t row0 = (size_t) (active ? row : 0) * D;
    float acc[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
    for (int base = 0; base < D; base += TILEV) {
        uint4 wv[TQ];
        if (active) {
#pragma unroll
            for (int q = 0; q < TQ; ++q) wv[q] = ldw<Q8>(wmat, sq, row0 + (size_t) base + (size_t) (lane + 32 * q) * 8);
        }
        __syncthreads();                                   // the previous tile is consumed
        const float4* src4 = reinterpret_cast<const float4*>(m.xn);
        float4* tile4 = reinterpret_cast<float4*>(tile);
        for (int i = t; i < T * (TILEV / 4); i += THREADS) {
            const int k = i / (TILEV / 4), off = i - k * (TILEV / 4);
            tile4[i] = src4[((size_t) k * D + base) / 4 + off];
        }
        __syncthreads();
        if (!active) continue;
#pragma unroll
        for (int q = 0; q < TQ; ++q) {
            const int j = lane + 32 * q;
#pragma unroll
            for (int k = 0; k < MAX_T; ++k)
                if (k < T) acc[k] += dotw<Q8>(q8, wv[q], tile + k * TILEV + j * 8);
        }
    }
    if (!active) return;
    float s[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) s[k] = k < T ? warp_sum(acc[k]) : 0.0f;
    // lane k writes token k (every lane holds every sum after the xor reduction)
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if (k >= T || lane != k) continue;
        if (inject_block) {
            m.a[k].inject_out[row] = s[k];
        } else {
            const float x = s[k] / (float) HC;
            m.a[k].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

constexpr int UPM_COLS = 16;                      // columns per block (x 4 streams = 64 rows, 8 per warp)
constexpr int UPM_BLOCKS = N / UPM_COLS;          // 160

// `gr_up_kernel` for T tokens: each row of w_up read once; the T dots reduced by xor so every lane holds every
// sum, and lane k runs token k's epilogue - the T epilogues in parallel instead of one after another.
template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false, bool Q8 = false>
__global__ void __launch_bounds__(THREADS) gr_up_multi_kernel(GrMulti m) {
    __shared__ __align__(16) float lo[MAX_T][LR];
    __shared__ float g[MAX_T][HC][UPM_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const int d0 = blockIdx.x * UPM_COLS;
    for (int i = t; i < T * LR; i += THREADS) lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    __syncthreads();
    for (int r = warp; r < HC * UPM_COLS; r += WARPS) {
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        Bf16x8 wa{}, wb{};
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
        wa = unpack8(__ldg(w4 + lane));
        wb = unpack8(lane < LR / 8 - 32 ? __ldg(w4 + 32 + lane) : make_uint4(0, 0, 0, 0));
        // the epilogue inputs of this lane's token, fetched while the dots run
        float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            rsc = a.rs[c];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            if (!EXACT_T && k >= T) break;
            float acc;
            acc = dot8u_ptr(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += dot8u_ptr(wb, lo[k] + (32 + lane) * 8);
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsc;
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);
}

// ================================ #315: split / staged - the same arithmetic, split differently ==================
// The multi read's variants beside the plain read (0.1.31's default; not main's opt-in STRATA_GR_V3 read, which sums
// in another order).  Every variant computes each output with exactly the plain read's operations in its order (so
// bitwise the plain read and the single-token read); only who does what, and when, changes.  `fused_gr_check`
// compares them on the card before a verify window uses them (STRATA_HC_SPLIT below).
//  - the norm (split and staged): one block per token AND stream instead of one per token.  Each thread visits
//    exactly the elements, in the order, it visited for that stream in the plain read, so every sum of squares and rs
//    is the same; it stores R' * w_norm and scales it in place, with the plain read's two roundings.
//    Before this scratch reuse, the split norm measured 0.23 ms per 4-token round (96 launches) on an RTX 4080
//    SUPER, compared with 0.58 ms for the plain read's norm.
//  - the down projection, split: the plain read's kernel.
//  - the down projection, staged: the plain read's 8 rows per block and lane order, but the activations arrive by
//    cp.async in half-stream tiles, two in flight, so no thread waits on a chain of loads, and they sit in shared
//    memory as two planes (floats 0-3 and 4-7 of every 8), so a warp's reads hit no bank twice.  The next tile's
//    weights are loaded while the current one is used.  A tile is 160 = 5 x 32 chunks of 8, so a lane still
//    accumulates its chunks lane + 32 q in ascending order, as in the plain read with either tile.
constexpr int kHcPlain = 1, kHcSplit = 2, kHcStaged = 3;
constexpr int H_TILE = 1280;                          // staged tile: half a stream = 160 chunks of 8, 5 per lane
constexpr int HQ = H_TILE / 8 / 32;
constexpr int N_HTILES = D / H_TILE;                  // 8
static_assert(N % H_TILE == 0, "a staged tile never straddles two streams");
static_assert(H_TILE % 256 == 0, "a staged tile holds whole rounds of 32 chunks: the plain read's lane order");

__global__ void __launch_bounds__(THREADS) gr_norm_split_kernel(GrMulti m) {
    __shared__ float part[WARPS];
    __shared__ float s_rs;
    const FusedGrArgs& a = m.a[blockIdx.x];
    const int c = blockIdx.y;
    float* xn = m.xn + (size_t) blockIdx.x * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss = 0.0f;
    for (int i = t * 4; i < D; i += THREADS * 4) {
        if (i / N != c) continue;                       // another stream's element: another block's
        const int d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
            r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
        }
        const float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
        ss += sq;
        // Retain the unscaled product in the existing scratch instead of reloading R
        // and recomputing the pending write after the reduction.
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        *reinterpret_cast<float4*>(xn + i) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
    const float v = warp_sum(ss);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w];
        s_rs = rsqrtf(s / (float) N + a.eps);
        a.rs[c] = s_rs;
    }
    __syncthreads();
    const float rs = s_rs;
    for (int d = t; d < N; d += THREADS) xn[c * N + d] *= rs;
}

// 143-amdk: gr_norm_split_kernel with the loads of a thread's (at most three) float4 issued together and the scaled
// product written once from registers.  The same elements go through the same operations in the same order (the
// thread's own sum of squares ascends in k; the warp sum, part[] sum and rsqrt are the kernel's), so rs and xn are
// bitwise the same; what goes is the loop's one-iteration-at-a-time loads and the second pass that read xn back from
// global memory (a write, a barrier, a read: two memory latencies).
__global__ void __launch_bounds__(THREADS) gr_norm_split_fast_kernel(GrMulti m) {
    __shared__ float part[WARPS];
    __shared__ float s_rs;
    const FusedGrArgs& a = m.a[blockIdx.x];
    const int c = blockIdx.y;
    float* xn = m.xn + (size_t) blockIdx.x * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    // the thread's float4 of stream c: i = 4 t + 1024 k in [c N, (c + 1) N), ascending k; N / 1024 = 2.5 -> at most 3
    constexpr int KS = 3;
    int k0 = (c * N - 4 * t + THREADS * 4 - 1) / (THREADS * 4);
    if (k0 < 0) k0 = 0;
    float4 r[KS], g[KS];
    bool ok[KS];
#pragma unroll
    for (int j = 0; j < KS; ++j) {
        const int i = t * 4 + (k0 + j) * THREADS * 4;
        ok[j] = i < (c + 1) * N;
        if (ok[j]) {
            r[j] = *reinterpret_cast<const float4*>(a.R + i);
            g[j] = *reinterpret_cast<const float4*>(a.w_norm + i);
            if (a.apply) {
                const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + (i - c * N));
                r[j].x = fmaf(b.x, gw, r[j].x); r[j].y = fmaf(b.y, gw, r[j].y);
                r[j].z = fmaf(b.z, gw, r[j].z); r[j].w = fmaf(b.w, gw, r[j].w);
            }
        }
    }
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < KS; ++j) {
        if (ok[j]) {
            const float sq = r[j].x * r[j].x + r[j].y * r[j].y + r[j].z * r[j].z + r[j].w * r[j].w;
            ss += sq;
            r[j] = make_float4(r[j].x * g[j].x, r[j].y * g[j].y, r[j].z * g[j].z, r[j].w * g[j].w);
        }
    }
    const float v = warp_sum(ss);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w];
        s_rs = rsqrtf(s / (float) N + a.eps);
        a.rs[c] = s_rs;
    }
    __syncthreads();
    const float rs = s_rs;
#pragma unroll
    for (int j = 0; j < KS; ++j)
        if (ok[j])
            *reinterpret_cast<float4*>(xn + t * 4 + (k0 + j) * THREADS * 4) =
                make_float4(r[j].x * rs, r[j].y * rs, r[j].z * rs, r[j].w * rs);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define STRATA_GR_CP_ASYNC 1
#endif
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
#if defined(STRATA_GR_CP_ASYNC)
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(gmem) : "memory");
#else
    *reinterpret_cast<float4*>(smem) = *reinterpret_cast<const float4*>(gmem);
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if defined(STRATA_GR_CP_ASYNC)
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void cp_async_wait1() {
#if defined(STRATA_GR_CP_ASYNC)
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void cp_async_wait0() {
#if defined(STRATA_GR_CP_ASYNC)
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}

// Stage tile `h` of every token into `buf`: [T][2 planes][160 chunks] float4, plane 0 = floats 0-3 of a chunk.
__device__ __forceinline__ void stage_htile(const GrMulti& m, int T, int h, float4* buf, int t, int k0 = 0) {
    for (int i = t; i < T * (H_TILE / 4); i += THREADS) {
        const int k = i / (H_TILE / 4), s4 = i - k * (H_TILE / 4);   // s4: float4 of the tile, chunk s4/2, half s4&1
        const float* src = m.xn + (size_t) (k0 + k) * D + (size_t) h * H_TILE + (size_t) s4 * 4;
        cp_async16(buf + (size_t) k * (H_TILE / 4) + (s4 & 1) * (H_TILE / 8) + (s4 >> 1), src);
    }
}

template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(THREADS) gr_down_staged_kernel(GrMulti m) {
    extern __shared__ __align__(16) float4 hbuf[];      // 2 buffers x [T][2][160] float4
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    // a launch of EXACT_T groups of MAX_T tokens, one per blockIdx.y (gridDim.y 1: the whole launch); every
    // (row, token) is summed by one warp exactly as before - only more blocks share the card (#down-tg)
    const int k0 = EXACT_T ? (int) blockIdx.y * MAX_T : 0;
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    const size_t buf_f4 = (size_t) T * (H_TILE / 4);    // float4 per buffer (the second one follows the first)
    float acc[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
    uint4 wv[HQ], wnext[HQ];
    if (active) {
#pragma unroll
        for (int q = 0; q < HQ; ++q) wv[q] = __ldg(w4 + lane + 32 * q);
    }
    stage_htile(m, T, 0, hbuf, t, k0);
    cp_async_commit();
    stage_htile(m, T, 1, hbuf + buf_f4, t, k0);
    cp_async_commit();
#pragma unroll 1
    for (int h = 0; h < N_HTILES; ++h) {
        if (active && h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wnext[q] = __ldg(w4 + (h + 1) * (H_TILE / 8) + lane + 32 * q);
        }
        if (h + 1 < N_HTILES) cp_async_wait1();         // tile h has landed (h + 1 may still be on its way)
        else cp_async_wait0();
        __syncthreads();
        const float4* cur = hbuf + (h & 1) * buf_f4;
        if (active) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) {
                const int j = lane + 32 * q;
                const Bf16x8 wvq = unpack8(wv[q]);
#pragma unroll
                for (int k = 0; k < MAX_T; ++k) {
                    if (EXACT_T || k < T) {
                        const float4* pk = cur + (size_t) k * (H_TILE / 4);
                        acc[k] += dot8u(wvq, pk[j], pk[H_TILE / 8 + j]);
                    }
                }
            }
        }
        __syncthreads();                                // every warp is done with buffer h & 1
        if (h + 2 < N_HTILES) stage_htile(m, T, h + 2, hbuf + (h & 1) * buf_f4, t, k0);
        cp_async_commit();                              // an empty group at the end keeps the wait counts simple
        if (h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wv[q] = wnext[q];
        }
    }
    if (!active) return;
    float s[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) s[k] = (EXACT_T || k < T) ? warp_sum(acc[k]) : 0.0f;
    // lane k writes token k (every lane holds every sum after the xor reduction)
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if ((!EXACT_T && k >= T) || lane != k) continue;
        if (inject_block) {
            m.a[k0 + k].inject_out[row] = s[k];
        } else {
            const float x = s[k] / (float) HC;
            m.a[k0 + k].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

// The tokens a down kernel may carry in one launch on the current card, and the plain read's tile: the plain read
// stages n_tok * TILEV floats (2560), staged two tiles of n_tok * H_TILE.  The shared-memory
// opt-in is set here once per device.
int down_chunk(bool staged, int* tile_out) {
    static bool attr[64] = {};
    static int chunk[64] = {};   // tokens the plain down kernel may carry in one launch on this card
    static int chunk_staged[64] = {};  // the same for staged
    static int tile[64] = {};    // the plain down kernel's TILEV on this card
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) {
        *tile_out = 2560;
        return kFusedGrMaxT;
    }
    if (!attr[dev]) {
        int optin = 0;
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        // the down kernel stages n_tok*TILEV floats of dynamic shared memory - 80 KB at the full 8 tokens of the
        // CUDA tile.  Cards whose opt-in is still below that (or that report no opt-in at all) slice the tokens;
        // the down kernel's outputs (lo, inject_out) are strictly per-token, so the chunk boundaries are safe, and
        // the up kernel below still sees every token of the batch in one launch.
        const int tv = 2560;
        tile[dev] = tv;
        int want = (int) (kFusedGrMaxT * tv * sizeof(float));
        if (optin > 0 && want > optin) want = optin;
        cudaFuncSetAttribute(gr_down_multi_kernel<2560>, cudaFuncAttributeMaxDynamicSharedMemorySize, want);
        cudaFuncSetAttribute(gr_down_multi_kernel<2560, kFusedGrMaxT, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, want);
        cudaFuncSetAttribute(gr_down_multi_kernel<2560, 4, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, want);
        auto set_staged_attr = [&](auto kernel, int max_t) {
            int ws = (int) (2 * max_t * H_TILE * sizeof(float));
            if (optin > 0 && ws > optin) ws = optin;
            if (ws > 48 * 1024)
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ws);
            cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        };
        set_staged_attr(gr_down_staged_kernel<1, true>, 1);
        set_staged_attr(gr_down_staged_kernel<2, true>, 2);
        set_staged_attr(gr_down_staged_kernel<3, true>, 3);
        set_staged_attr(gr_down_staged_kernel<4, true>, 4);
        set_staged_attr(gr_down_staged_kernel<5, true>, 5);
        set_staged_attr(gr_down_staged_kernel<6, true>, 6);
        set_staged_attr(gr_down_staged_kernel<4, false>, 4);
        set_staged_attr(gr_down_staged_kernel<kFusedGrMaxT, false>, kFusedGrMaxT);
        cudaGetLastError();      // drop any error the attempt left behind
        int per_block = 0;
        cudaDeviceGetAttribute(&per_block, cudaDevAttrMaxSharedMemoryPerBlock, dev);
        const int usable = optin > 0 ? optin : per_block;
        const int capacity = usable / (int) (tv * sizeof(float));
        chunk[dev] = capacity < 1 ? 1 : (capacity > kFusedGrMaxT ? kFusedGrMaxT : capacity);
        const int capacity_staged = usable / (int) (2 * H_TILE * sizeof(float));
        chunk_staged[dev] = capacity_staged < 1 ? 1 : (capacity_staged > kFusedGrMaxT ? kFusedGrMaxT : capacity_staged);
        attr[dev] = true;
    }
    *tile_out = tile[dev] ? tile[dev] : 2560;
    return staged ? chunk_staged[dev] : chunk[dev];
}

// The multi read as `variant` (kHcPlain, kHcSplit or kHcStaged): the norm, the down projection in launches of as
// many tokens as fit the card, the up projection; the profile's stamps after the norm and after the down projection.
// kHcPlain is the default read exactly as before (never main's opt-in STRATA_GR_V3 path).

bool norm_fast_on() {
    return false;
}

void launch_multi(const GrMulti& m, int variant, cudaStream_t st, unsigned long long* stamp_buf, int stamp_i0) {
    const int n_tok = m.T;
    if (variant >= kHcSplit) {
        if (norm_fast_on()) gr_norm_split_fast_kernel<<<dim3((unsigned) n_tok, HC), THREADS, 0, st>>>(m);
        else gr_norm_split_kernel<<<dim3((unsigned) n_tok, HC), THREADS, 0, st>>>(m);
    } else gr_norm_multi_kernel<<<n_tok, THREADS, 0, st>>>(m);
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, (void*) st);
    const bool staged = variant >= kHcStaged;
    int tv = 2560;
    const int chunk_tok = down_chunk(staged, &tv);
    const size_t per_tok = (staged ? (size_t) 2 * H_TILE : (size_t) tv) * sizeof(float);
    static const bool no_multi_gr = [] {
        const char* no = std::getenv("STRATA_NO_MULTI_GR");
        return no != nullptr && no[0] != '\0' && no[0] != '0';
    }();
    const bool exact_t = !no_multi_gr;
    for (int c0 = 0; c0 < n_tok; c0 += chunk_tok) {
        const int ct = n_tok - c0 < chunk_tok ? n_tok - c0 : chunk_tok;
        GrMulti c{};
        if (ct == n_tok) {
            c = m;
        } else {
            c.xn = m.xn + (size_t) c0 * D;
            c.T = ct;
            for (int k = 0; k < ct; ++k) c.a[k] = m.a[c0 + k];
        }
        const size_t smem = (size_t) ct * per_tok;
        if (staged) {
            // #783 PR-g (stuchapin909): a launch of exactly ct <= 6 tokens is its own instantiation, the loop bounds
            // are compile-time (the same sums in the same order); STRATA_NO_MULTI_GR=1 keeps the generic kernels
            if (exact_t && ct == 1) gr_down_staged_kernel<1, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 2) gr_down_staged_kernel<2, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 3) gr_down_staged_kernel<3, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 4) gr_down_staged_kernel<4, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 5) gr_down_staged_kernel<5, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 6) gr_down_staged_kernel<6, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else gr_down_staged_kernel<><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
        } else {
            gr_down_multi_kernel<2560><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
        }
    }
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, (void*) st);
    switch (no_multi_gr ? kFusedGrMaxT : n_tok) {
        case 1: gr_up_multi_kernel<1, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        case 2: gr_up_multi_kernel<2, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        case 3: gr_up_multi_kernel<3, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        case 4: gr_up_multi_kernel<4, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        case 5: gr_up_multi_kernel<5, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        case 6: gr_up_multi_kernel<6, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        default: gr_up_multi_kernel<kFusedGrMaxT, false><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
    }
}

/// STRATA_HC_SPLIT: unset or 2 = the newest the check accepts (staged), 1 = at most split, 0 = the plain read
int env_variant() {
    const char* e = std::getenv("STRATA_HC_SPLIT");
    if (e == nullptr || e[0] == '\0') return kHcStaged;
    if (e[0] == '0') return kHcPlain;
    if (e[0] == '1') return kHcSplit;
    return kHcStaged;
}

// per device: the variant `fused_gr_check` chose (0 = not checked yet)
std::atomic<int> g_variant[64];


}  // namespace

bool fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream, unsigned long long* stamp_buf,
                         int stamp_i0) {
    if (n_tok < 1 || n_tok > kFusedGrMaxT || xn_scratch == nullptr) {
        std::fprintf(stderr, "fused_gr_read_multi: invalid arguments\n");
        std::exit(1);
    }
    GrMulti m;
    for (int t = 0; t < n_tok; ++t) {
        m.a[t] = a[t];
        const FusedGrArgs& x = a[t];
        if (!x.R || !x.w_norm || !(x.w_down && x.w_up) || !x.lo || !x.rs || !x.mixed || (x.w_inject && !x.inject_out) ||
            (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) || x.w_down != a[0].w_down || x.w_up != a[0].w_up ||
            x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm) {
            std::fprintf(stderr, "fused_gr_read_multi: invalid arguments for token %d\n", t);
            std::exit(1);
        }
    }
    // every token needs its q8_1 destination and token 0 the counters; otherwise none is written
    bool q8 = a[0].q8_cnt != nullptr;
    for (int t = 0; t < n_tok; ++t) q8 = q8 && a[t].q8_mixed != nullptr;
    if (!q8)
        for (int t = 0; t < n_tok; ++t) m.a[t].q8_mixed = nullptr;
    m.xn = xn_scratch;
    m.T = n_tok;
    cudaStream_t st = (cudaStream_t) stream;
    // the default read: v1, or the bitwise-equal v2 / v3 this card's check accepted
    launch_multi(m, fused_gr_variant(), st, stamp_buf, stamp_i0);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_read_multi: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return q8;
}

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    return n_embd == N && hc == HC && hc_lr == LR;
}

void fused_gr_read(const FusedGrArgs& a, void* stream) {
    if (!a.R || !a.w_norm || !a.w_down || !a.w_up || !a.lo || !a.rs || !a.mixed ||
        (a.w_inject && !a.inject_out) || (a.apply && (!a.bo_prev || !a.inj_prev || !a.R_out)) ||
        (a.apply && a.inj_prev == a.inject_out)) {
        std::fprintf(stderr, "fused_gr_read: invalid arguments\n");
        std::exit(1);
    }
    cudaStream_t st = (cudaStream_t) stream;
    gr_down_kernel<<<DOWN_BLOCKS + 1, THREADS, 0, st>>>(a);
    gr_up_kernel<<<UP_BLOCKS, THREADS, 0, st>>>(a);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_read: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}


namespace {

/// The plain read, split, staged and (for one token) the single-token read on random bf16 weights and random inputs,
/// 1..8 tokens, with and without the pending write; every output of split and staged compared with the plain read's
/// bit for bit (and the plain read's with the single-token read's).  `why[v]` gets the first difference of variant
/// v; false if the check itself could not run.
bool fused_gr_selftest(bool ok_variant[4], std::string why[4]) {
    constexpr int TM = kFusedGrMaxT;
    constexpr int NV = 4;                             // sets: 0 = plain, 1 = split, 2 = staged, 3 = single-token
    std::mt19937 rng(20260930u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> h_down((size_t) LR * D), h_up((size_t) D * LR), h_inj((size_t) HC * D);
    std::vector<float> h_norm(D), h_R((size_t) TM * D), h_bo((size_t) TM * N), h_ip((size_t) TM * HC);
    for (auto& w : h_down) w = bf16_from_f32(0.02f * nd(rng));
    for (auto& w : h_up) w = bf16_from_f32(0.05f * nd(rng));
    for (auto& w : h_inj) w = bf16_from_f32(0.02f * nd(rng));
    for (auto& w : h_norm) w = 1.0f + 0.1f * nd(rng);
    for (auto& x : h_R) x = nd(rng);
    for (auto& x : h_bo) x = 0.5f * nd(rng);
    for (auto& x : h_ip) x = 2.0f * nd(rng);
    for (int v = 0; v < 4; ++v) { ok_variant[v] = v == 1; why[v].clear(); }

    const size_t n_set = (size_t) TM * (D + D + LR + HC + HC + N);   // R_out, xn, lo, rs, inject, mixed (floats)
    const size_t bytes = h_down.size() * 2 + h_up.size() * 2 + h_inj.size() * 2 +
                         (h_norm.size() + h_R.size() + h_bo.size() + h_ip.size() + NV * n_set) * 4 + 32 * 256;
    uint8_t* base = nullptr;
    if (cudaMalloc((void**) &base, bytes) != cudaSuccess) {
        cudaGetLastError();
        why[0] = "no room for the check (" + std::to_string(bytes >> 20) + " MiB)";
        return false;
    }
    size_t off = 0;
    auto take = [&](size_t b) { void* p = base + off; off += (b + 255) / 256 * 256; return p; };
    uint16_t* d_down = (uint16_t*) take(h_down.size() * 2);
    uint16_t* d_up = (uint16_t*) take(h_up.size() * 2);
    uint16_t* d_inj = (uint16_t*) take(h_inj.size() * 2);
    float* d_norm = (float*) take(h_norm.size() * 4);
    float* d_R = (float*) take(h_R.size() * 4);
    float* d_bo = (float*) take(h_bo.size() * 4);
    float* d_ip = (float*) take(h_ip.size() * 4);
    struct Set { float *R_out, *xn, *lo, *rs, *inj, *mixed; } set[NV];
    for (Set& x : set) {
        x.R_out = (float*) take((size_t) TM * D * 4); x.xn = (float*) take((size_t) TM * D * 4);
        x.lo = (float*) take((size_t) TM * LR * 4); x.rs = (float*) take((size_t) TM * HC * 4);
        x.inj = (float*) take((size_t) TM * HC * 4); x.mixed = (float*) take((size_t) TM * N * 4);
    }
    cudaStream_t st = nullptr;
    bool ok = off <= bytes && cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) == cudaSuccess &&
              cudaMemcpyAsync(d_down, h_down.data(), h_down.size() * 2, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_up, h_up.data(), h_up.size() * 2, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_inj, h_inj.data(), h_inj.size() * 2, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_norm, h_norm.data(), h_norm.size() * 4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_R, h_R.data(), h_R.size() * 4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_bo, h_bo.data(), h_bo.size() * 4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_ip, h_ip.data(), h_ip.size() * 4, cudaMemcpyHostToDevice, st) == cudaSuccess;
    if (!ok) why[0] = "setting up the check failed";
    std::vector<float> h1, h2;
    // true when equal; false with the first difference in `w` (or a read-back failure in `ok`)
    auto same = [&](const float* d1, const float* d2, size_t n, const char* what, int T, int apply, std::string& w) {
        h1.resize(n);
        h2.resize(n);
        if (cudaMemcpyAsync(h1.data(), d1, n * 4, cudaMemcpyDeviceToHost, st) != cudaSuccess ||
            cudaMemcpyAsync(h2.data(), d2, n * 4, cudaMemcpyDeviceToHost, st) != cudaSuccess ||
            cudaStreamSynchronize(st) != cudaSuccess) {
            why[0] = "reading back the check failed";
            ok = false;
            return false;
        }
        for (size_t i = 0; i < n; ++i) {
            uint32_t u1, u2;
            std::memcpy(&u1, &h1[i], 4);
            std::memcpy(&u2, &h2[i], 4);
            if (u1 != u2) {
                char buf[192];
                std::snprintf(buf, sizeof buf, "%s differs for %d token(s)%s at %zu (%08x, not %08x)", what, T,
                              apply ? " with the pending write" : "", i, u2, u1);
                w = buf;
                return false;
            }
        }
        return true;
    };
    auto all_same = [&](const Set& s1, const Set& s2, int T, int apply, std::string& w) {
        return same(s1.lo, s2.lo, (size_t) T * LR, "lo", T, apply, w) &&
               same(s1.rs, s2.rs, (size_t) T * HC, "rs", T, apply, w) &&
               same(s1.inj, s2.inj, (size_t) T * HC, "inject", T, apply, w) &&
               same(s1.mixed, s2.mixed, (size_t) T * N, "mixed", T, apply, w) &&
               same(s1.R_out, s2.R_out, (size_t) T * D, "R", T, apply, w);
    };
    ok_variant[2] = ok_variant[3] = ok;
    bool single_ok = ok;
    for (int apply = 0; apply < 2 && ok; ++apply) {
        for (int T = 1; T <= TM && ok; ++T) {
            FusedGrArgs a[NV][TM];
            for (int v = 0; v < NV; ++v) {
                const size_t span = (size_t) ((uint8_t*) (set[v].mixed + (size_t) TM * N) - (uint8_t*) set[v].R_out);
                ok = ok && cudaMemsetAsync(set[v].R_out, 0xFF, span, st) == cudaSuccess;   // the whole set
                for (int k = 0; k < T; ++k) {
                    FusedGrArgs& x = a[v][k];
                    x.R = d_R + (size_t) k * D; x.R_out = set[v].R_out + (size_t) k * D; x.apply = apply != 0;
                    x.bo_prev = d_bo + (size_t) k * N; x.inj_prev = d_ip + (size_t) k * HC;
                    x.w_norm = d_norm; x.w_down = d_down; x.w_up = d_up; x.w_inject = d_inj; x.eps = 1e-6f;
                    x.lo = set[v].lo + (size_t) k * LR; x.rs = set[v].rs + (size_t) k * HC;
                    x.inject_out = set[v].inj + (size_t) k * HC; x.mixed = set[v].mixed + (size_t) k * N;
                }
            }
            for (int v = 0; v < 3; ++v) {
                GrMulti m;
                for (int k = 0; k < T; ++k) m.a[k] = a[v][k];
                m.xn = set[v].xn;
                m.T = T;
                launch_multi(m, v + 1, st, nullptr, 0);
            }
            if (T == 1) fused_gr_read(a[3][0], st);
            if (cudaGetLastError() != cudaSuccess || cudaStreamSynchronize(st) != cudaSuccess) {
                why[0] = "a kernel of the check failed";
                ok = false;
                break;
            }
            if (ok_variant[2] && !all_same(set[0], set[1], T, apply, why[2])) ok_variant[2] = false;
            if (ok && ok_variant[3] && !all_same(set[0], set[2], T, apply, why[3])) ok_variant[3] = false;
            if (ok && T == 1 && single_ok && !all_same(set[3], set[0], T, apply, why[1])) single_ok = false;
        }
    }
    if (!single_ok && ok) {                            // the plain read itself disagrees with the single-token read
        why[2] = why[3] = "the plain read differs from the single-token read: " + why[1];
        ok_variant[2] = ok_variant[3] = false;
    }
    if (st != nullptr) {
        cudaStreamSynchronize(st);
        cudaStreamDestroy(st);
    }
    cudaFree(base);
    cudaGetLastError();
    if (!ok) ok_variant[2] = ok_variant[3] = false;
    return ok;
}

}  // namespace

int fused_gr_variant() {
    int dev = 0;
    cudaGetDevice(&dev);
    const int v = dev >= 0 && dev < 64 ? g_variant[dev].load() : 0;
    if (v > 0) return v;
    // not checked on this card: the plain read, unless STRATA_HC_SPLIT names a variant (a test such as gr_parity)
    const char* e = std::getenv("STRATA_HC_SPLIT");
    return e != nullptr && (e[0] == '1' || e[0] == '2') ? env_variant() : kHcPlain;
}

void fused_gr_check() {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64 || g_variant[dev].load() > 0) return;
    const int want = env_variant();
    if (want == kHcPlain) {
        g_variant[dev].store(kHcPlain);
        std::fprintf(stderr, "strata hc: CUDA%d: the hyper-connection read runs as the plain one (STRATA_HC_SPLIT=0)\n",
                     dev);
        return;
    }
    bool okv[4];
    std::string why[4];
    const bool ran = fused_gr_selftest(okv, why);
    int use = kHcPlain;
    if (want >= kHcStaged && okv[kHcStaged]) use = kHcStaged;
    else if (okv[kHcSplit]) use = kHcSplit;
    g_variant[dev].store(use);
    if (!ran)
        std::fprintf(stderr, "strata hc: CUDA%d: the check of split/staged could not run (%s)\n", dev, why[0].c_str());
    static const char* const name[4] = {"", "plain", "split", "staged"};
    for (int v = kHcStaged; v >= kHcSplit; --v)
        if (v <= want && !okv[v] && ran)
            std::fprintf(stderr, "strata hc: CUDA%d: the %s read differs from the plain read on this card - not used: "
                                 "%s\n", dev, name[v], why[v].c_str());
    static const char* const what[4] = {
        "", "the plain read (the norm per token, the down projection on 41 blocks)",
        "split (the norm per token and stream, then the plain read's down projection)",
        "staged (the norm per token and stream, the down projection's activations staged ahead by cp.async)"};
    std::fprintf(stderr, "strata hc: CUDA%d: the hyper-connection read runs as %s%s\n", dev, what[use],
                 use >= kHcSplit ? "; checked bit for bit against the plain read on this card (STRATA_HC_SPLIT=1 or 0 "
                                   "for the earlier ones)" : "");
}

}  // namespace strata::kernels
