// src/kernels/cuda/qsa_select.cu - see include/strata/kernels/qsa_select.hpp.
#include "strata/core/geometry.hpp"
#include <cstdlib>
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/qsa_two_pass.hpp"
#include "strata/kernels/qsa_two_pass_status.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
static_assert(IDX_DIM == strata::core::kCompiledGeometry.idx_key_dim && IDX_HEADS == strata::core::kCompiledGeometry.idx_q_heads && R == strata::core::kCompiledGeometry.idx_block,
              "qsa_select is compiled for core/geometry.hpp");
constexpr int SCORE_WARPS = 8;
constexpr int TOPK_T = 256;

__device__ __forceinline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_kernel(const float* __restrict__ pooled,
                                                                        const float* __restrict__ dead,
                                                                        const float* __restrict__ q_idx,
                                                                        const int32_t* __restrict__ steps,
                                                                        int64_t max_blocks, float* __restrict__ out) {
    const int64_t qi = blockIdx.y;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    const int64_t b = (int64_t) blockIdx.x * SCORE_WARPS + (threadIdx.x >> 5);
    if (b > n_bid || b >= max_blocks) return;
    const int lane = threadIdx.x & 31;
    const float* key = (b == n_bid) ? dead : pooled + b * IDX_DIM;
    const float4 k4 = *reinterpret_cast<const float4*>(key + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (b == n_bid && n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + b] = score;
    }
}

__global__ void __launch_bounds__(TOPK_T) block_topk_kernel(const float* __restrict__ scores,
                                                            const int32_t* __restrict__ steps, int64_t max_blocks,
                                                            int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int hist[256];
    __shared__ int s_a[TOPK_T], s_b[TOPK_T];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x;
    if (n_kv <= width) {                               // everything is selected: the identity, ascending
        for (int64_t j = t; j < n_kv; j += TOPK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;                      // blocks 0..n_bid, the last possibly empty
    const int64_t per = (nb + TOPK_T - 1) / TOPK_T;
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    // ---- radix select: the largest key thr with (cells with key >= thr) >= width, 8 bits at a time
    uint32_t prefix = 0;
    int above = 0;                                     // cells strictly above the digits fixed so far
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += TOPK_T) hist[i] = 0;
        __syncthreads();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = b0; b < b1; ++b) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[(k >> shift) & 255], w);
        }
        __syncthreads();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= width) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;          // cells equal to thr that fit, lowest index first
    // ---- per-thread counts of cells above and at the threshold, then their exclusive prefixes
    int gt = 0, eq = 0;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    __syncthreads();
    if (t == 0) {
        int ag = 0, ae = 0;
        for (int i = 0; i < TOPK_T; ++i) {
            const int g = s_a[i], e = s_b[i];
            s_a[i] = ag; s_b[i] = ae;
            ag += g; ae += e;
        }
    }
    __syncthreads();
    const int64_t eq_before = s_b[t];
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    __syncthreads();
    s_a[t] = sel;
    __syncthreads();
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < TOPK_T; ++i) { const int c = s_a[i]; s_a[i] = a; a += c; }
    }
    __syncthreads();
    int64_t wpos = s_a[t];
    int64_t eq_left = my_eq;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
}


// ---- QSA select on tensor cores (perf-review, after D-1): the block scores of many queries are one GEMM,
// rows (query, indexer head) x columns (blocks), K = 128, with relu per head summed. 3xTF32 (each operand split
// into a TF32 hi and lo part, hi*hi + hi*lo + lo*hi) keeps FP32-level accuracy; the summation order differs from
// the warp kernel, so a score can move in its last bits and a near-tie can select differently (not bitwise).
// Blocks < n_bid only; the tail block n_bid (the `dead` key, +1e9) is scored by the warp kernel's own code.
constexpr int TC_QT = 16;                 // queries per CTA (one m16 tile per indexer head)
constexpr int TC_NB = 32;                 // blocks per tile (4 warps x n8)
constexpr int TC_ITER = 4;                // tiles per CTA (the query tile is loaded once)
constexpr int TC_QS = IDX_HEADS * IDX_DIM + 4;   // query row stride in floats
constexpr int TC_KS = IDX_DIM + 4;               // key row stride

__device__ __forceinline__ uint32_t tf32_hi(float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ void mma_tf32(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void __launch_bounds__(128) block_scores_tc_kernel(const float* __restrict__ pooled,
                                                              const float* __restrict__ q_idx,
                                                              const int32_t* __restrict__ steps, int64_t nq,
                                                              int64_t max_blocks, int64_t reach,
                                                              float* __restrict__ out) {
    extern __shared__ __align__(16) float sm[];
    float* sQ = sm;                                   // [TC_QT][TC_QS]
    float* sK = sm + TC_QT * TC_QS;                   // [TC_NB][TC_KS]
    __shared__ int s_nbid[TC_QT];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, gid = lane >> 2, tig = lane & 3;
    const int64_t q0 = (int64_t) blockIdx.y * TC_QT;
    for (int i = t; i < TC_QT * IDX_HEADS * IDX_DIM / 4; i += 128) {
        const int r = i / (IDX_HEADS * IDX_DIM / 4), c = i % (IDX_HEADS * IDX_DIM / 4);
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (q0 + r < nq) v = reinterpret_cast<const float4*>(q_idx + (q0 + r) * IDX_HEADS * IDX_DIM)[c];
        *reinterpret_cast<float4*>(sQ + r * TC_QS + c * 4) = v;
    }
    if (t < TC_QT) s_nbid[t] = q0 + t < nq ? steps[(q0 + t) * kStepCount + kStepNBid] : 0;
    int lo_nbid = 0x7fffffff, hi_nbid = 0;
    __syncthreads();
    for (int i = 0; i < TC_QT; ++i) {
        if (q0 + i >= nq) break;
        lo_nbid = min(lo_nbid, s_nbid[i]);
        hi_nbid = max(hi_nbid, s_nbid[i]);
    }
    for (int it = 0; it < TC_ITER; ++it) {
        const int64_t b0 = ((int64_t) blockIdx.x * TC_ITER + it) * TC_NB;
        if (b0 >= reach || b0 >= hi_nbid) break;      // blocks >= every query's n_bid: nothing to score
        __syncthreads();                              // the previous tile's reads are done
        for (int i = t; i < TC_NB * IDX_DIM / 4; i += 128) {
            const int r = i / (IDX_DIM / 4), c = i % (IDX_DIM / 4);
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (b0 + r < hi_nbid) v = reinterpret_cast<const float4*>(pooled + (b0 + r) * IDX_DIM)[c];
            *reinterpret_cast<float4*>(sK + r * TC_KS + c * 4) = v;
        }
        __syncthreads();
        float acc[IDX_HEADS][4];
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h) acc[h][0] = acc[h][1] = acc[h][2] = acc[h][3] = 0.f;
        const float* kr = sK + (warp * 8 + gid) * TC_KS;
#pragma unroll 4
        for (int k0 = 0; k0 < IDX_DIM; k0 += 8) {
            const float kx0 = kr[k0 + tig], kx1 = kr[k0 + tig + 4];
            uint32_t bh[2], bl[2];
            bh[0] = tf32_hi(kx0);
            bh[1] = tf32_hi(kx1);
            bl[0] = tf32_hi(kx0 - __uint_as_float(bh[0]));
            bl[1] = tf32_hi(kx1 - __uint_as_float(bh[1]));
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float* qa = sQ + h * IDX_DIM + k0 + tig;
                const float x0 = qa[gid * TC_QS], x1 = qa[(gid + 8) * TC_QS];
                const float x2 = qa[gid * TC_QS + 4], x3 = qa[(gid + 8) * TC_QS + 4];
                uint32_t ah[4], al[4];
                ah[0] = tf32_hi(x0); ah[1] = tf32_hi(x1); ah[2] = tf32_hi(x2); ah[3] = tf32_hi(x3);
                al[0] = tf32_hi(x0 - __uint_as_float(ah[0]));
                al[1] = tf32_hi(x1 - __uint_as_float(ah[1]));
                al[2] = tf32_hi(x2 - __uint_as_float(ah[2]));
                al[3] = tf32_hi(x3 - __uint_as_float(ah[3]));
                mma_tf32(acc[h], al, bh);
                mma_tf32(acc[h], ah, bl);
                mma_tf32(acc[h], ah, bh);
            }
        }
        // relu per head, heads added in order (as the warp kernel), written where the block completed for the query
        const int64_t bc = b0 + warp * 8 + 2 * tig;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int qr = gid + half * 8;
            const int64_t qi = q0 + qr;
            if (qi >= nq) continue;
            const int nb = s_nbid[qr];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int64_t b = bc + j;
                if (b >= nb || b >= max_blocks) continue;
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) {
                    const float d = acc[h][half * 2 + j];
                    score += d > 0.0f ? d : 0.0f;
                }
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

// the tail block n_bid of each query: exactly block_scores_kernel's arithmetic for that block
__global__ void __launch_bounds__(32) block_scores_tail_kernel(const float* __restrict__ dead,
                                                               const float* __restrict__ q_idx,
                                                               const int32_t* __restrict__ steps,
                                                               int64_t max_blocks, float* __restrict__ out) {
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    if (n_bid >= max_blocks) return;
    const int lane = threadIdx.x & 31;
    const float4 k4 = *reinterpret_cast<const float4*>(dead + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + n_bid] = score;
    }
}

// ---- the same top-k with each query's keys read once: 1,024 threads hold up to TK_PER consecutive blocks' keys in
// registers (contexts up to 4 * 1024 * TK_PER cells), per-warp histograms, block-wide scans. The selection rule is
// block_topk_kernel's (radix threshold, ties to the lowest index, cells ascending): identical ids.
constexpr int TK_T = 1024;
constexpr int TK_PER = 33;
constexpr int TK_PER_MAX = TK_PER;

__device__ __forceinline__ int block_excl_scan(int v, int* s_warp, int& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = s_warp[lane];
        int z = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, z, o);
            if (lane >= o) z += y;
        }
        s_warp[lane] = z - w;               // exclusive per warp
        if (lane == 31) s_warp[32] = z;     // total
    }
    __syncthreads();
    const int r = s_warp[warp] + x - v;
    total = s_warp[32];
    __syncthreads();
    return r;
}

template <int PER>
__global__ void __launch_bounds__(TK_T) block_topk_reg_kernel(const float* __restrict__ scores,
                                                              const int32_t* __restrict__ steps, int64_t max_blocks,
                                                              int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int hist[TK_T / 32][256];
    __shared__ int s_warp[33];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {
        for (int64_t j = t; j < n_kv; j += TK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + TK_T - 1) / TK_T;       // <= PER (the caller checks)
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    uint32_t key[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) key[j] = (b0 + j < b1) ? order_key(sc[b0 + j]) : 0u;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
        __syncwarp();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t b = b0 + j;
            if (b >= b1) break;
            const int w = weight(b);
            if (w == 0) continue;
            if ((key[j] & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[warp][(key[j] >> shift) & 255], w);
        }
        __syncthreads();
        if (t < 256) {                                // fold the warps' histograms into warp 0's
            int s = 0;
            for (int w2 = 0; w2 < TK_T / 32; ++w2) s += hist[w2][t];
            hist[0][t] = s;
        }
        __syncthreads();
        // the digit: the first bin from the top where the count above reaches `width` (bin 0 when none of 255..1
        // does).  Threads 0-255 (warps 0-7) prefix-sum the 256 bins in descending order instead of one thread
        // walking them: the same digit and count above.
        int v = 0, x = 0;
        if (t < 256) {
            v = hist[0][255 - t];
            x = v;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_up_sync(0xffffffffu, x, o);
                if (lane >= o) x += y;
            }
            if (lane == 31) s_warp[warp] = x;
        }
        __syncthreads();
        if (t < 256) {
            int inc = x;
            for (int w2 = 0; w2 < warp; ++w2) inc += s_warp[w2];
            const int before = above + inc - v;
            if (before < width && (above + inc >= width || t == 255)) {
                s_digit = 255 - t;
                s_above = before;
            }
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;
    int gt = 0, eq = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) gt += w;
        else if (key[j] == thr) eq += w;
    }
    int tot;
    const int eq_before = block_excl_scan(eq, s_warp, tot);
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    int64_t wpos = block_excl_scan(sel, s_warp, tot);
    int64_t eq_left = my_eq;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (key[j] == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
}

// ---- a query with more blocks than the register kernel holds (contexts past 4 * 1024 * TK_PER cells): the register
// kernel's threads, per-warp histograms and warp scans, with each key read from memory again on every pass.  A
// histogram has no order, so on the four radix passes thread t reads blocks t, t + 1024, ...: a warp reads 32
// neighbours at a time.  The cells are emitted ascending in the order (warp, row, lane): warp w holds the `per` rows
// of 32 consecutive blocks from block w * 32 * per.  block_topk_kernel's selection rule (radix threshold, ties to the
// lowest index): identical ids.
__global__ void __launch_bounds__(TK_T) block_topk_wide_kernel(const float* __restrict__ scores,
                                                               const int32_t* __restrict__ steps, int64_t max_blocks,
                                                               int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int hist[TK_T / 32][256];
    __shared__ int s_gt[TK_T / 32], s_eq[TK_T / 32];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {
        for (int64_t j = t; j < n_kv; j += TK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + TK_T - 1) / TK_T;       // rows of 32 blocks per warp
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
        __syncwarp();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = t; b < nb; b += TK_T) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[warp][(k >> shift) & 255], w);
        }
        __syncthreads();
        if (t < 256) {                                // fold the warps' histograms into warp 0's
            int sum = 0;
            for (int w2 = 0; w2 < TK_T / 32; ++w2) sum += hist[w2][t];
            hist[0][t] = sum;
        }
        __syncthreads();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[0][d] >= width) break;
                cum += hist[0][d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int eq_budget = (int) (width - above);      // cells equal to thr that fit, lowest index first
    const int64_t w0 = (int64_t) warp * 32 * per;
    int gt = 0, eq = 0;
    for (int64_t b = w0 + lane; b < nb && b < w0 + 32 * per; b += 32) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        gt += __shfl_xor_sync(0xffffffffu, gt, o);
        eq += __shfl_xor_sync(0xffffffffu, eq, o);
    }
    if (lane == 0) { s_gt[warp] = gt; s_eq[warp] = eq; }
    __syncthreads();
    if (t == 0) {                                     // per warp: the selected cells and the tied cells before it
        int eb = 0, sb = 0;
        for (int w2 = 0; w2 < TK_T / 32; ++w2) {
            const int g2 = s_gt[w2], e2 = s_eq[w2];
            const int take = eq_budget - eb < 0 ? 0 : (eq_budget - eb > e2 ? e2 : eq_budget - eb);
            s_gt[w2] = sb;
            s_eq[w2] = eb;
            sb += g2 + take;
            eb += e2;
        }
    }
    __syncthreads();
    int run_sel = s_gt[warp], run_eq = s_eq[warp];
    for (int64_t r0 = w0; r0 < nb && r0 < w0 + 32 * per; r0 += 32) {
        const int64_t b = r0 + lane;
        const int w = b < nb ? weight(b) : 0;
        const uint32_t k = w ? order_key(sc[b]) : 0u;
        const int my_gt = (w && k > thr) ? w : 0, my_eq = (w && k == thr) ? w : 0;
        if (__ballot_sync(0xffffffffu, (my_gt | my_eq) != 0) == 0u) continue;   // a row without a selected block
        int pe = my_eq;                               // tied cells up to and with this lane
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, pe, o);
            if (lane >= o) pe += y;
        }
        const int left = eq_budget - (run_eq + pe - my_eq);
        const int take = left < 0 ? 0 : (left > my_eq ? my_eq : left);
        const int my_sel = my_gt + take;
        int ps = my_sel;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, ps, o);
            if (lane >= o) ps += y;
        }
        int32_t* dst = out + run_sel + ps - my_sel;
        for (int c = 0; c < my_sel; ++c) dst[c] = (int32_t) (b * R + c);
        run_eq += __shfl_sync(0xffffffffu, pe, 31);
        run_sel += __shfl_sync(0xffffffffu, ps, 31);
    }
}


// Block scores with every key block read ONCE for all of a call's queries (block_scores_kernel's grid is
// (max_blocks / 8) x nq: ~24,600 mostly-idle blocks per layer at a decode window, each key re-read per query).  A fixed
// grid strides over the blocks; per (block, query) the same arithmetic in the same order as block_scores_kernel.
// qsa_block_scores takes it for every call without an active-block count and at most MQ queries: the captured decode
// window, the uncaptured decode, and prefill's pooled16 call.
constexpr int MQ = 8;
__device__ __forceinline__ void block_scores_multi_body(const float* __restrict__ pooled,
                                                        const float* __restrict__ dead,
                                                        const float* __restrict__ q_idx,
                                                        const int32_t* __restrict__ steps, int nq, int64_t max_blocks,
                                                        float* __restrict__ out, bool early_exit,
                                                        const QsaScoreRows& srow) {
    __shared__ __align__(16) float qs[MQ * IDX_HEADS * IDX_DIM];
    __shared__ int64_t s_nkv[MQ], s_nbid[MQ];
    if (!early_exit)
        for (int i = threadIdx.x; i < nq * IDX_HEADS * IDX_DIM; i += blockDim.x) qs[i] = q_idx[i];
    if (threadIdx.x < nq) {
        s_nkv[threadIdx.x] = steps[threadIdx.x * kStepCount + kStepNKv];
        s_nbid[threadIdx.x] = steps[threadIdx.x * kStepCount + kStepNBid];
    }
    __syncthreads();
    int64_t top = 0;
    for (int q = 0; q < nq; ++q) top = s_nbid[q] > top ? s_nbid[q] : top;
    if (early_exit) {
        // (#783, stuchapin909) a CTA whose first block is past the batch's largest n_bid has no work: the grid is a
        // fixed 256 CTAs and a decode window reaches only a few of them, so those return before staging the queries.
        // Every thread of the CTA takes the same branch (top and blockIdx are CTA-uniform), so the barrier is safe.
        if ((int64_t) blockIdx.x * SCORE_WARPS > top) return;
        for (int i = threadIdx.x; i < nq * IDX_HEADS * IDX_DIM; i += blockDim.x) qs[i] = q_idx[i];
        __syncthreads();
    }
    const int lane = threadIdx.x & 31;
    const int64_t wstride = (int64_t) gridDim.x * SCORE_WARPS;
    for (int64_t b = (int64_t) blockIdx.x * SCORE_WARPS + (threadIdx.x >> 5); b <= top && b < max_blocks; b += wstride) {
        // srow.n > 0: each query reads its own sequence's keys; else one set of keys for every query
        const bool own = srow.n > 0;
        const float4 kp = own ? float4{} : *reinterpret_cast<const float4*>(pooled + b * IDX_DIM + lane * 4);
        const float4 kd = own ? float4{} : *reinterpret_cast<const float4*>(dead + lane * 4);
        for (int qi = 0; qi < nq; ++qi) {
            const int64_t n_bid = s_nbid[qi];
            if (b > n_bid) continue;
            const float4 k4 = !own ? ((b == n_bid) ? kd : kp)
                            : (b == n_bid) ? *reinterpret_cast<const float4*>(srow.dead[qi] + lane * 4)
                                           : *reinterpret_cast<const float4*>(srow.pooled[qi] + b * IDX_DIM + lane * 4);
            const float* q = qs + qi * IDX_HEADS * IDX_DIM + lane * 4;
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
                float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                score += d > 0.0f ? d : 0.0f;
            }
            if (lane == 0) {
                if (b == n_bid && s_nkv[qi] % R != 0) score += 1e9f;
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_multi_kernel(const float* __restrict__ pooled,
                                                                              const float* __restrict__ dead,
                                                                              const float* __restrict__ q_idx,
                                                                              const int32_t* __restrict__ steps, int nq,
                                                                              int64_t max_blocks, float* __restrict__ out,
                                                                              bool early_exit, QsaScoreRows srow = {}) {
    block_scores_multi_body(pooled, dead, q_idx, steps, nq, max_blocks, out, early_exit, srow);
}

// QSAROWS (docs/QSAROWS.md, STRATA_QSA_ROWS=1): block_scores_multi_kernel's work for NQ <= 4 queries with less issue
// pressure per (block, query).  nsys of a 131k decode window (RTX 2080 Ti + RTX 3060): the multi kernel costs ~20 us more a
// call for every row of the window on both cards - per (block, query) it reads the four query slices from shared memory
// (4 x LDS.128) and reduces each head's lane products with its own five shuffles (20 a block and query), and those two
// pipes bound it, not the key reads (the keys are read once for all the queries, as there).  Here (1) the lane's slices
// of the NQ queries (4 heads x 4 floats each) live in registers for the whole kernel, (2) the four heads' lane products
// are reduce-scattered with warp_sum's pairing (xor 16, 8, then 4, 2, 1: six shuffles for the four heads) and gathered
// to lane 0 in head order (three more).  Every sum adds the same two operands as block_scores_kernel's warp_sum at the
// same level (float add commutes), the dot expression is the same text in the same file, the relu-and-add runs over the
// heads in the same order: the same score bits.  The tail block and the +1e9 are as in the multi kernel.
template <int NQ>
__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_rows_kernel(const float* __restrict__ pooled,
                                                                             const float* __restrict__ dead,
                                                                             const float* __restrict__ q_idx,
                                                                             const int32_t* __restrict__ steps, int nq,
                                                                             int64_t max_blocks, float* __restrict__ out) {
    __shared__ int64_t s_nkv[NQ], s_nbid[NQ];
    if (threadIdx.x < nq) {
        s_nkv[threadIdx.x] = steps[threadIdx.x * kStepCount + kStepNKv];
        s_nbid[threadIdx.x] = steps[threadIdx.x * kStepCount + kStepNBid];
    }
    __syncthreads();
    int64_t top = 0;
    for (int q = 0; q < nq; ++q) top = s_nbid[q] > top ? s_nbid[q] : top;
    if ((int64_t) blockIdx.x * SCORE_WARPS > top) return;   // the multi kernel's early exit (CTA-uniform)
    const int lane = threadIdx.x & 31;
    float4 qr[NQ][IDX_HEADS];
#pragma unroll
    for (int qi = 0; qi < NQ; ++qi)
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h)
            qr[qi][h] = qi < nq ? *reinterpret_cast<const float4*>(q_idx + qi * IDX_HEADS * IDX_DIM + h * IDX_DIM + lane * 4)
                                : make_float4(0.f, 0.f, 0.f, 0.f);
    const float4 kd = *reinterpret_cast<const float4*>(dead + lane * 4);
    const bool hi16 = (lane & 16) != 0, hi8 = (lane & 8) != 0;
    const int64_t wstride = (int64_t) gridDim.x * SCORE_WARPS;
    for (int64_t b = (int64_t) blockIdx.x * SCORE_WARPS + (threadIdx.x >> 5); b <= top && b < max_blocks; b += wstride) {
        const float4 kp = *reinterpret_cast<const float4*>(pooled + b * IDX_DIM + lane * 4);
#pragma unroll
        for (int qi = 0; qi < NQ; ++qi) {
            if (qi >= nq) break;
            const int64_t n_bid = s_nbid[qi];
            if (b > n_bid) continue;
            const float4 k4 = (b == n_bid) ? kd : kp;
            float d[IDX_HEADS];
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float4 q4 = qr[qi][h];
                d[h] = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
            }
            // reduce-scatter: after xor 16 the lane keeps heads {0,1} (bit 4 clear) or {2,3}; after xor 8 one head
            float r2[2];
#pragma unroll
            for (int i = 0; i < 2; ++i)
                r2[i] = (hi16 ? d[i + 2] : d[i]) + __shfl_xor_sync(0xffffffffu, hi16 ? d[i] : d[i + 2], 16);
            float r = (hi8 ? r2[1] : r2[0]) + __shfl_xor_sync(0xffffffffu, hi8 ? r2[0] : r2[1], 8);
            r += __shfl_xor_sync(0xffffffffu, r, 4);
            r += __shfl_xor_sync(0xffffffffu, r, 2);
            r += __shfl_xor_sync(0xffffffffu, r, 1);
            // lane 8h holds head h's full sum: gather to lane 0 and add the relus in head order, as the warp kernel
            const float s1 = __shfl_sync(0xffffffffu, r, 8);
            const float s2 = __shfl_sync(0xffffffffu, r, 16);
            const float s3 = __shfl_sync(0xffffffffu, r, 24);
            if (lane == 0) {
                float score = 0.0f;
                score += r > 0.0f ? r : 0.0f;
                score += s1 > 0.0f ? s1 : 0.0f;
                score += s2 > 0.0f ? s2 : 0.0f;
                score += s3 > 0.0f ? s3 : 0.0f;
                if (b == n_bid && s_nkv[qi] % R != 0) score += 1e9f;
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

bool scores_multi_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_SCORES_MULTI"); return v == nullptr || std::atoi(v) != 0; }();
    return on;
}
// STRATA_QSA_EARLY_EXIT=0 keeps the old staging (every CTA loads the queries first); the scores are the same bits
bool scores_early_exit() {
    static const bool on = [] { const char* v = std::getenv("STRATA_QSA_EARLY_EXIT"); return v == nullptr || std::atoi(v) != 0; }();
    return on;
}
}  // namespace

void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535) {
        std::fprintf(stderr, "qsa_block_scores: unsupported indexer geometry\n");
        std::exit(1);
    }
    // a block past a query's n_bid returns at once: the grid need only reach the batch's largest n_bid (C-1)
    if (scores_multi_on() && nq <= MQ && active_blocks <= 0) {   // no active count: decode (captured or not) and prefill's pooled16
        // QSAROWS: STRATA_QSA_ROWS=1 (exactly) takes block_scores_rows_kernel for calls of up to 4 queries - the same bits
        static const bool rows_env = [] { const char* v = std::getenv("STRATA_QSA_ROWS"); return v != nullptr && v[0] == '1' && v[1] == '\0'; }();
        if (rows_env && qsa_block_scores_rows(pooled, dead, q_idx, steps, nq, max_blocks, s, scores, stream)) return;
        block_scores_multi_kernel<<<256, SCORE_WARPS * 32, 0, (cudaStream_t) stream>>>(pooled, dead, q_idx, steps, (int) nq,
                                                                                     max_blocks, scores, scores_early_exit());
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores multi: %s\n", cudaGetErrorString(e)); std::exit(1); }
        return;
    }
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const dim3 grid((unsigned) ((reach + SCORE_WARPS - 1) / SCORE_WARPS), (unsigned) nq);
    block_scores_kernel<<<grid, SCORE_WARPS * 32, 0, (cudaStream_t) stream>>>(pooled, dead, q_idx, steps, max_blocks,
                                                                              scores);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

bool qsa_block_scores_rows(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                           int64_t max_blocks, const QsaShapes& s, float* scores, void* stream) {
    if (nq <= 0) return true;
    if (nq > 4 || s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R) return false;
    cudaStream_t st = (cudaStream_t) stream;
    const int n = (int) nq;
    if (n == 1) block_scores_rows_kernel<1><<<256, SCORE_WARPS * 32, 0, st>>>(pooled, dead, q_idx, steps, n, max_blocks, scores);
    else if (n == 2) block_scores_rows_kernel<2><<<256, SCORE_WARPS * 32, 0, st>>>(pooled, dead, q_idx, steps, n, max_blocks, scores);
    else if (n == 3) block_scores_rows_kernel<3><<<256, SCORE_WARPS * 32, 0, st>>>(pooled, dead, q_idx, steps, n, max_blocks, scores);
    else block_scores_rows_kernel<4><<<256, SCORE_WARPS * 32, 0, st>>>(pooled, dead, q_idx, steps, n, max_blocks, scores);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores rows: %s\n", cudaGetErrorString(e)); std::exit(1); }
    return true;
}

bool qsa_block_scores_tc(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                         int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    if (nq <= 0) return true;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535 * TC_QT) return false;
    {   // sm_80 or newer (TF32 MMA); an older card keeps the warp kernel
        static int cc_major[64] = {};
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
        if (cc_major[dev] == 0) {
            int major = 0;
            if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            cc_major[dev] = major;
        }
        if (cc_major[dev] < 8) return false;
    }
    static bool attr[64] = {};   // the shared-memory opt-in is per device
    int adev = 0;
    cudaGetDevice(&adev);
    const int bytes = (TC_QT * TC_QS + TC_NB * TC_KS) * (int) sizeof(float);
    if (!attr[adev]) {
        if (cudaFuncSetAttribute(block_scores_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) !=
            cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        attr[adev] = true;
    }
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const int64_t per = (int64_t) TC_NB * TC_ITER;
    const dim3 grid((unsigned) ((reach + per - 1) / per), (unsigned) ((nq + TC_QT - 1) / TC_QT));
    block_scores_tc_kernel<<<grid, 128, bytes, (cudaStream_t) stream>>>(pooled, q_idx, steps, nq, max_blocks, reach,
                                                                         scores);
    block_scores_tail_kernel<<<(unsigned) nq, 32, 0, (cudaStream_t) stream>>>(dead, q_idx, steps, max_blocks, scores);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores_tc: %s\n", cudaGetErrorString(e)); std::exit(1); }
    return true;
}

void qsa_block_topk_ref(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                        const QsaShapes& s, int32_t* ids, void* stream) {
    if (nq <= 0) return;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    block_topk_kernel<<<(unsigned) nq, TOPK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

constexpr int CL_MAXQ = 16;    // larger calls (prefill sub-batches) fill the GPU with one CTA per query already

// What the sm_90+ cluster top-k runs on an sm_86 card and build: no launch, false, and on its first call per device
// the probe's cudaGetLastError(), which clears an error left pending on the calling thread.
bool qsa_block_topk_cluster(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                            const QsaShapes& s, int32_t* ids, void* stream) {
    (void) scores; (void) steps; (void) ids; (void) stream;
    if (nq <= 0) return true;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s) || nq > 65535 || max_blocks <= 0) return false;
    static int ok[64] = {};   // per device: 0 not probed yet, 2 probed (the cluster kernel does not run here)
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (ok[dev] == 0) {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaGetLastError();
        ok[dev] = 2;
    }
    return false;
}

// What the split top-k's device check runs here: no split, and a failed cudaGetDevice clears the pending error.
void topk_split_device() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) cudaGetLastError();
}

void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream, int64_t active_blocks) {
    // keys in registers when every query's blocks fit (contexts up to ~135K cells), else (CUDA) the same threads
    // reading them from memory; the same ids.
    if (nq <= 0) return;
    // STRATA_QSA_CLUSTER=0: no call to qsa_block_topk_cluster
    static const bool cluster = [] {
        const char* v = std::getenv("STRATA_QSA_CLUSTER");
        return !v || std::atoi(v) != 0;
    }();
    if (cluster && nq <= CL_MAXQ && qsa_block_topk_cluster(scores, steps, nq, max_blocks, cap, s, ids, stream))
        return;
    if (active_blocks <= 0 && max_blocks >= 16384 && s.idx_block == R && cap >= qsa_selection_width(kTopkMaxCells, s))
        topk_split_device();
    const int64_t reach = max_blocks;
    const int64_t fit = (int64_t) TK_T * TK_PER;
    constexpr int64_t kWideMinBlocks = 4608;   // RTX 3060: a prompt batch below ~18K cells is faster on the ref
    const bool too_small = reach > fit && active_blocks > 0 && active_blocks < kWideMinBlocks;
    if (too_small) {
        qsa_block_topk_ref(scores, steps, nq, max_blocks, cap, s, ids, stream);
        return;
    }
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    // Same ids as the ref kernel.
    if (reach > fit)   // past the register kernel's reach
        block_topk_wide_kernel<<<(unsigned) nq, TK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
    else
    if (reach <= (int64_t) TK_T * TK_PER)
        block_topk_reg_kernel<TK_PER><<<(unsigned) nq, TK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
    else
        block_topk_reg_kernel<TK_PER_MAX><<<(unsigned) nq, TK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

// ==== STRATA_SELECT_TWO_PASS (the prompt path's block scores in two precisions; qsa_two_pass.hpp has the bound) ====
// qsa_block_scores_tc computes every block score in 3xTF32.  Here every block is scored once in single-pass TF32
// (S1), a per-block bound eps encloses today's 3xTF32 score S3 in [S1 - eps, S1 + eps], and only the blocks whose
// interval reaches the query's top-k cut-off are scored again, with block_scores_tc_kernel's arithmetic, bit for bit.
// qsa_block_topk then reads a row whose every block lies on the same side of the threshold as in today's row and
// whose blocks at the threshold hold today's exact scores: the same threshold, the same ties, the same cells.  The row
// itself is not today's (kept blocks hold S1), so it is for qsa_block_topk only: nothing else reads m.sel_scores.
__device__ unsigned long long g_tp_stats[two_pass::kStatCount];

namespace {
namespace tp = strata::kernels::two_pass;

constexpr int TP1_ITER = 8;      // 32-block tiles per single-pass CTA (its query fragments are loaded once)
constexpr int TPB_T = 512;       // the band kernel's threads (one query per CTA)
static_assert(TPB_T == IDX_HEADS * IDX_DIM, "the band kernel reads one query element per thread");

__device__ __forceinline__ float tp_inf() { return __int_as_float(0x7f800000); }
__device__ __forceinline__ float tp_key_float(uint32_t k) {   // order_key's inverse (keys of non-NaN floats)
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

// The single-pass enclosure of block b's 3xTF32 score from its single-pass score s: [lo, hi], eps rounded up and the
// ends rounded outwards.  False: no bound (a non-finite score, norm or coefficient, or a bound past kMaxEps).
__device__ __forceinline__ bool tp_interval(float s, float2 kn, float P, float Q, float& lo, float& hi, float& e) {
    e = __fmaf_ru(P, kn.y, __fmaf_ru(Q, kn.x, tp::kAbs));
    if (!(e <= tp::kMaxEps) || !(fabsf(s) <= FLT_MAX)) return false;
    lo = __fsub_rd(s, e);
    hi = __fadd_ru(s, e);
    return true;
}

// The key norms (upper bounds): per pooled block |tf32(k)| and |tf32(k - tf32(k))|, the split block_scores_tc_kernel
// makes, in double with every operation rounded up; a non-finite norm becomes +inf (its blocks are never bounded).
__global__ void __launch_bounds__(256) tp_keys_kernel(const float* __restrict__ pooled, int64_t n_blocks,
                                                      float2* __restrict__ kn) {
    const int64_t b = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    if (b >= n_blocks) return;
    const int lane = threadIdx.x & 31;
    const float4 v = reinterpret_cast<const float4*>(pooled + b * IDX_DIM)[lane];
    const float x[4] = {v.x, v.y, v.z, v.w};
    double sh = 0.0, sl = 0.0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float h = __uint_as_float(tf32_hi(x[j]));
        const float l = __uint_as_float(tf32_hi(x[j] - h));
        sh = __dadd_ru(sh, __dmul_ru((double) h, (double) h));
        sl = __dadd_ru(sl, __dmul_ru((double) l, (double) l));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sh = __dadd_ru(sh, __shfl_xor_sync(0xffffffffu, sh, o));
        sl = __dadd_ru(sl, __shfl_xor_sync(0xffffffffu, sl, o));
    }
    if (lane == 0) {
        float kh = __double2float_ru(__dsqrt_ru(sh)), kl = __double2float_ru(__dsqrt_ru(sl));
        if (!(kh <= FLT_MAX)) kh = tp_inf();
        if (!(kl <= FLT_MAX)) kl = tp_inf();
        kn[b] = make_float2(kh, kl);
    }
}

// Pass 1: S1 = sum over the heads of relu(tf32(q) . tf32(k)), one MMA per step (block_scores_tc_kernel makes three).
// Warp h is head h and keeps its 16 steps of A fragments in registers; a CTA is one 16-query tile (blockIdx.x, so the
// CTAs that run together read the same keys) and TP1_ITER tiles of 32 blocks.  Blocks < n_bid only, as today; the
// tail block is block_scores_tail_kernel's.  Also clears the tile flags the band kernel sets.
__global__ void __launch_bounds__(128) tp_score1_kernel(const float* __restrict__ pooled,
                                                        const float* __restrict__ q_idx,
                                                        const int32_t* __restrict__ steps, int64_t nq,
                                                        int64_t max_blocks, int64_t reach, float* __restrict__ out,
                                                        int* __restrict__ tiles) {
    __shared__ __align__(16) uint32_t sK[TC_NB * TC_KS];
    __shared__ float red[IDX_HEADS][TC_QT][TC_NB + 1];
    __shared__ int s_nbid[TC_QT];
    const int t = threadIdx.x, lane = t & 31, h = t >> 5, gid = lane >> 2, tig = lane & 3;
    const int64_t q0 = (int64_t) blockIdx.x * TC_QT;
    if (blockIdx.y == 0 && t == 0) tiles[blockIdx.x] = 0;
    if (t < TC_QT) s_nbid[t] = q0 + t < nq ? steps[(q0 + t) * kStepCount + kStepNBid] : 0;
    uint32_t a[16][4];   // rows gid and gid + 8, k = 8s + tig and 8s + tig + 4 (the m16n8k8 A fragment)
    {
        const bool r0 = q0 + gid < nq, r1 = q0 + gid + 8 < nq;
        const float* qa = q_idx + (r0 ? q0 + gid : 0) * (IDX_HEADS * IDX_DIM) + h * IDX_DIM + tig;
        const float* qb = q_idx + (r1 ? q0 + gid + 8 : 0) * (IDX_HEADS * IDX_DIM) + h * IDX_DIM + tig;
#pragma unroll
        for (int s = 0; s < 16; ++s) {
            a[s][0] = tf32_hi(r0 ? qa[8 * s] : 0.f);
            a[s][1] = tf32_hi(r1 ? qb[8 * s] : 0.f);
            a[s][2] = tf32_hi(r0 ? qa[8 * s + 4] : 0.f);
            a[s][3] = tf32_hi(r1 ? qb[8 * s + 4] : 0.f);
        }
    }
    __syncthreads();
    int hi_nbid = 0;
    for (int i = 0; i < TC_QT; ++i) hi_nbid = max(hi_nbid, s_nbid[i]);
    for (int it = 0; it < TP1_ITER; ++it) {
        const int64_t b0 = ((int64_t) blockIdx.y * TP1_ITER + it) * TC_NB;
        if (b0 >= reach || b0 >= hi_nbid) break;
        __syncthreads();                              // the previous tile's sK and red reads are done
        for (int i = t; i < TC_NB * IDX_DIM / 4; i += 128) {
            const int r = i / (IDX_DIM / 4), c = i % (IDX_DIM / 4);
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (b0 + r < hi_nbid) v = reinterpret_cast<const float4*>(pooled + (b0 + r) * IDX_DIM)[c];
            *reinterpret_cast<uint4*>(sK + r * TC_KS + c * 4) =
                make_uint4(tf32_hi(v.x), tf32_hi(v.y), tf32_hi(v.z), tf32_hi(v.w));
        }
        __syncthreads();
        float acc[4][4];
#pragma unroll
        for (int n = 0; n < 4; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
#pragma unroll
        for (int s = 0; s < 16; ++s) {
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                const uint32_t* kr = sK + (n * 8 + gid) * TC_KS + 8 * s + tig;
                const uint32_t b[2] = {kr[0], kr[4]};
                mma_tf32(acc[n], a[s], b);
            }
        }
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int c = n * 8 + 2 * tig;
            red[h][gid][c] = acc[n][0] > 0.0f ? acc[n][0] : 0.0f;
            red[h][gid][c + 1] = acc[n][1] > 0.0f ? acc[n][1] : 0.0f;
            red[h][gid + 8][c] = acc[n][2] > 0.0f ? acc[n][2] : 0.0f;
            red[h][gid + 8][c + 1] = acc[n][3] > 0.0f ? acc[n][3] : 0.0f;
        }
        __syncthreads();
        for (int e = t; e < TC_QT * TC_NB; e += 128) {
            const int r = e / TC_NB, c = e % TC_NB;
            const int64_t qi = q0 + r, b = b0 + c;
            if (qi >= nq || b >= s_nbid[r] || b >= max_blocks) continue;
            float score = 0.0f;
#pragma unroll
            for (int h2 = 0; h2 < IDX_HEADS; ++h2) score += red[h2][r][c];
            out[qi * max_blocks + b] = score;
        }
    }
}

// block_topk_kernel's threshold over a list in shared memory: the largest key k such that the weighted entries with
// a key >= k reach `width`, 8 bits a pass with that kernel's digit rule (the largest digit whose cells at or above it
// reach the width, else 0).  unit: every entry weighs 1 (the sample); else a listed block weighs R and the tail block
// n_bid w_tail.  Every thread of the CTA calls it; it ends synchronized.
__device__ uint32_t tp_select(const uint32_t* __restrict__ key, const int32_t* __restrict__ blk, int n, int width,
                              bool unit, int64_t n_bid, int w_tail, int* hist, int* s_misc) {
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += TPB_T) hist[i] = 0;
        __syncthreads();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int i0 = 0; i0 < n; i0 += TPB_T) {       // the same trip count in every thread: whole warps match
            const int i = i0 + t;
            const uint32_t k = i < n ? key[i] : 0u;
            const bool hit = i < n && (k & hi_mask) == (prefix & hi_mask);
            const bool tail = hit && !unit && blk[i] == n_bid;
            const int d = (int) ((k >> shift) & 255u);
            const int g = hit && !tail ? d : 256;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
            // no __match_any_sync on 6.x: one add per thread (integer atomics: the same sums, only more of them)
            if (g < 256) atomicAdd(&hist[d], unit ? 1 : R);
#else
            const unsigned grp = __match_any_sync(0xffffffffu, g);   // one add per digit and warp
            if (g < 256 && lane == __ffs(grp) - 1) atomicAdd(&hist[d], (unit ? 1 : R) * __popc(grp));
#endif
            if (tail) atomicAdd(&hist[d], w_tail);
        }
        __syncthreads();
        if (warp == 0) {                              // the digit: lane l holds digits 8l .. 8l + 7
            int v[8], local = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) { v[j] = hist[lane * 8 + j]; local += v[j]; }
            int suf = local;                          // cells in this lane's digits and every higher one
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_down_sync(0xffffffffu, suf, o);
                if (lane + o < 32) suf += y;
            }
            int cum = above + suf - local, best = -1, best_cum = 0;
#pragma unroll
            for (int j = 7; j >= 0; --j) {
                const int dd = lane * 8 + j;
                if (best < 0 && dd >= 1 && cum + v[j] >= width) { best = dd; best_cum = cum; }
                cum += v[j];
            }
            const unsigned m = __ballot_sync(0xffffffffu, best >= 0);
            const int top = m ? 31 - __clz((int) m) : 0;
            const int dsel = __shfl_sync(0xffffffffu, best, top), csel = __shfl_sync(0xffffffffu, best_cum, top);
            const int all = __shfl_sync(0xffffffffu, suf, 0), h0 = __shfl_sync(0xffffffffu, v[0], 0);
            if (lane == 0) {
                s_misc[0] = m ? dsel : 0;
                s_misc[1] = m ? csel : above + all - h0;   // digit 0: every cell of digits 1..255 is above it
            }
        }
        __syncthreads();
        prefix |= (uint32_t) s_misc[0] << shift;
        above = s_misc[1];
        __syncthreads();
    }
    return prefix;
}

// Per query: the blocks to score again.  1. the coefficients P, Q (qsa_two_pass.hpp) from the query's head norms;
// 2. a candidate threshold F from every 32nd block (only when the blocks outnumber the list): the ks-th largest
// sampled S1 - eps; 3. one read of every block: the candidates (S1 + eps >= F), the cells with S1 - eps >= F (at least
// `width` of them, so the threshold T of today's row and the single-pass one T1 are both >= F and every block that
// matters below is a candidate) and E = the largest candidate eps; 4. T1 over the candidates; 5. T lies in
// [max(T1 - E, F), T1 + E]: a candidate whose interval reaches it (and is not the exact tail block) is listed for a
// re-score, by its column residue b % 8.  A query that cannot be settled (no bound for some candidate, a list or a
// residue list full, too few cells above F) is flagged: its 16-query tile is scored again in full.
__global__ void __launch_bounds__(TPB_T) tp_band_kernel(const float* __restrict__ scores,
                                                        const float* __restrict__ q_idx,
                                                        const int32_t* __restrict__ steps, int64_t max_blocks,
                                                        const float2* __restrict__ kn, float2* __restrict__ qc,
                                                        int* __restrict__ mode, int* __restrict__ cnt,
                                                        int* __restrict__ ids, int* __restrict__ tiles) {
    extern __shared__ __align__(16) uint32_t tp_sm[];
    uint32_t* c_key = tp_sm;                                           // [kCandCap] S1 keys (first: the sample)
    int32_t* c_blk = reinterpret_cast<int32_t*>(tp_sm + tp::kCandCap);  // [kCandCap] their blocks
    __shared__ int hist[256];
    __shared__ int s_misc[2], s_n, s_bad, s_wf[TPB_T / 32], s_cnt[8];
    __shared__ float s_e[TPB_T / 32], s_P, s_Q, s_E;
    __shared__ double s_hn[TPB_T / 32][2];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    if (qi == 0 && t == 0) atomicAdd(&g_tp_stats[tp::kStCalls], 1ull);
    if (n_kv <= width) {                              // every cell is selected: qsa_block_topk reads no score
        if (t == 0) { mode[qi] = 0; atomicAdd(&g_tp_stats[tp::kStIdentity], 1ull); }
        return;
    }
    {   // 1. a = sum_h |tf32(q_h)|, c = sum_h |tf32(q_h - tf32(q_h))|, rounded up; warps 4h .. 4h + 3 hold head h
        const float x = q_idx[qi * IDX_HEADS * IDX_DIM + t];
        const float hx = __uint_as_float(tf32_hi(x));
        const float lx = __uint_as_float(tf32_hi(x - hx));
        double sh = __dmul_ru((double) hx, (double) hx), sl = __dmul_ru((double) lx, (double) lx);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            sh = __dadd_ru(sh, __shfl_xor_sync(0xffffffffu, sh, o));
            sl = __dadd_ru(sl, __shfl_xor_sync(0xffffffffu, sl, o));
        }
        if (lane == 0) { s_hn[warp][0] = sh; s_hn[warp][1] = sl; }
        if (t == 0) { s_n = 0; s_bad = 0; }
        if (t < 8) s_cnt[t] = 0;
        __syncthreads();
        if (t == 0) {
            double a = 0.0, c = 0.0;
            for (int hh = 0; hh < IDX_HEADS; ++hh) {
                double eh = 0.0, el = 0.0;
                for (int w = 0; w < 4; ++w) { eh = __dadd_ru(eh, s_hn[4 * hh + w][0]); el = __dadd_ru(el, s_hn[4 * hh + w][1]); }
                a = __dadd_ru(a, __dsqrt_ru(eh));
                c = __dadd_ru(c, __dsqrt_ru(el));
            }
            double P = 0.0, Q = 0.0;
            tp::coefficients(a, c, P, Q);             // RN in double: far inside kSlack
            s_P = __double2float_ru(P);
            s_Q = __double2float_ru(Q);
        }
        __syncthreads();
    }
    const float P = s_P, Q = s_Q;
    const int64_t nb = n_bid + 1;
    const int w_tail = (int) (n_kv - n_bid * R);
    const float* sc = scores + qi * max_blocks;
    bool full = !(P <= tp::kMaxEps && Q <= tp::kMaxEps);
    float F = -tp_inf();
    if (!full && nb > tp::kCandCap) {                 // 2. the sample
        int64_t stride = tp::kSampleStride;
        while ((nb + stride - 1) / stride > tp::kCandCap) stride *= 2;
        const int ns = (int) ((nb + stride - 1) / stride);
        for (int i = t; i < ns; i += TPB_T) {
            const int64_t b = (int64_t) i * stride;
            uint32_t k = 0u;                          // no bound, or no cells: the lowest key
            if (b < n_bid) {
                float lo, hi, e;
                if (tp_interval(sc[b], kn[b], P, Q, lo, hi, e)) k = order_key(lo);
            } else if (w_tail > 0) {
                k = order_key(sc[b]);
            }
            c_key[i] = k;
        }
        __syncthreads();
        const int ks = (int) (2 * ((width + R * stride - 1) / (R * stride)) + 16);
        const uint32_t fk = tp_select(c_key, c_blk, ns, ks < ns ? ks : ns, true, n_bid, w_tail, hist, s_misc);
        if (fk != 0u) F = tp_key_float(fk);
    }
    int wf = 0;
    float emax = 0.0f;
    bool bad = false;
    if (!full) {                                      // 3. one read of every block
        for (int64_t b = t; b < nb; b += TPB_T) {
            float lo, hi, e = 0.0f, s;
            int w;
            if (b < n_bid) {
                w = R;
                s = sc[b];
                if (!tp_interval(s, kn[b], P, Q, lo, hi, e)) { bad = true; continue; }
            } else {
                w = w_tail;
                if (w == 0) continue;
                s = lo = hi = sc[b];                  // the tail kernel's score: today's
            }
            if (lo >= F) wf += w;
            if (hi >= F) {
                const int pos = atomicAdd(&s_n, 1);
                if (pos < tp::kCandCap) { c_key[pos] = order_key(s); c_blk[pos] = (int32_t) b; }
                emax = fmaxf(emax, e);
            }
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        wf += __shfl_xor_sync(0xffffffffu, wf, o);
        emax = fmaxf(emax, __shfl_xor_sync(0xffffffffu, emax, o));
    }
    bad = __any_sync(0xffffffffu, bad);
    if (lane == 0) { s_wf[warp] = wf; s_e[warp] = emax; if (bad) s_bad = 1; }
    __syncthreads();
    if (t == 0) {
        int W = 0;
        float E = 0.0f;
        for (int w = 0; w < TPB_T / 32; ++w) { W += s_wf[w]; E = fmaxf(E, s_e[w]); }
        s_misc[0] = W;
        s_E = E;
    }
    __syncthreads();
    const int n = s_n;
    full = full || s_bad != 0 || n > tp::kCandCap || s_misc[0] < width;
    if (!full) {
        // 4. T1, then 5. the window [lo_b, hi_b] that holds today's threshold
        const uint32_t t1k = tp_select(c_key, c_blk, n, (int) width, false, n_bid, w_tail, hist, s_misc);
        const float T1 = tp_key_float(t1k), E = s_E;
        const float lo_b = fmaxf(__fsub_rd(T1, E), F), hi_b = __fadd_ru(T1, E);
        for (int i = t; i < n; i += TPB_T) {
            const int64_t b = c_blk[i];
            if (b >= n_bid) continue;                 // the tail block holds today's score already
            float lo, hi, e;
            tp_interval(tp_key_float(c_key[i]), kn[b], P, Q, lo, hi, e);   // bounded in 3: the same result
            if (hi >= lo_b && lo <= hi_b) {
                const int r = (int) (b & 7);
                const int p = atomicAdd(&s_cnt[r], 1);
                if (p < tp::kResCap) ids[(qi * 8 + r) * tp::kResCap + p] = (int32_t) b;
            }
        }
        __syncthreads();
    }
    if (t == 0) {
        int tot = 0, mx = 0;
        for (int r = 0; r < 8; ++r) { tot += s_cnt[r]; mx = max(mx, s_cnt[r]); }
        if (mx > tp::kResCap) full = true;
        atomicAdd(&g_tp_stats[tp::kStQueries], 1ull);
        atomicAdd(&g_tp_stats[tp::kStBlocks], (unsigned long long) n_bid);
        if (full) {
            mode[qi] = 2;
            if (atomicExch(&tiles[qi / TC_QT], 1) == 0) atomicAdd(&g_tp_stats[tp::kStFullTiles], 1ull);
            atomicAdd(&g_tp_stats[tp::kStFull], 1ull);
        } else {
            for (int r = 0; r < 8; ++r) cnt[qi * 8 + r] = s_cnt[r];
            qc[qi] = make_float2(P, Q);
            mode[qi] = 1;
            atomicAdd(&g_tp_stats[tp::kStBand], (unsigned long long) tot);
            atomicMax(&g_tp_stats[tp::kStBandMax], (unsigned long long) tot);
            atomicAdd(&g_tp_stats[tp::kStCand], (unsigned long long) n);
            atomicMax(&g_tp_stats[tp::kStCandMax], (unsigned long long) n);
        }
    }
}

// The listed blocks scored again exactly as block_scores_tc_kernel scores them: the query in its row of the m16
// tile (qi % 16), each block in its column of the n8 tile (b % 8), the same TF32 splits, the same three MMAs per step
// in the same order into the same accumulator, the same relu sum.  The other rows and columns hold zeros where
// today's kernel holds other queries and blocks; an MMA output element depends only on its own row of A, column of B
// and accumulator, so the element is today's bit for bit (qsa_two_pass_parity and STRATA_SELECT_TWO_PASS_CHECK
// compare every one with today's kernel).  Warp w takes tiles w, w + 4, ...: tile j holds the j-th listed block of
// every residue.
__global__ void __launch_bounds__(128) tp_rescore_kernel(const float* __restrict__ pooled,
                                                         const float* __restrict__ q_idx, int64_t max_blocks,
                                                         const int* __restrict__ mode, const int* __restrict__ cnt,
                                                         const int* __restrict__ ids, const int* __restrict__ tiles,
                                                         float* __restrict__ out) {
    __shared__ __align__(16) float sq[IDX_HEADS * IDX_DIM];
    __shared__ int s_cnt[8];
    const int64_t qi = blockIdx.x;
    if (mode[qi] != 1 || tiles[qi / TC_QT] != 0) return;   // nothing listed, or the full kernel takes the tile
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, gid = lane >> 2, tig = lane & 3;
    for (int i = t; i < IDX_HEADS * IDX_DIM; i += 128) sq[i] = q_idx[qi * IDX_HEADS * IDX_DIM + i];
    if (t < 8) s_cnt[t] = cnt[qi * 8 + t];
    __syncthreads();
    int ntile = 0;
    for (int r = 0; r < 8; ++r) ntile = max(ntile, s_cnt[r]);
    const int row = (int) (qi % TC_QT);
    const bool mine0 = gid == (row & 7) && row < 8, mine1 = gid == (row & 7) && row >= 8;
    const int* lst = ids + qi * 8 * tp::kResCap;
    for (int j = warp; j < ntile; j += 4) {
        const int bcol = j < s_cnt[gid] ? lst[gid * tp::kResCap + j] : -1;   // this lane's column: gid
        const float* kr = pooled + (int64_t) (bcol >= 0 ? bcol : 0) * IDX_DIM;
        float acc[IDX_HEADS][4];
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h) acc[h][0] = acc[h][1] = acc[h][2] = acc[h][3] = 0.f;
#pragma unroll 4
        for (int k0 = 0; k0 < IDX_DIM; k0 += 8) {
            const float kx0 = bcol >= 0 ? kr[k0 + tig] : 0.f, kx1 = bcol >= 0 ? kr[k0 + tig + 4] : 0.f;
            uint32_t bh[2], bl[2];
            bh[0] = tf32_hi(kx0);
            bh[1] = tf32_hi(kx1);
            bl[0] = tf32_hi(kx0 - __uint_as_float(bh[0]));
            bl[1] = tf32_hi(kx1 - __uint_as_float(bh[1]));
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float* qa = sq + h * IDX_DIM + k0 + tig;
                const float x0 = mine0 ? qa[0] : 0.f, x1 = mine1 ? qa[0] : 0.f;
                const float x2 = mine0 ? qa[4] : 0.f, x3 = mine1 ? qa[4] : 0.f;
                uint32_t ah[4], al[4];
                ah[0] = tf32_hi(x0); ah[1] = tf32_hi(x1); ah[2] = tf32_hi(x2); ah[3] = tf32_hi(x3);
                al[0] = tf32_hi(x0 - __uint_as_float(ah[0]));
                al[1] = tf32_hi(x1 - __uint_as_float(ah[1]));
                al[2] = tf32_hi(x2 - __uint_as_float(ah[2]));
                al[3] = tf32_hi(x3 - __uint_as_float(ah[3]));
                mma_tf32(acc[h], al, bh);
                mma_tf32(acc[h], ah, bl);
                mma_tf32(acc[h], ah, bh);
            }
        }
        if (gid == (row & 7)) {                       // row `row`, columns 2 tig and 2 tig + 1
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const int c = 2 * tig + jj;
                if (j >= s_cnt[c]) continue;
                const int64_t b = lst[c * tp::kResCap + j];
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) {
                    const float d = row >= 8 ? (jj ? acc[h][3] : acc[h][2]) : (jj ? acc[h][1] : acc[h][0]);
                    score += d > 0.0f ? d : 0.0f;
                }
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

// A flagged 16-query tile scored again in full: block_scores_tc_kernel's body as it is above, the CTA taking the tile
// (blockIdx.y) only when the band kernel flagged it and striding over its 128-block ranges.  Each 32-block tile is
// computed exactly as today's kernel computes it (the same shared-memory tiles, warps, MMAs and sums), so every
// score of the tile is today's.
__global__ void __launch_bounds__(128) tp_full_kernel(const float* __restrict__ pooled,
                                                      const float* __restrict__ q_idx,
                                                      const int32_t* __restrict__ steps, int64_t nq,
                                                      int64_t max_blocks, int64_t reach,
                                                      const int* __restrict__ tiles, float* __restrict__ out) {
    if (tiles[blockIdx.y] == 0) return;
    extern __shared__ __align__(16) float sm[];
    float* sQ = sm;                                   // [TC_QT][TC_QS]
    float* sK = sm + TC_QT * TC_QS;                   // [TC_NB][TC_KS]
    __shared__ int s_nbid[TC_QT];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, gid = lane >> 2, tig = lane & 3;
    const int64_t q0 = (int64_t) blockIdx.y * TC_QT;
    for (int i = t; i < TC_QT * IDX_HEADS * IDX_DIM / 4; i += 128) {
        const int r = i / (IDX_HEADS * IDX_DIM / 4), c = i % (IDX_HEADS * IDX_DIM / 4);
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (q0 + r < nq) v = reinterpret_cast<const float4*>(q_idx + (q0 + r) * IDX_HEADS * IDX_DIM)[c];
        *reinterpret_cast<float4*>(sQ + r * TC_QS + c * 4) = v;
    }
    if (t < TC_QT) s_nbid[t] = q0 + t < nq ? steps[(q0 + t) * kStepCount + kStepNBid] : 0;
    int lo_nbid = 0x7fffffff, hi_nbid = 0;
    __syncthreads();
    for (int i = 0; i < TC_QT; ++i) {
        if (q0 + i >= nq) break;
        lo_nbid = min(lo_nbid, s_nbid[i]);
        hi_nbid = max(hi_nbid, s_nbid[i]);
    }
    (void) lo_nbid;
    for (int64_t x = blockIdx.x; x * TC_NB * TC_ITER < reach; x += gridDim.x) {
        for (int it = 0; it < TC_ITER; ++it) {
            const int64_t b0 = (x * TC_ITER + it) * TC_NB;
            if (b0 >= reach || b0 >= hi_nbid) break;
            __syncthreads();
            for (int i = t; i < TC_NB * IDX_DIM / 4; i += 128) {
                const int r = i / (IDX_DIM / 4), c = i % (IDX_DIM / 4);
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (b0 + r < hi_nbid) v = reinterpret_cast<const float4*>(pooled + (b0 + r) * IDX_DIM)[c];
                *reinterpret_cast<float4*>(sK + r * TC_KS + c * 4) = v;
            }
            __syncthreads();
            float acc[IDX_HEADS][4];
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) acc[h][0] = acc[h][1] = acc[h][2] = acc[h][3] = 0.f;
            const float* kr = sK + (warp * 8 + gid) * TC_KS;
#pragma unroll 4
            for (int k0 = 0; k0 < IDX_DIM; k0 += 8) {
                const float kx0 = kr[k0 + tig], kx1 = kr[k0 + tig + 4];
                uint32_t bh[2], bl[2];
                bh[0] = tf32_hi(kx0);
                bh[1] = tf32_hi(kx1);
                bl[0] = tf32_hi(kx0 - __uint_as_float(bh[0]));
                bl[1] = tf32_hi(kx1 - __uint_as_float(bh[1]));
#pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) {
                    const float* qa = sQ + h * IDX_DIM + k0 + tig;
                    const float x0 = qa[gid * TC_QS], x1 = qa[(gid + 8) * TC_QS];
                    const float x2 = qa[gid * TC_QS + 4], x3 = qa[(gid + 8) * TC_QS + 4];
                    uint32_t ah[4], al[4];
                    ah[0] = tf32_hi(x0); ah[1] = tf32_hi(x1); ah[2] = tf32_hi(x2); ah[3] = tf32_hi(x3);
                    al[0] = tf32_hi(x0 - __uint_as_float(ah[0]));
                    al[1] = tf32_hi(x1 - __uint_as_float(ah[1]));
                    al[2] = tf32_hi(x2 - __uint_as_float(ah[2]));
                    al[3] = tf32_hi(x3 - __uint_as_float(ah[3]));
                    mma_tf32(acc[h], al, bh);
                    mma_tf32(acc[h], ah, bl);
                    mma_tf32(acc[h], ah, bh);
                }
            }
            const int64_t bc = b0 + warp * 8 + 2 * tig;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int qr = gid + half * 8;
                const int64_t qi = q0 + qr;
                if (qi >= nq) continue;
                const int nbq = s_nbid[qr];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int64_t b = bc + j;
                    if (b >= nbq || b >= max_blocks) continue;
                    float score = 0.0f;
#pragma unroll
                    for (int h = 0; h < IDX_HEADS; ++h) {
                        const float d = acc[h][half * 2 + j];
                        score += d > 0.0f ? d : 0.0f;
                    }
                    out[qi * max_blocks + b] = score;
                }
            }
        }
    }
}

// STRATA_SELECT_TWO_PASS_CHECK: one query against today's row and ids.  Ids: every entry qsa_block_topk wrote.  Scored-
// again blocks (listed, the tail, or every block of a full query): bitwise today's score.  Kept blocks: |S1 - S3| <= eps.
__global__ void __launch_bounds__(256) tp_check_kernel(const float* __restrict__ scores, const int32_t* __restrict__ ids,
                                                       const float* __restrict__ ref, const int32_t* __restrict__ ref_ids,
                                                       const int32_t* __restrict__ steps, int64_t max_blocks,
                                                       int64_t cap, const float2* __restrict__ kn,
                                                       const float2* __restrict__ qc, const int* __restrict__ mode,
                                                       const int* __restrict__ cnt, const int* __restrict__ lists,
                                                       const int* __restrict__ tiles) {
    extern __shared__ uint32_t tp_mark[];
    __shared__ unsigned long long s_acc[6];
    __shared__ unsigned int s_ratio;
    const int t = threadIdx.x;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    if (t < 6) s_acc[t] = 0ull;
    if (t == 0) s_ratio = 0u;
    const int64_t m = n_kv < width ? n_kv : width;
    unsigned long long dif = 0;
    for (int64_t j = t; j < m; j += 256) dif += ids[qi * cap + j] != ref_ids[qi * cap + j];
    const bool sparse = n_kv > width;
    const int md = !sparse ? 0 : tiles[qi / TC_QT] != 0 ? 2 : mode[qi];   // a flagged tile: every block scored again
    const int64_t nb = n_bid + 1, words = (nb + 31) / 32;
    if (sparse) for (int64_t i = t; i < words; i += 256) tp_mark[i] = 0u;
    __syncthreads();
    atomicAdd(&s_acc[0], dif);
    if (sparse && md == 1) {
        for (int r = 0; r < 8; ++r) {
            const int c = cnt[qi * 8 + r];
            for (int j = t; j < c; j += 256) {
                const int b = lists[(qi * 8 + r) * tp::kResCap + j];
                atomicOr(&tp_mark[b >> 5], 1u << (b & 31));
            }
        }
    }
    __syncthreads();
    if (sparse) {
        const float* s1 = scores + qi * max_blocks;
        const float* s3 = ref + qi * max_blocks;
        const float2 pq = md == 1 ? qc[qi] : make_float2(0.f, 0.f);
        unsigned long long again = 0, again_dif = 0, kept = 0, kept_out = 0;
        float ratio = 0.0f;
        for (int64_t b = t; b < nb; b += 256) {
            const bool exact = md == 2 || b == n_bid || ((tp_mark[b >> 5] >> (b & 31)) & 1u);
            if (exact) {
                ++again;
                again_dif += __float_as_uint(s1[b]) != __float_as_uint(s3[b]);
            } else {
                ++kept;
                const float e = __fmaf_ru(pq.x, kn[b].y, __fmaf_ru(pq.y, kn[b].x, tp::kAbs));
                const double d = fabs((double) s1[b] - (double) s3[b]);
                kept_out += !(d <= (double) e);
                const float r = (float) (d / (double) e);
                ratio = fmaxf(ratio, r == r ? r : tp_inf());
            }
        }
        atomicAdd(&s_acc[1], again);
        atomicAdd(&s_acc[2], again_dif);
        atomicAdd(&s_acc[3], kept);
        atomicAdd(&s_acc[4], kept_out);
        atomicMax(&s_ratio, __float_as_uint(ratio));
    }
    __syncthreads();
    if (t == 0) {
        atomicAdd(&g_tp_stats[tp::kChkQueries], 1ull);
        if (s_acc[0]) atomicAdd(&g_tp_stats[tp::kChkIdQueries], 1ull);
        atomicAdd(&g_tp_stats[tp::kChkIdEntries], s_acc[0]);
        atomicAdd(&g_tp_stats[tp::kChkRescored], s_acc[1]);
        atomicAdd(&g_tp_stats[tp::kChkRescoredDiff], s_acc[2]);
        atomicAdd(&g_tp_stats[tp::kChkKept], s_acc[3]);
        atomicAdd(&g_tp_stats[tp::kChkKeptOut], s_acc[4]);
        atomicMax(&g_tp_stats[tp::kChkRatioMax], (unsigned long long) s_ratio);
    }
}

// The card runs qsa_block_scores_tc's tensor-core kernel (the reference of the two-pass scores), and the full kernel
// takes its shared memory.  The same dispatch rule as qsa_block_scores_tc: sm_80+, STRATA_QSA_WARP not set.
bool tp_device_ok() {
    static int ok[64] = {};   // 0 unknown, 1 yes, 2 no
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (ok[dev] == 0) {
        int major = 0;
        bool good = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess;
        const int bytes = (TC_QT * TC_QS + TC_NB * TC_KS) * (int) sizeof(float);
        good = good && major >= 8 &&
               cudaFuncSetAttribute(block_scores_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) ==
                   cudaSuccess &&
               cudaFuncSetAttribute(tp_full_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) == cudaSuccess;
        cudaGetLastError();
        ok[dev] = good ? 1 : 2;
    }
    return ok[dev] == 1;
}
}  // namespace

bool qsa_two_pass_on() {
    static two_pass::RequestStatus status(std::getenv("STRATA_SELECT_TWO_PASS"));
    if (!status.requested()) return false;
    // the card current at this call (a layer split can mix cards): sm_75 or STRATA_QSA_WARP=1|select keep the other
    // scorer there, so the switch says once that it is not used; checked here too, so a short prompt says it as well
    if (!tp_device_ok()) (void) status.on("this CUDA card");
    return true;
}
bool qsa_two_pass_check_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_SELECT_TWO_PASS_CHECK");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}
int64_t qsa_two_pass_min_blocks() {
    static const int64_t v = [] {
        const char* e = std::getenv("STRATA_SELECT_TWO_PASS_MIN");
        const long long x = e ? std::atoll(e) : (long long) two_pass::kMinBlocksDefault;
        return (int64_t) (x < 0 ? 0 : x);
    }();
    return v;
}
uint64_t qsa_two_pass_scratch_bytes(int64_t nq_cap, int64_t max_blocks) { return two_pass::layout(nq_cap, max_blocks).total; }

bool qsa_two_pass_keys(const float* pooled, int64_t n_blocks, int64_t nq_cap, int64_t max_blocks, void* scratch,
                       uint64_t scratch_bytes, void* stream) {
    if (!qsa_two_pass_on() || !tp_device_ok() || n_blocks > max_blocks || nq_cap <= 0 || nq_cap > 4096) return false;
    const two_pass::Layout L = two_pass::layout(nq_cap, max_blocks);
    if (scratch == nullptr || scratch_bytes < L.total || (reinterpret_cast<uintptr_t>(scratch) & 15u) != 0) return false;
    if (n_blocks <= 0) return true;
    float2* kn = reinterpret_cast<float2*>(static_cast<uint8_t*>(scratch) + L.kn);
    tp_keys_kernel<<<(unsigned) ((n_blocks + 7) / 8), 256, 0, (cudaStream_t) stream>>>(pooled, n_blocks, kn);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa two-pass keys: %s\n", cudaGetErrorString(e)); std::exit(1); }
    return true;
}

bool qsa_block_scores_two_pass(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps,
                               int64_t nq, int64_t nq_cap, int64_t max_blocks, const QsaShapes& s, float* scores,
                               void* scratch, uint64_t scratch_bytes, void* stream, int64_t active_blocks) {
    if (nq <= 0) return true;
    if (!qsa_two_pass_on() || s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R ||
        nq > nq_cap || nq_cap > 4096 || !tp_device_ok())
        return false;
    const two_pass::Layout L = two_pass::layout(nq_cap, max_blocks);
    if (scratch == nullptr || scratch_bytes < L.total || (reinterpret_cast<uintptr_t>(scratch) & 15u) != 0) return false;
    uint8_t* base = static_cast<uint8_t*>(scratch);
    const float2* kn = reinterpret_cast<const float2*>(base + L.kn);
    float2* qc = reinterpret_cast<float2*>(base + L.qc);
    int* mode = reinterpret_cast<int*>(base + L.mode);
    int* cnt = reinterpret_cast<int*>(base + L.cnt);
    int* ids = reinterpret_cast<int*>(base + L.ids);
    int* tiles = reinterpret_cast<int*>(base + L.tiles);
    const cudaStream_t st = (cudaStream_t) stream;
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    static std::atomic<bool> said{false};
    if (!said.exchange(true))
        std::fprintf(stderr, "qsa select: two-pass scores on (STRATA_SELECT_TWO_PASS=1): calls whose last query has %lld+ "
                             "blocks score every block in single-pass TF32 and again in 3xTF32 where the bound reaches the "
                             "cut-off (the same cells as the 3xTF32 scorer)\n", (long long) qsa_two_pass_min_blocks());
    const unsigned qtiles = (unsigned) ((nq + TC_QT - 1) / TC_QT);
    const int64_t per = (int64_t) TC_NB * TP1_ITER;
    tp_score1_kernel<<<dim3(qtiles, (unsigned) ((reach + per - 1) / per)), 128, 0, st>>>(pooled, q_idx, steps, nq,
                                                                                         max_blocks, reach, scores, tiles);
    block_scores_tail_kernel<<<(unsigned) nq, 32, 0, st>>>(dead, q_idx, steps, max_blocks, scores);
    tp_band_kernel<<<(unsigned) nq, TPB_T, (size_t) two_pass::kCandCap * 8, st>>>(scores, q_idx, steps, max_blocks, kn, qc,
                                                                                 mode, cnt, ids, tiles);
    tp_rescore_kernel<<<(unsigned) nq, 128, 0, st>>>(pooled, q_idx, max_blocks, mode, cnt, ids, tiles, scores);
    const int64_t ranges = (reach + (int64_t) TC_NB * TC_ITER - 1) / ((int64_t) TC_NB * TC_ITER);
    const int bytes = (TC_QT * TC_QS + TC_NB * TC_KS) * (int) sizeof(float);
    tp_full_kernel<<<dim3((unsigned) (ranges < 32 ? ranges : 32), qtiles), 128, bytes, st>>>(pooled, q_idx, steps, nq,
                                                                                             max_blocks, reach, tiles,
                                                                                             scores);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa two-pass scores: %s\n", cudaGetErrorString(e)); std::exit(1); }
    return true;
}

void qsa_two_pass_check(const float* scores, const int32_t* ids, const float* ref_scores, const int32_t* ref_ids,
                        const int32_t* steps, int64_t nq, int64_t nq_cap, int64_t max_blocks, int64_t cap,
                        void* scratch, void* stream) {
    if (nq <= 0 || scratch == nullptr) return;
    const size_t mark = (size_t) ((max_blocks + 31) / 32) * 4;
    if (mark > 48 * 1024) {
        static std::atomic<bool> said{false};
        if (!said.exchange(true)) std::fprintf(stderr, "qsa two-pass check: %lld blocks are too many to check\n", (long long) max_blocks);
        return;
    }
    const two_pass::Layout L = two_pass::layout(nq_cap, max_blocks);
    const uint8_t* base = static_cast<const uint8_t*>(scratch);
    tp_check_kernel<<<(unsigned) nq, 256, mark, (cudaStream_t) stream>>>(
        scores, ids, ref_scores, ref_ids, steps, max_blocks, cap, reinterpret_cast<const float2*>(base + L.kn),
        reinterpret_cast<const float2*>(base + L.qc), reinterpret_cast<const int*>(base + L.mode),
        reinterpret_cast<const int*>(base + L.cnt), reinterpret_cast<const int*>(base + L.ids),
        reinterpret_cast<const int*>(base + L.tiles));
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa two-pass check: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

bool qsa_two_pass_stats(unsigned long long* out, bool reset, void* stream) {
    const cudaStream_t st = (cudaStream_t) stream;
    void* p = nullptr;
    if (cudaGetSymbolAddress(&p, g_tp_stats) != cudaSuccess) { cudaGetLastError(); return false; }
    const size_t bytes = sizeof(unsigned long long) * two_pass::kStatCount;
    if (cudaMemcpyAsync(out, p, bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess ||
        (reset && cudaMemsetAsync(p, 0, bytes, st) != cudaSuccess) || cudaStreamSynchronize(st) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}

}  // namespace strata::kernels
