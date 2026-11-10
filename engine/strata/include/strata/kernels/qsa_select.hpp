// include/strata/kernels/qsa_select.hpp - plan v0.3 P5/P7: the QSA indexer's block scores and top-k selection,
// for many queries at once and at long context.
//
// The per-token path scores every pooled block with one FP64 block per row (qsa_index_kernel) and selects with a
// single-block kernel that makes 32 bit-serial passes over every CELL (topk_kernel): ~0.1-0.2 ms per query at 32K,
// once per QSA layer and token - 1-2 ms of a decode token and most of a 32K prompt's time.  Here:
//
//   qsa_block_scores : one warp per (query, block), FP32; relu per indexer head, summed; block n_bid (the
//                      incomplete tail) scores the `dead` key and gets +1e9 when it has cells - which is exactly
//                      what the per-token path reads there, because pooled[n_bid] holds `dead` at that time (in
//                      a chunk it may already hold a block completed later, so it is not read).
//   qsa_block_topk   : one block per query; a 4-pass radix select over the query's n_bid + 1 blocks, each
//                      weighted by its cell count, then the cells emitted in ascending order with ties to the
//                      lowest index - the same selection as topk_kernel, over a quarter of the elements.
//
// Queries carry their own step record (pos, n_kv, n_bid, width) as everywhere else in QSA.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

/// scores [nq, max_blocks]; q_idx [nq, idx_n_head, idx_dim] (normed and rotated); steps [nq, kStepCount].
/// Batch windows: query q's own indexer keys - pooled[q], dead[q] (q < n <= 8).
struct QsaScoreRows {
    const float* pooled[8] = {};
    const float* dead[8] = {};
    int n = 0;
};
void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream,
                      int64_t active_blocks = -1);   ///< perf-review C-1: > 0 launches only this many blocks (the
                                                     ///< batch's largest n_bid + 1; not for a captured graph)
/// QSAROWS (docs/QSAROWS.md): block_scores_multi_kernel's scores for a call of up to 4 queries, bit for bit, with the
/// queries in registers and the four heads reduce-scattered (six shuffles instead of twenty a block and query).
/// qsa_block_scores takes it with STRATA_QSA_ROWS=1; false (nothing launched) above 4 queries or on another geometry.
/// For qsa_rows_bench's parity and timing it is callable on its own.
bool qsa_block_scores_rows(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                           int64_t max_blocks, const QsaShapes& s, float* scores, void* stream);

/// The same scores on tensor cores (3xTF32, FP32-level accuracy but another summation order: not bitwise; the tail
/// block n_bid is the warp kernel's arithmetic). For the prompt path; false (nothing launched) on another geometry.
bool qsa_block_scores_tc(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                         int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks);

/// ids [nq, cap] (cells, ascending); `cap` >= the largest selection width.
void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream,
                    int64_t active_blocks = -1);   ///< > 0: no query of the call has more than this many blocks (n_bid + 1;
                                                   ///< the same contract as qsa_block_scores's). The register kernel is
                                                   ///< chosen by this, not by the capacity max_blocks (a long
                                                   ///< --max-context otherwise sends every short prompt to the slow one).
                                                   ///< Omit it for captured graphs whose context can grow after capture.
/// The probe left of the sm_90+ cluster top-k: launches nothing and returns false (true when nq <= 0); its first call
/// per device clears the thread's pending CUDA error. qsa_block_topk calls it for up to 16 queries unless
/// STRATA_QSA_CLUSTER=0.
bool qsa_block_topk_cluster(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                            const QsaShapes& s, int32_t* ids, void* stream);
/// The original kernel (256 threads, one shared histogram, keys read from memory on every radix pass), for tests:
/// the same ids.
void qsa_block_topk_ref(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                        const QsaShapes& s, int32_t* ids, void* stream);

/// STRATA_SELECT_TWO_PASS=1 (the prompt path, CUDA sm_80+; the bound and the rule: qsa_two_pass.hpp).  A scores row
/// that qsa_block_topk turns into exactly the ids of qsa_block_scores_tc's row: every block is scored once in
/// single-pass TF32, the blocks whose error bound reaches the query's top-k cut-off are scored again with
/// qsa_block_scores_tc's own 3xTF32 arithmetic (bit for bit), and a query whose bound cannot be settled has its
/// 16-query tile scored again in full.  The kept blocks hold their single-pass score, so the row is NOT
/// qsa_block_scores_tc's: it is for qsa_block_topk only.  `scratch` (qsa_two_pass_scratch_bytes(nq_cap, max_blocks),
/// 16-byte aligned) holds the key norms qsa_two_pass_keys wrote for this layer and chunk.  False (nothing launched):
/// the switch is off, HIP, a card or STRATA_QSA_WARP where qsa_block_scores_tc would not run its tensor-core kernel,
/// another geometry, nq > nq_cap, too little scratch.
bool qsa_two_pass_on();             ///< STRATA_SELECT_TWO_PASS=1, read once (HIP: false; prints once where unused)
bool qsa_two_pass_check_on();       ///< STRATA_SELECT_TWO_PASS_CHECK=1, read once (a diagnostic, see qsa_two_pass_check)
int64_t qsa_two_pass_min_blocks();  ///< STRATA_SELECT_TWO_PASS_MIN (blocks), read once; default two_pass::kMinBlocksDefault
uint64_t qsa_two_pass_scratch_bytes(int64_t nq_cap, int64_t max_blocks);
/// Once per layer and chunk, after the indexer appends and before the calls: the norms of the pooled keys [0, n_blocks).
bool qsa_two_pass_keys(const float* pooled, int64_t n_blocks, int64_t nq_cap, int64_t max_blocks, void* scratch,
                       uint64_t scratch_bytes, void* stream);
bool qsa_block_scores_two_pass(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps,
                               int64_t nq, int64_t nq_cap, int64_t max_blocks, const QsaShapes& s, float* scores,
                               void* scratch, uint64_t scratch_bytes, void* stream, int64_t active_blocks);
/// STRATA_SELECT_TWO_PASS_CHECK: right after a two-pass call and its qsa_block_topk, against today's row and ids for
/// the same queries (ref_scores from qsa_block_scores_tc, ref_ids from qsa_block_topk on them): counts the queries
/// whose ids differ, the scored-again blocks whose score is not today's bit for bit, and the kept blocks whose
/// single-pass score lies outside the bound (two_pass::Stat).
void qsa_two_pass_check(const float* scores, const int32_t* ids, const float* ref_scores, const int32_t* ref_ids,
                        const int32_t* steps, int64_t nq, int64_t nq_cap, int64_t max_blocks, int64_t cap,
                        void* scratch, void* stream);
/// The counters (two_pass::kStatCount of them, this device), read on `stream` (synchronized); reset: zeroed after.
bool qsa_two_pass_stats(unsigned long long* out, bool reset, void* stream);

}  // namespace strata::kernels
