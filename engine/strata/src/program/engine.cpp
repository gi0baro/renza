// src/program/engine.cpp - Strata's serve program as a library (strata/program/engine.hpp): Engine::open is its
// start-up and the serve set-up, a Session its requests (sync, one decode window, finish), single GPU, one session.
// Token ids in, token ids out: the tokenizer and the stdin/stdout protocol are the client's.

#include "strata/program/engine.hpp"
#include "strata/program/id_list.hpp"
#include "strata/platform/memory.hpp"
#include "strata/core/dma_batch.hpp"
#include "strata/core/device.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/core/coupled_draft.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/foresight_swap.hpp"
#include "strata/core/pinned.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/platform/aux_cpus.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/mtp.hpp"
#include "strata/prefill/prefill.hpp"
#if defined(STRATA_PREFILL_MMQ)
#include "strata/prefill/moe_mmq.hpp"
#endif
#include "strata/core/native_dense.hpp"
#include "strata/program/conv_cache.hpp"
#include "strata/program/split_rules.hpp"
#include "strata/prefill/share_rules.hpp"
#include "strata/program/split_ring.hpp"
#include "strata/program/draft_kv_plan.hpp"
#include "strata/program/message_boundary.hpp"
#include "strata/spec/draft_policy.hpp"
#include "strata/spec/draft_source.hpp"
#include "strata/spec/suffix_drafter.hpp"
#include "strata/core/progress.hpp"
#include "strata/core/device.hpp"
#include "strata/core/emulate.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#include <unistd.h>
#include <cerrno>
#include <sys/stat.h>
#include <sys/sysmacros.h>

#include <cuda_runtime.h>

#include <array>
#include <cstdarg>
#include <chrono>
#include <algorithm>
#include <memory>
#include <iostream>
#include <functional>
#include <future>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <new>
#include <charconv>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <exception>
#include <stdexcept>
#include <string>
#include <set>
#include <vector>

namespace {
// Windows' WDDM driver model: native Windows, or WSL2 (its GPU goes through /dev/dxg to the Windows driver).  There,
// pinning a large arena into two CUDA contexts leaves WDDM refusing every later allocation (the 5080 + 3090 rig);
// a Linux driver has no such limit (#253: the 8 GiB cap cost a 4090 + 3060 split 3x of its prompt speed).
bool under_wddm() {
    static const bool dxg = std::filesystem::exists("/dev/dxg");
    return dxg;
}

// #620 #486: " (N MiB of M MiB VRAM free on this GPU)" for an allocation's failure message (clears the error first)
std::string vram_free_note() {
    (void) cudaGetLastError();
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        (void) cudaGetLastError();
        return {};
    }
    char buf[96];
    std::snprintf(buf, sizeof buf, " (%llu MiB of %llu MiB VRAM free on this GPU)", (unsigned long long) (free_b >> 20),
                  (unsigned long long) (total_b >> 20));
    return buf;
}

// STRATA_CKPT_POOL=1 (read once; off by default): the serve loop keeps the running-state vectors of the conversation
// checkpoints it drops (evicted, pruned) and saves the next checkpoint into them, so a save resizes nothing (no
// zero-fill of ~118 MB of fresh vectors).  Every byte is overwritten by the copy, so the checkpoints are the same
bool ckpt_pool_env() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_CKPT_POOL");
        return v && v[0] != '0';
    }();
    return on;
}

cudaError_t adapt_copy_h2d(void* dst, const void* src, size_t bytes, cudaStream_t st) {
    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, st);
}
void unpin_blobs(std::vector<void*>& v) {
    for (void* p : v) (void) cudaHostUnregister(p);
    (void) cudaGetLastError();
    v.clear();
}

// STRATA_RSS_TRACE=1: the process's mapped file pages (the arena) at a startup step - who reads the arena back
void rss_probe(const char* where) {
    static const bool on = std::getenv("STRATA_RSS_TRACE") != nullptr;
    if (!on) return;
    FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return;
    char line[256];
    long long kb = -1;
    while (std::fgets(line, sizeof line, f))
        if (std::strncmp(line, "RssFile:", 8) == 0) { kb = std::atoll(line + 8); break; }
    std::fclose(f);
    std::fprintf(stderr, "strata rss: %s: RssFile %.2f GiB\n", where, (double) kb / (1024.0 * 1024.0));
    std::fflush(stderr);
}

// perf-review D-4: the lent slots are refilled with queued copies and one wait; STRATA_REFILL_BLOCKING=1 waits on each
bool refill_blocking() {
    static const bool v = std::getenv("STRATA_REFILL_BLOCKING") != nullptr;
    return v;
}

using Clock = std::chrono::steady_clock;

// The resident RAM mode and the adaptive tier.  A swap copies `in` (held in RAM) into the slot of `out` (held only
// by that slot).  Before the slot is overwritten, `out`'s bytes are copied back from it into an exchange buffer, so
// the CPU computes `out` from RAM while the swap is in flight; when the swap has landed, `commit_exchanges` moves
// them into `in`'s place in RAM.  The RAM copy then again holds exactly the experts no core slot does, and no swap
// reads the file.  Swaps that need no exchange (`out` in the lend region is held in RAM already; or `in` is not) go
// on as before; ones beyond the buffers' room wait for a later round.  Runs on the adaptive tier's thread while the
// GPU commits and drafts: the copies back are on its stream, and waited for before the refills are queued.
//
// `locate(layer)` names the cache the copy back reads and its stream and device (-1: the current one).
struct SwapHome {
    strata::core::ExpertCache* cache;
    cudaStream_t stream;
    int dev;
};
template <class Swap, class Locate>
bool resident_stage_swaps(strata::core::FileExpertSource& src, const std::vector<int32_t>& host_res,
                          int64_t n_expert, std::vector<Swap>& swaps, Locate locate) {
    if (!src.complement_ready() || swaps.empty()) return true;
    struct Staged { int32_t layer, in, out; int64_t q; };
    std::vector<Staged> staged;
    std::vector<SwapHome> used;
    std::vector<Swap> kept;
    kept.reserve(swaps.size());
    for (const Swap& s : swaps) {
        if (!src.has_resident(s.layer, s.in) || src.has_resident(s.layer, s.out)) { kept.push_back(s); continue; }
        const int64_t q = (int64_t) staged.size();
        if (q >= src.exchange_capacity()) continue;
        const int32_t slot = host_res[(size_t) s.layer * (size_t) n_expert + (size_t) s.out];
        if (slot < 0) continue;
        const SwapHome home = locate(s.layer);
        const strata::core::OnDevice on(home.dev);
        if (const cudaError_t e = cudaMemcpyAsync(src.exchange_buffer(q, s.layer), home.cache->device_slot(slot),
                            (size_t) strata::kernels::cpu::expert_layout().blob_bytes(s.layer), cudaMemcpyDeviceToHost,
                            home.stream); e != cudaSuccess) {
            std::fprintf(stderr, "strata: copy back of layer %d slot %d (device %d) failed: %s\n", (int) s.layer,
                         (int) slot, home.dev, cudaGetErrorString(e));
            return false;
        }
        bool seen = false;
        for (const SwapHome& u : used) seen = seen || u.stream == home.stream;
        if (!seen) used.push_back(home);
        staged.push_back({s.layer, s.in, s.out, q});
        kept.push_back(s);
    }
    if (!staged.empty()) {
        for (const SwapHome& u : used) {
            const strata::core::OnDevice on(u.dev);
            if (const cudaError_t e = cudaStreamSynchronize(u.stream); e != cudaSuccess) {
                std::fprintf(stderr, "strata: copy back sync (device %d) failed: %s\n", u.dev, cudaGetErrorString(e));
                return false;
            }
        }
        for (const Staged& x : staged)
            if (!src.stage_exchange(x.layer, x.in, x.out, x.q)) {
                std::fprintf(stderr, "strata: stage_exchange refused layer %d in %d out %d\n", (int) x.layer,
                             (int) x.in, (int) x.out);
                return false;
            }
    }
    swaps.swap(kept);
    return true;
}

struct Options {
    std::string pack = "pack/full";
    std::vector<int64_t> tokens;      // the prompt, PRE-TOKENIZED
    int64_t max_new = 16;
    int64_t max_context = 4096;
    std::string ple_gguf;              // the ORIGINAL second GGUF shard: the PLE table is not in the pack
    bool no_ple = false;              // explicit diagnostic ablation; never a normal inference default
    bool stream_token = false;        // R2.6 experiment: ordered work on the session stream
    bool gr_fp32_activations = false;  // pinned CUDA single-token BF16 activation contract
    bool gr_native_mmvf = false;       // pinned projection reduction tree as well as FP32 inputs
    bool native_bf16 = false;          // SSM gates, router and indexer projections only
    bool native_bf16_extra = false;    // PLE value and shared expert scalar gate
    bool native_ple_key = false;       // unchanged Q2_0 key and CUDA Q8_1 activations
    bool native_moe_combine = false;   // pinned fused CUDA weighted reduction
    bool native_gdn = false;          // pinned CUDA recurrence and preprocessing
    bool native_flash_attn_short = false; // diagnostic pinned attention, context <=256
    bool native_qsa_indexer = false;  // pinned F16 key cache and F32 pooling
    bool native_qsa = false;          // pinned F32 QSA norms and gate
    /// THE CONTEXT EXTENSION (rope scaling, rope_scaling.hpp).  These knobs resolve to ONE process config,
    /// set once before `session_init` builds the rope table and the graphs capture the kernels.  There is
    /// deliberately no per-request form: K sits in the cache POST-RoPE, so one cache must never mix two
    /// scalings.  Precedence: an EXPLICIT flag over the model file's rope keys over the struct defaults -
    /// `none` and `1` are explicit values (the opt-outs), the absent flag is not.
    std::string rope_scaling;           ///< --rope-scaling none|linear|yarn (llama.cpp's names); empty = the flag is absent
    double rope_scale = 0;              ///< --rope-scale F: the extension factor; 0 = the flag is absent (the model file's, else 1 = off)
    double rope_freq_base = 0;          ///< --rope-freq-base N: 0 = the model's (1e7)
    double rope_freq_scale = 0;         ///< --rope-freq-scale F: the raw ggml knob; 0 = 1/--rope-scale
    double yarn_orig_ctx = 0;           ///< --yarn-orig-ctx N: 0 = the model's, else 262144
    double yarn_ext_factor = -1.0;      ///< --yarn-ext-factor F: <0 = auto (1 for yarn, 0 otherwise)
    double yarn_attn_factor = 1.0;      ///< --yarn-attn-factor F
    double yarn_beta_fast = 32.0;       ///< --yarn-beta-fast F
    double yarn_beta_slow = 1.0;        ///< --yarn-beta-slow F
    bool native_rope = false;         // pinned text-only CUDA rotary arithmetic
    bool native_ple_postops = false;  // pinned PLE postprojection arithmetic
    bool native_router = false;       // pinned fused 512-expert top-10 router
    bool cpu_oracle_q8_0 = false;      // pinned x86 activation scales/codes at both expert stages
    std::string native_head_gguf;      // native output.weight experiment; same model shard as the pack
    std::vector<std::string> native_dense_gguf; // repeat for native GDN/QSA projection shards
    /// Every shard of --native's model (strata::gguf_split_paths: the metadata shard first; a missing shard is an
    /// error) and of --native-head-gguf's (the same list unless that names another model).
    std::vector<std::string> native_shards, native_head_shards;
    /// Plan v0.3 P1: the whole native arithmetic set as ONE switch (model shard 1). It enables exactly the
    /// combination recorded in bench/results/2026-09-23-attention-ple plus the native indexer, and never the
    /// <=256-token attention adapter. It becomes the default once P0 shows it is not slower.
    std::string native_preset;
    /// The token embedding from this GGUF instead of --native's (tools/embd_bf16_pack.py: BF16 as shipped)
    std::string embd_gguf;
    /// Plan v0.3 P2: how the n-gram table is read. Direct (default) = unbuffered SSD reads, table never in RAM.
    std::string ple_io = "direct";
    bool ple_io_explicit = false;      ///< --ple-io was given: the engine only warns when direct measures slow
    int64_t ple_row_cache = 1 << 20;   ///< bounded row cache (rows of 90 B); 0 disables
    int ple_inflight = 256;   // the prompt path reads a chunk's rows at once: 64 left the SSD half idle (32K: 303 -> 189 ms)
    double ple_delay_us = 0;           ///< fault injection: every row read completes no earlier than this
    bool ple_sync_submit = false;      ///< A/B arm: submit reads on the token thread, no I/O worker
    std::string kv = "fp16";           ///< plan v0.3 P7: KV storage, fp16 (default) or int8 (half the VRAM)
    int64_t kv_resident = 0;           ///< KV streaming: resident cells per QSA layer (0: all in VRAM)
    bool kv_grow = false;              ///< the elastic K/V (--kv-grow, STRATA_KV_GROW=1; kvg_ensure)
    bool no_capture = false;          // run the layers directly instead of replaying graphs
    bool no_pool = false;             // skip the CPU expert pool: the GPU-only floor
    int pool_workers = 0;         ///< R2.2: 0 = "all physical cores minus the host's"; >0 overrides
    int pool_tasks = 0;           ///< Batched CPU expert tasks per phase; 0 keeps the existing policy
    /// #272: the pool's core layout; `all` (the default) is the layout it always had, auto / p-cores are opt-in
    strata::kernels::cpu::PoolAffinity pool_affinity = strata::kernels::cpu::PoolAffinity::All;
    /// --pool-affinity measure (opt-in, hybrid CPUs, native packs): `all` and `p-cores` are timed on this pack's
    /// own experts at start and the faster layout is taken (measure_pool_affinity)
    bool pool_measure = false;
    /// --host-core first|last|sibling (STRATA_HOST_CORE): the host thread's core (see HostCore in pool.hpp)
    std::string host_core;
    /// --aux-cpus off|auto|LIST (STRATA_AUX_CPUS): where the non-pool, non-host threads run (strata/platform/aux_cpus.hpp)
    std::string aux_cpus;
    /// R2.2's first half, as an A/B arm.  **ON by default**, because the measurement that justifies it is the
    /// pool's own drain: 33.7 GB/s against 5/6 x 44.14 = 36.8 for five workers, on a machine whose sixth core
    /// is reserved for a host thread that has nothing to do while the drain runs.
    bool no_host_worker = false;
    bool coupled_draft = strata::core::coupled_draft_env(); ///< Coupled draft sampling for MTP drafter under sampling
    bool mmap_experts = false;    ///< R2.1: opt OUT of the resident arena, back to MapViewOfFile
    std::string shared_expert_arena; ///< Linux: optional file backing for the resident arena shared by processes
    bool resident_cpu_experts = false; ///< mmap-backed static-cache misses copied into ordinary RAM
    /// `--resident-experts` (the low-RAM PC's resident mode, chosen by setup): `--resident-cpu-experts` with the copy
    /// page-locked when the driver allows (else locked in the working set), 4 GiB of RAM headroom, and plain mmap
    /// (with a warning) when even the experts no slot holds do not fit.
    bool resident_pin = false;
    uint64_t resident_headroom = 8ull << 30;
    bool resident_soft = false;
    bool resident_cpu_explicit = false;   ///< #384: --resident-cpu-experts given by itself (not only implied)
    /// CS-T `--resident-budget-gib N`: the resident mode with a RAM budget - the N GiB of experts the GPU cache does
    /// not hold that the expert profile ranks hottest are copied into RAM, the rest are read from the files in place
    /// (the GGUF shards when the pack has no experts.bin).  0 = the whole complement (--resident-experts).
    uint64_t resident_budget = 0;
    /// R4: slots of VRAM-resident experts.  **0 = off, and off is the default.**
    /// **THE COMMENT THAT USED TO BE HERE WAS FALSE AND ROUND 328 MEASURED IT.**  It said "the cache has no
    /// consumer yet - `moe_hit_grouped_s2` does not exist - so switching it on costs the fill traffic and
    /// saves nothing".  The kernel exists (`src/kernels/cuda/s2_expert_grouped.cu`), it is wired at line ~660
    /// via `expert_hit_run`, and switching the cache on **does** move work off the CPU pool: the drain fell
    /// **19.076 -> 10.312 ms/token** at 4096 per-layer slots, for **-2.7 ms/token** end to end.  What was
    /// true is that the ADMISSION POLICY gave every slot to the first position, which is why the earlier
    /// measurement found nothing - see `expert_cache_per_layer`.
    int expert_cache = 0;
    bool expert_cache_cpu_order = false;
    /// **R4.2g.  ROUND 328 MEASURED THAT THE GLOBAL ADMISSION POLICY CANNOT WORK, AND THIS IS THE FIX.**
    /// The default policy hands out slots in arrival order from one counter shared by all 48 layers, so the
    /// first `n_slots` distinct pairs - about 26 LAYERS OF POSITION 0 - take every slot and hits are confined
    /// to them.  Measured at 256 slots: **1781 of 60000 = 2.97%**, against **21.4%** for 8 slots per layer and
    /// **70.4%** for 64, from `Memory/cache_allocation.py` on the same run's routing.  Off by default.
    bool expert_cache_per_layer = false;
    /// The PLE gather's prefetch, as an A/B arm.  The gather measured 2.10-2.61 ms/token because its sixteen
    /// row reads are sixteen SEPARATE page faults into a 26.8 GB mapping; see `ple_prefetch_enable`.
    bool no_ple_prefetch = false;
    /// R4.2e: a `profile.bin` from `tools/make_profile.py`.  **When given, it decides residency instead of the
    /// compulsory-miss policy**, which is the whole point: a profile ranked by routing frequency over a whole
    /// trace is what the plan's `h = 0.6447` refers to, and compulsory-miss measured 0.4864 because it fills
    /// with whatever the prompt touched FIRST.  Empty means no profile.
    std::string expert_profile;
    /// #477 (--serve, opt-in): where to save what the adaptive tier learned, as a profile `--expert-profile` reads
    /// (the resident experts first, then the routing counted since the start); on QUIT and every
    /// `expert_profile_save_min` minutes between requests.  Empty (the default): nothing is counted or written.
    std::string expert_profile_save;
    double expert_profile_save_min = 10.0;
    /// R4.2d: **ON by default**, because the measurement is unambiguous and the alternative is known-broken.
    /// Without it, 17 of 10,562 layers had the hit work done when the pool returned; with it, 9,190.  The
    /// A/B arm is `--no-hit-poke`.
    bool no_hit_poke = false;
    bool shared_late = false;          ///< plan v0.3 P3 A/B: shared expert inside post[l] (old order)
    bool keep_canonical = false;       ///< plan v0.3 P1 A/B: load canonical copies of natively served tensors
    bool no_token_graph = false;       ///< plan v0.3 P3 A/B: two graphs per layer instead of one per token
    bool no_fused_gr = false;          ///< plan v0.3 P3 A/B: the six-kernel native gr_read + separate gr_write
    bool no_fast_attn = false;         ///< plan v0.3 P3 A/B: gather + one-block-per-head QSA attention
    bool no_publish_kernel = false;    ///< plan v0.3 P3 A/B: memcpy nodes for the doorbell and QSA step
    bool no_fused_gdn = false;         ///< plan v0.3 P3 A/B: llama.cpp-layout GDN step + separate out norm
    bool no_fast_select = false;       ///< plan v0.3 P7 A/B: FP64 row scores + bit-serial cell top-k
    /// Plan v0.3 P4: `--expert-cache auto` sizes the VRAM tier from what is free after the weights, the session
    /// and the KV state, minus this reserve for the graphs, the hit scratch and the head.
    int vram_reserve_mib = 700;
    bool vram_reserve_given = false;   ///< --vram-reserve-mib on the command line (#496: no smaller automatic reserve)
    /// Plan v0.3 P5: batched prompt processing in chunks of this many tokens (0 = the token path).
    int64_t prefill_chunk = 0;
    /// `--prefill auto`: the largest chunk (up to 8192) whose buffers the expert cache can lend.  A chunk of
    /// stream_all_min tokens or more (1024 by default) streams every expert the GPU does not hold once, routed or not
    /// (a smaller one only the experts it routes to), so a bigger chunk streams fewer bytes per token (the "ubatch"
    /// effect).
    bool prefill_auto = false;
    /// #282, opt-in: the largest chunk `--prefill auto` may take - 8192 by default; `--prefill auto:16384` or
    /// `auto:32768` (or STRATA_PREFILL_AUTO_MAX) lets it go further, never past the context
    int64_t prefill_auto_max = 8192;
    bool no_split_rows = false;        ///< plan v0.3 P4 A/B: one whole expert per pool thread
    /// Plan v0.3 P5: the prompt path borrows the top expert-cache slots for its buffers and refills them after
    /// the prompt (default); `--no-prefill-borrow` reserves the buffers' VRAM for the whole session instead.
    bool no_prefill_borrow = false;
    /// Plan v0.3 P6: speculative decoding with a verify window of this many tokens (the last accepted token and
    /// spec-1 drafts); 0 = plain decode.
    int spec = 0;
    /// Plan v0.3 P6: the MTP draft layer's runtime directory (tools/mtp_rt.py); drafts come from it.
    std::string mtp;
    int64_t mtp_window = 32768;   ///< the draft layer attends to the last N cells (0 = every cell)
    /// Plan v0.3 P6: the share (0..1) of each layer's distinct missed experts the GPU reads over PCIe from the
    /// pinned arena while the CPU computes the rest (verify windows).
    double pcie_frac = -1.0;   ///< < 0: the model's default (0.2 direct for the Q2_0 pack, 0.55 DMA for native packs)
    bool pcie_frac_auto = false;     ///< single-GPU serve (CUDA + HIP/gfx1100), no --pcie-frac: the share is refined in the first decode windows (PcieAuto)
    double pcie_probe_gbps = 0.0;    ///< the start-up probe's reading (for PcieAuto)
    std::string pcie_mode = "auto";   ///< auto | dma | kernel | direct
    /// Plan v0.3 P6: every `adapt_every` rounds, swap up to `adapt_swaps` of the most-routed missing experts into
    /// the VRAM tier in place of the least-routed resident ones (decayed counts).  0 = static residency.
    int adapt_every = 4;
    float adapt_decay = 0.7f;   ///< the usage counts are multiplied by this after each adaptation (--adapt-decay)
    /// an expert is swapped into the VRAM tier only when its decayed routing count beats the one it replaces by more than
    /// this (--adapt-min-gain; both the blocking and the asynchronous tier)
    float adapt_min_gain = 1.5f;
    /// --adapt-async 1 (--serve with the resident RAM mode; opt-in): the adaptive tier's rounds advance between verify
    /// windows on a helper thread instead of one window waiting for a whole round (see the tier in the --serve block).
    /// Not bit-exact run to run.  0 = the blocking tier.
    int adapt_async = 0;
    /// Plan v0.3 P6: a draft enters the verify window only while every draft before it (and itself) has at least
    /// this probability under the draft layer; 0 = always --spec-1 drafts.
    double spec_min_p = 0.0;
    /// The end-of-turn tokens a request stops at (<|endoftext|> 248044, <|im_end|> 248046, or --eos-ids).
    std::vector<int64_t> eos_ids = {248044, 248046};
    bool spec_split = false;   ///< opt-in split verify window (the overlap study: exact, ~7% slower)
    /// Plan v0.3 P8: stay resident and take requests on stdin (see the --serve block in main).
    bool serve = false;
    /// The vision path: keep a per-cell (t, h, w) rotary position table so --serve can take GENI requests.
    bool vision = false;
    int adapt_swaps = 96;
    /// --serve: how many conversation checkpoints to keep between requests (0 = every request reads its whole
    /// prompt again, the v0.1.2 behaviour).  One is the GDN recurrence of the 36 layers, the QSA indexer tails and
    /// the PLE history (~118 MB of host RAM); the KV cache itself is positional and stays where it is.
    int prompt_cache = 6;
    /// --memory-limit-mib N / STRATA_MEMORY_LIMIT_MIB: the total RAM this engine's container or cgroup may use, for a
    /// container that cannot see its real limit (0 or unset = none; the flag wins over the variable)
    int64_t memory_limit_mib = -1;
    /// --serve: also keep a checkpoint every N freshly read prompt tokens (0 = only at the last turn boundary)
    int64_t prompt_cache_every = 16384;
    bool prompt_cache_tail = false;   // optional extra checkpoint at an existing near-tail chunk boundary
    bool prompt_cache_spread = false;   ///< --prompt-cache-spread: keep checkpoints spread back from the end, not the latest
    /// --serve: a prompt read from token 0 is also checkpointed at its first turn boundary - the end of the system
    /// prompt, which every chat of the same client shares - when that is at least N tokens (0 = never)
    int64_t prompt_cache_root = 2048;
    /// --serve: the token that opens a chat turn (<|im_start|>).  The last one in a prompt is where the chat's
    /// history ends and the new assistant turn begins, which is the checkpoint the next request can reuse.
    int64_t turn_token = 248045;
    /// --serve (#458, opt-in): the role token of a short turn the server puts right before the new assistant turn
    /// (the "system" of a trailing reasoning-effort turn).  When the turn before the last <|im_start|> opens with it,
    /// the checkpoint goes in front of that turn instead, so it holds the conversation and not the effort text.
    /// -1 = off (the checkpoint at the last <|im_start|>, as before).
    int64_t tail_role_token = -1;
    /// --serve: a text part of the prompt of at most N tokens (a chat message, the assistant header, a short tool
    /// result) goes through the verify windows, S tokens at a time, instead of the batched prompt path (0 = always
    /// the batched path)
    int64_t short_read = 64;
    bool prefill_head_alias = false;   // lend the native output head's VRAM to prompt buffers between request phases
    /// The suffix drafter (prompt lookup): when the text being written repeats an earlier stretch of the context (code
    /// edits, quoted input, tool-call JSON) by at least this many tokens, the window may be filled with what followed
    /// it there instead of the MTP's drafts, where the MTP's own first guess agrees and the draft policy expects it to
    /// pay (strata/spec/draft_policy.hpp).  On by default; 0 = MTP only.
    int suffix_draft = 3;
    /// The MTP's own window cap (0 = --spec): with --spec 6 --mtp-max-t 4 the long windows come from suffix matches.
    int mtp_max_t = 0;
    /// --lookup-chain K (opt-in, 0 = off): prompt lookup CHAINED after the MTP's proposal.  The suffix drafter matches
    /// the context followed by the MTP's drafts and appends up to K of the tokens that followed that match to the same
    /// verify window (the window grows to at most 8).  Greedy verification keeps the output; its own counts are kept.
    int lookup_chain = 0;
    int lookup_chain_min = 3;   ///< --lookup-chain-min M: the shortest match (in tokens, drafts included) it extends on
    /// --mtp-hnorm stream (opt-in): the draft layer's pre_fc_norm_hidden normalizes each hyper-connection stream on
    /// its own (llama.cpp's qwen4exp MTP graph) instead of one RMS over all four (the default, tools/mtp_probe.py).
    bool mtp_hnorm_stream = false;
    /// --mtp-q4 (opt-in): the draft layer's projections and draft head from 4-bit copies (fewer bytes per draft step)
    std::string mtp_q4;   ///< "" (off), all, proj (the projections only) or head (the draft head only)
    /// --mtp-draft-vocab FILE (opt-in): the draft head's token subset from FILE instead of <mtp>/draft_vocab.bin
    std::string mtp_draft_vocab;
};

void usage() {
    std::fprintf(stderr,
                 "strata generate --pack DIR --serve [options]\n"
                 "\n"
                 "  --pack DIR           the pack directory (default pack/full)\n"
                 "  --ple-gguf PATH      required PLE table (original second GGUF shard); with --native, the model's\n"
                 "                       shard that holds per_layer_token_embd.weight when not given\n"
                 "  --no-ple             explicit diagnostic ablation of the PLE layer\n"
                 "  --ple-io direct|mmap|ram  n-gram table reads (plan v0.3 P2). direct (default): unbuffered SSD\n"
                 "                       reads, the table never enters RAM or the file cache; mmap: A/B arm;\n"
                 "                       ram: the whole table locked in RAM at start (VirtualLock on Windows,\n"
                 "                       mlock elsewhere), so no SSD read sits on the prompt or token path\n"
                 "  --ple-row-cache N    bounded cache of fetched rows, 90 B each (default 1048576; 0 = off)\n"
                 "  --ple-inflight N     outstanding SSD reads (default 256)\n"
                 "  --ple-delay-us U     fault injection: each row read completes no earlier than U us\n"
                 "  --ple-sync-submit    A/B arm: submit table reads on the token thread (default: an I/O thread)\n"
                 "  --kv fp16|int8       KV storage (plan v0.3 P7): int8 codes + fp16 scale per 64 values, half the\n"
                 "                       VRAM; default fp16 until gate G-C accepts int8\n"
                 "  --kv q4_0            4-bit K/V after a Hadamard rotation (PR #21): half of int8's memory,\n"
                 "                       slightly lower precision (see bench/results/2026-09-27-kv-q4)\n"
                 "  --kv k8v4            hybrid: INT8 K (exact attention scores) + rotated Q4_0 V, 816 B/cell\n"
                 "                       (vs int8's 1,056); streams with --kv-resident too\n"
                 "  --kv-resident N      KV streaming: keep N cells of each QSA layer in VRAM (min 20480) and the\n"
                 "                       whole K/V in pinned RAM; the freed VRAM goes to expert slots. 0 (default):\n"
                 "                       all of it in VRAM. A context of N cells or fewer is not streamed\n"
                 "  --kv-grow            the K/V takes VRAM only for the cells the requests reach and the expert\n"
                 "                       cache holds the rest, giving slots back as the context grows (one GPU, with\n"
                 "                       --expert-profile; STRATA_KV_GROW=1/0 also). Default: the whole --max-context\n"
                 "                       allocated at start\n"
                 "  --stream-token       enqueue token work on the session stream (experimental)\n"
                 "  --gr-fp32-activations  experimental CUDA-oracle GR activation precision\n"
                 "  --gr-native-mmvf      experimental pinned GR norm/projections; implies FP32 activations\n"
                 "  --native-bf16         experimental CUDA-oracle SSM/router/indexer BF16 projections\n"
                 "  --native-bf16-extra   experimental CUDA-oracle PLE/shared gate BF16 projections\n"
                 "  --native-ple-key      experimental native PLE key; requires --native-dense-gguf\n"
                 "  --native-moe-combine  experimental pinned CUDA routed/shared combination\n"
                 "  --native-gdn          experimental pinned CUDA GDN norms/gates/recurrence\n"
                 "  --native-flash-attn-short  diagnostic pinned vector attention; --max-context <=256\n"
                 "  --native-qsa-indexer  experimental pinned indexer key cache and pooling\n"
                 "  --native-qsa          experimental pinned QSA normalization and output gate\n"
                 "  --native-rope         experimental pinned text-only CUDA rotary arithmetic\n"
                 "  --native-ple-postops  experimental pinned PLE postprojection arithmetic\n"
                 "  --native-router       experimental pinned CUDA 512-expert top-10 routing\n"
                 "  --cpu-oracle-q8-0     experimental pinned CPU expert quantization and dot reduction\n"
                 "  --native SHARD1      every full-context native path at once (plan v0.3 P1): stream-token,\n"
                 "                       GR MMVF, BF16, head, dense + PLE key, MoE combine, GDN, router, QSA,\n"
                 "                       indexer, RoPE, PLE postops, and the CPU q8_0 contract unless the\n"
                 "                       expert cache is on. Individual --native-* flags stay for A/B.\n"
                 "  --native-head-gguf PATH  native Q5_K head from model shard 1; requires --stream-token\n"
                 "  --embd-gguf PATH     the token embedding from this GGUF instead of --native's (tools/embd_bf16_pack.py:\n"
                 "                       BF16 as the checkpoint ships it; mapped host memory, no VRAM)\n"
                 "  --native-dense-gguf PATH native GDN/QSA/shared projections; repeat for each source model shard\n"
                 "  --expert-cache-cpu-order  experimental GPU expert reduction matching CPU order\n"
                 "  --max-context N      KV/state capacity (default 4096)\n"
                 "  --rope-scaling T     extend the context past the trained one: none, linear\n"
                 "                       (position interpolation) or yarn - llama.cpp's types and names.\n"
                 "                       Default: the model file's rope keys, else none. Fixed at startup:\n"
                 "                       K in the cache is post-RoPE, so one run one scaling\n"
                 "  --rope-scale F       the extension factor for linear/yarn (default: the model file's\n"
                 "                       factor, else 1 = off)\n"
                 "  --rope-freq-base N   the raw ggml knobs: the frequency base (0 = the model's 1e7) and\n"
                 "  --rope-freq-scale F  the angle shrink (0 = 1/--rope-scale)\n"
                 "  --yarn-orig-ctx N    the trained context the correction targets (0 = 262144)\n"
                 "  --yarn-ext-factor F --yarn-attn-factor F --yarn-beta-fast F --yarn-beta-slow F\n"
                 "                       YaRN's knobs; defaults: -1 (auto: 1 for yarn), 1, 32, 1\n"
                 "  --no-capture         run the layers directly instead of replaying graphs\n"
                 "  --shared-late        A/B: shared expert after the CPU pool (default: overlapped with it)\n"
                 "  --keep-canonical     A/B: also load canonical copies of natively served tensors (more VRAM)\n"
                 "  --vision             --serve takes images too (GENI requests; embeddings from strata-vision)\n"
                 "  --prompt-cache N     --serve: keep N conversation checkpoints between requests (default 6, ~118 MB\n"
                 "                       of RAM each; 0 = read every prompt from the start)\n"
                 "  --memory-limit-mib N  --serve: the total RAM (MiB) this engine's container or cgroup may use, when\n"
                 "                       the container cannot see its real limit (also STRATA_MEMORY_LIMIT_MIB; the flag\n"
                 "                       wins; 0 = none)\n"
                 "  --prompt-cache-every N  --serve: also checkpoint every N fresh prompt tokens (default 16384, 0 = off)\n"
                 "  --prompt-cache-tail  --serve, single GPU: one extra checkpoint near the prompt's end, at an existing\n"
                 "                       chunk boundary (default off; requires --prompt-cache and --prompt-cache-every > 0)\n"
                 "  --prompt-cache-spread  --serve: keep the checkpoints spread back from the conversation's end (dense\n"
                 "                       near it, thinner further back) instead of the most recently used: a client that\n"
                 "                       rewrites its history N tokens back reads about N again, not the whole conversation\n"
                 "  --turn-token ID      --serve: the token that opens a chat turn (default 248045, <|im_start|>)\n"
                 "  --tail-role-token ID --serve: a turn of this role right before the last turn is left out of\n"
                 "                       the conversation checkpoint (a trailing effort turn; default -1 = off)\n"
                 "  --short-read N       --serve: read at most N fresh text tokens through the decode windows instead\n"
                 "                       of the batched prompt path (default 64, 0 = off)\n"
                 "  --prefill-head-alias --serve: unload the native output head while each fresh prompt is read,\n"
                 "                       then release prompt buffers and reload it before decode\n"
                 "  --suffix-draft N     prompt lookup: draft from an earlier repeat of the last N+ tokens of context\n"
                 "                       when it pays (default 3; 0 = MTP only)\n"
                 "  --mtp-max-t M        cap the MTP's windows at M tokens (0 = --spec; longer ones come from suffixes)\n"
                 "  --spec N             the MTP drafter's verify window: how many tokens it proposes per check (setup\n"
                 "                       writes 4; --suffix-draft lets it grow by 2, up to 8, where a repeat is likely)\n"
                 "  --spec-min-p P       how sure the draft layer must be to extend a verify window by another guess\n"
                 "                       (setup writes 0.5; --calibrate measures it on this PC, see docs/DETAILS.md)\n"
                 "  --lookup-chain K     opt-in: after the MTP's drafts, add up to K prompt-lookup drafts that continue\n"
                 "                       them (the window grows to at most 8; default 0 = off)\n"
                 "  --lookup-chain-min M  the shortest context match --lookup-chain extends on (default 3)\n"
                 "  --mtp-hnorm pooled|stream  the draft layer's hidden-input norm: one RMS over all four streams\n"
                 "                       (default) or one per stream (llama.cpp's MTP graph)\n"
                 "  --mtp-q4 all|proj|head  opt-in: the draft layer's projections and/or draft head as 4-bit copies\n"
                 "  --mtp-draft-vocab FILE  opt-in: the draft head's token subset (default <mtp>/draft_vocab.bin)\n"
                 "  --no-token-graph     A/B: two graphs per layer (the host launches each) instead of one per token\n"
                 "  --no-fused-gr        A/B: the six-kernel hyper-connection read and a separate write (native)\n"
                 "  --prefill CHUNK      batched prompt processing in chunks of CHUNK tokens (needs --native); auto =\n"
                 "                       the largest chunk up to 8192 whose buffers the expert cache can lend;\n"
                 "                       auto:16384 / auto:32768 (or STRATA_PREFILL_AUTO_MAX) allow bigger ones\n"
                 "  --no-pool            skip the CPU expert pool (the GPU-only floor)\n"
                 "  --ple-gguf PATH      the n-gram/PLE shard.  WITHOUT IT LAYER 1's PLE IS SILENTLY SKIPPED,\n"
                 "                       which changes every number downstream - pass it for any real run\n"
                 "  --expert-profile P   R4.2e: pre-load the VRAM tier from a `profile.bin` (see\n"
                 "                       tools/make_profile.py) instead of admitting on first use.\n"
                 "  --expert-profile-save P  --serve, #477: save what the adaptive tier learned (the experts in\n"
                 "                       VRAM, then the routing counted since the start) as a profile at P, on\n"
                 "                       QUIT and every --expert-profile-save-every MIN minutes (default 10;\n"
                 "                       0 = on QUIT only) between requests; start from it with --expert-profile P\n"
                 "  --no-hit-poke        R4.2d's A/B arm.  The hit path pokes the driver once right after its\n"
                 "                       launch so the GPU starts while the CPU pool runs; without it the work\n"
                 "                       waits for the next driver entry and does not overlap at all.\n"
                 "  --no-ple-prefetch     A/B arm: read the PLE table's sixteen rows one at a time, instead of\n"
                 "                       issuing them in one PrefetchVirtualMemory call.\n"
                 "  --expert-cache N     R4: keep N expert blobs resident in VRAM and compute their rows on the\n"
                 "                       GPU via `moe_hit_grouped_s2`.  DEFAULT 0.  Measured at 4096 slots\n"
                 "                       with --expert-cache-per-layer: 54.4%% hits, CPU pool drain 19.1 -> 10.3\n"
                 "                       ms/token, -2.7 ms/token end to end.\n"
                 "  --expert-cache-per-layer  R4.2g: give each layer its OWN slots instead of letting the first\n"
                 "                       position take all of them.  The default policy fills in arrival order\n"
                 "                       from one shared counter, so 256 slots went to ~26 layers of position 0\n"
                 "                       and measured **2.97%%**.  Per-layer, the same routing gives 21.4%% at 8\n"
                 "                       slots/layer and 70.4%% at 64.  On a native pack, N is still the budget of N\n"
                 "                       largest blobs; each layer's slots are that layer's own blob.\n"
                 "  --no-host-worker     R2.2: the A/B arm.  By default the HOST THREAD joins the drain, so the\n"
                 "                       pool is six threads on six cores instead of five plus an idle core;\n"
                 "                       this flag restores the five-worker form for comparison on `pool phases`.\n"
                 "  --pool-workers N     R2.2: CPU expert pool worker count.  Default 0 = every physical core\n"
                 "                       except the one the host loop spins on (with --pool-affinity auto or\n"
                 "                       p-cores on a hybrid CPU: P-cores minus 1).  A sweep is how the pool's\n"
                 "                       deviation from `cpu_s2` is attributed.\n"
                 "  --pool-tasks N       Batched CPU expert tasks per GU/Down phase (0..4096). Default 0 =\n"
                 "                       3 per participating thread; positive counts are capped by row count.\n"
                 "  --host-core WHERE    The host thread's core: first (the default: the first physical core, as always) or\n"
                 "                       last (the last physical core, the workers on the others).  Windows sends a GPU's\n"
                 "                       interrupts to one logical processor, usually the first, and every copy that lands\n"
                 "                       raises one: a spinning host there waits for them.  Moves threads only, never a\n"
                 "                       result.  Not on hybrid CPUs.  sibling: the first core's other hardware thread\n"
                 "                       (SMT), so the interrupts keep the first logical processor and the workers keep\n"
                 "                       every core; hybrid CPUs too, first without SMT (STRATA_HOST_CORE sets it too).\n"
                 "  --aux-cpus SET       Where the threads that are neither pool workers nor the host thread run (the\n"
                 "                       adaptive tier's job thread, the prefill helpers, the CUDA driver's threads, ...):\n"
                 "                       off (the default: wherever they were created), auto (the SMT siblings of the host's\n"
                 "                       core, then physical cores with no worker; never a CPU that shares a core with a\n"
                 "                       worker; nothing is done when none is spare) or a CPU list such as 24,26 (Linux; the\n"
                 "                       host's and the workers' CPUs are dropped from it).  Moves threads only, never a\n"
                 "                       result.  STRATA_AUX_CPUS sets it too; a request's strata_tune aux_cpus=0|1 beats it.\n"
                 "  --pool-affinity MODE Worker CPU affinity: all (default: one worker per physical core, as\n"
                 "                       always), auto (hybrid CPUs: P-cores first, then their SMT siblings,\n"
                 "                       then E-cores) or p-cores (P-cores and their siblings only).  measure (opt-in,\n"
                 "                       hybrid CPUs, native packs): all and p-cores are timed on the pack's own experts\n"
                 "                       at start (about a second) and the faster one is taken\n"
                 "  --coupled-draft      enable coupled draft sampling for MTP drafter under sampling (STRATA_SPEC_COUPLED)\n"
                 "  --no-coupled-draft   disable coupled draft sampling (propose argmax drafts)\n"
                 "  --mmap-experts       R2.1: opt OUT of the resident expert arena, back to MapViewOfFile.\n"
                 "                       The A/B arm: the mmap's rate depends on the OS page cache holding\n"
                 "                       34 GB, and measured 71.97 vs 34.78 ms/token cold vs warm.\n"
                 "  --shared-expert-arena FILE  Linux: back the resident arena with one MAP_SHARED file.\n"
                 "                       Put this file on /dev/shm, not ordinary SSD storage.\n"
                 "                       A small header binds an existing backing file to the same pack.\n"
                 "  --resident-cpu-experts  with mmap and a static profile, keep the experts the GPU cache does not\n"
                 "                       hold resident in ordinary RAM (and the prompt path's lend region as far as\n"
                 "                       RAM allows); adaptive swaps exchange them, so none is read from the file again.\n"
                 "  --resident-experts   the low-RAM PC's resident mode (setup): --mmap-experts --resident-cpu-experts\n"
                 "                       with the copy page-locked when possible, 4 GiB headroom, plain mmap if it\n"
                 "                       does not fit.  Same answers as --mmap-experts for the same placement.\n"
                 "  --adapt-min-gain F   the adaptive tier swaps an expert in only when its decayed routing count beats\n"
                 "                       the one it replaces by more than F (default 1.5; higher = fewer, surer swaps)\n"
                 "  --adapt-async 1      --serve with the resident RAM mode or the arena (one GPU; opt-in): the adaptive\n"
                 "                       tier's swaps advance between verify windows instead of a window waiting for a\n"
                 "                       whole round (the arena: copy-in only).  Not bit-exact run to run.  0 (default)\n"
                 "                       = the blocking tier.\n");
}

// --lookup-chain: the context's last tokens for the draft sources (at most 64), the MTP's pending drafts last
int chain_tail(const strata::spec::SuffixDrafter& sfx, const int32_t* pending, int n_pending, std::vector<int32_t>& tail) {
    const std::vector<int32_t>& h = sfx.history();
    const int keep = std::max(0, std::min((int) h.size(), 64 - n_pending));
    tail.assign(h.end() - keep, h.end());
    tail.insert(tail.end(), pending, pending + n_pending);
    return (int) tail.size();
}

// --lookup-chain: the registered extra draft sources (strata/spec/draft_source.hpp) see the committed context too
void extra_sources_reset() {
    for (auto& src : strata::spec::extra_draft_sources()) src->reset();
}
void extra_sources_append(const int32_t* t, size_t n) {
    for (auto& src : strata::spec::extra_draft_sources()) src->append(t, n);
}

using strata::program::parse_i64_list;

/// The pool's adapter plus the wall-clock it spent, so the report can say how much of the token was the CPU.
struct Drive {
    strata::core::ExpertDispatch d;
    double cpu_ms = 0;
    int64_t calls = 0;
};

/// Plan v0.3 P6: the pool for a verify window.
void drive_pool_multi(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                      int64_t layer) {
    Drive* t = (Drive*) user;
    t->d.layers = layer;
    const Clock::time_point a = Clock::now();
    strata::core::expert_pool_dispatch_multi(t->d, x_f, ids, n_tok, k, out);
    t->cpu_ms += std::chrono::duration<double, std::milli>(Clock::now() - a).count();
    ++t->calls;
}

// #605: whether `path` is on a rotational disk (1), not (0) or unknown (-1), from sysfs; `dev` is the disk's name
int on_rotational_disk(const std::string& path, std::string& dev) {
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) return -1;
    char sys[64];
    std::snprintf(sys, sizeof sys, "/sys/dev/block/%u:%u", (unsigned) major(st.st_dev), (unsigned) minor(st.st_dev));
    std::error_code ec;
    std::filesystem::path p = std::filesystem::canonical(sys, ec);
    for (int up = 0; !ec && up < 2; ++up, p = p.parent_path()) {   // a partition has its disk's queue
        std::ifstream q(p / "queue" / "rotational");
        int r = -1;
        if (q >> r) {
            dev = p.filename().string();
            return r;
        }
    }
    return -1;
}

// #1341 #964: CUDA_LAUNCH_BLOCKING=1 in the environment.  A verify window's graph waits on the GPU for flags the host
// raises once cudaGraphLaunch has returned, and a blocking launch returns only when the graph has finished, so the
// first window never ends (the batched prompt path has no such wait and runs).  HIP's own switches are not checked.
bool launch_blocking() {
    const char* v = std::getenv("CUDA_LAUNCH_BLOCKING");
    return v != nullptr && std::atoi(v) != 0;
}

/// #2065: below this much VRAM free at a request's start the driver can page GPU memory and a window may never finish
constexpr int64_t kVramSafeMib =
    512;

/// STRATA_TRACE=1: the VRAM left at a step of the startup (finds what fills the card after the cache is sized)
void mem_mark(const char* where) {
    static const char* const env = std::getenv("STRATA_TRACE");
    if (env == nullptr) return;
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::fprintf(stderr, "strata trace: %lld MiB free after %s\n", (long long) (free_b >> 20), where);
    // STRATA_TRACE=2: also wait for the device here and name the first step after which it reports an error.  A
    // "sticky" failure (#2013: unspecified launch failure surfacing in a later, unrelated call) is then pinned to
    // the step that ran the faulting work.  Only a diagnostic: it adds a synchronisation per mark.
    if (env[0] == '2') {
        const cudaError_t sync_e = cudaDeviceSynchronize();
        const cudaError_t last_e = cudaGetLastError();
        std::fprintf(stderr, "strata trace: device %s after %s (sync: %s, last error: %s)\n",
                     sync_e == cudaSuccess && last_e == cudaSuccess ? "ok" : "ERROR", where, cudaGetErrorString(sync_e),
                     cudaGetErrorString(last_e));
    }
}

/// #463's A/B: STRATA_ADAPT_NOWAIT=1 lets a verify window start before the adaptive tier's copies have landed (0.1.37)
bool adapt_nowait() {
    static const bool v = [] { const char* e = std::getenv("STRATA_ADAPT_NOWAIT"); return e && e[0] == '1'; }();
    return v;
}

/// #463 without the stall: a verify window takes the adaptive tier's previous swaps only once they are this many windows
/// old, and then waits for their copies (which have landed by then).  The window that first computes a swapped-in expert on
/// the GPU depends only on the window count, as with #463's wait, so the answers are as reproducible.  Default 1 = #463 and
/// 0.1.39 (the very next window waits for the copies: ~4.6 ms per window on an RTX 2060 SUPER over PCIe 3.0 x8); lag 2 is
/// opt-in (STRATA_ADAPT_LAG=2).  (#764)
int adapt_lag() {
    static const int v = [] { const char* e = std::getenv("STRATA_ADAPT_LAG"); return e ? std::max(1, std::atoi(e)) : 1; }();
    return v;
}

/// STRATA_ADAPT_OVERLAP=1: the adaptive tier's copies land while the next decode window already runs.  The window
/// sees the swapped-in experts as resident from its first layer, exactly as after #463's wait (so the tokens are the same
/// ones); what changes is only when the bytes arrive: the copies go in layer order and the verify window holds a layer's
/// plan back until that layer's copies have landed (AdaptGate in strata/core/verify.hpp).  The ~10 ms a 96-swap round
/// took at the next window's start (PCIe 3.0 x16, Q2_0) is then hidden behind the window.  Opt-in on every backend
/// (=1 turns it on); the startup log names the state, and why when it cannot run.
bool adapt_overlap() {
    static const bool v = [] {
        const char* e = std::getenv("STRATA_ADAPT_OVERLAP");
        // Opt-in on every backend (#2010).  Windows CUDA since 0.1.43: an RTX 5070 decoded Q2_0 2.6% slower with it
        // (story decode, -2.1% code), IQ3_XXS no faster.  Linux CUDA was on by default from 0.1.43 (same tokens, +5.5..8.3%
        // decode on an A4000, +4.4% on a P100) until an RTX 5090 Laptop (WSL2, Core Ultra 9 275HX, native IQ3_XXS pack,
        // --expert-cache-per-layer) decoded 21% slower with it: 84.6 vs 106.6 tok/s medians of 5 generations of 256
        // tokens from one build, the slowest of the fast run above the fastest of the slow one.
        return e && e[0] == '1';
    }();
    return v;
}

/// ExpertDispatch::evict_sync where the tier does not wait for its copies; STRATA_ADAPT_EVICT_SYNC=0 turns it off.
bool adapt_evict_sync() {
    static const bool v = [] { const char* e = std::getenv("STRATA_ADAPT_EVICT_SYNC"); return e == nullptr || std::atoi(e) != 0; }();
    return v;
}

/// STRATA_ADAPT_CHECKRES=1: before a verify window, the device's residency table read back into `back` and
/// compared with the host's (the pool's plan); the mismatched entries, the first ones reported.
int64_t res_check(const int32_t* d_res, const std::vector<int32_t>& host_res, std::vector<int32_t>& back,
                  int64_t n_expert, int64_t window) {
    if (cudaMemcpy(back.data(), d_res, host_res.size() * sizeof(int32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "strata: res check: the read-back failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 0;
    }
    static int printed = 0;
    int64_t bad = 0;
    for (size_t i = 0; i < host_res.size(); ++i)
        if (back[i] != host_res[i]) {
            ++bad;
            if (printed < 20) {
                ++printed;
                std::fprintf(stderr, "strata: res check: window %lld layer %lld expert %lld: host %d, device %d\n",
                             (long long) window, (long long) (i / (size_t) n_expert), (long long) (i % (size_t) n_expert),
                             (int) host_res[i], (int) back[i]);
            }
        }
    return bad;
}

/// The pre-launch hook's patch: the changed entries of the host's residency table (`idx`) as (index, entry) pairs into
/// the mapped `pairs` (room for `cap`); their count, or -1 when they do not fit (then the whole table goes up).
int64_t res_patch_pairs(const std::vector<int32_t>& idx, const std::vector<int32_t>& host_res, int32_t* pairs, size_t cap) {
    if (pairs == nullptr || idx.size() > cap) return -1;
    for (size_t k = 0; k < idx.size(); ++k) {
        pairs[2 * k] = idx[k];
        pairs[2 * k + 1] = host_res[(size_t) idx[k]];
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return (int64_t) idx.size();
}

/// Mapped host memory for res_patch_pairs, or null.  The kernels read
/// it at the host's address: each of `devs` (-1: the current device) must map it there, else the whole table goes up.
int32_t* res_pairs_alloc(size_t cap, const std::vector<int>& devs) {
    void* p = nullptr;
    if (cudaHostAlloc(&p, cap * 2 * sizeof(int32_t), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    for (int dev : devs) {
        const strata::core::OnDevice on(dev);
        void* d = nullptr;
        if (cudaHostGetDevicePointer(&d, p, 0) != cudaSuccess || d != p) {
            (void) cudaGetLastError();
            cudaFreeHost(p);
            return nullptr;
        }
    }
    return (int32_t*) p;
}

/// --adapt-async: one background job at a time (the asynchronous adaptive tier's copies and memcpys).  `post` hands
/// it over, `idle` says whether it has finished; the owner never posts while a job runs.
class JobThread {
public:
    JobThread() : th_([this] { loop(); }) {}
    ~JobThread() {
        { std::lock_guard<std::mutex> lk(mu_); quit_ = true; }
        cv_.notify_one();
        th_.join();
    }
    JobThread(const JobThread&) = delete;
    JobThread& operator=(const JobThread&) = delete;
    void post(std::function<void()> f) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            job_ = std::move(f);
            has_ = true;
            done_.store(false, std::memory_order_release);
        }
        cv_.notify_one();
    }
    bool idle() const { return done_.load(std::memory_order_acquire); }
    void wait() const { while (!idle()) std::this_thread::yield(); }
private:
    // The second logical processor of the last physical core that has two: an SMT sibling no pool worker or host loop
    // is pinned to (the pool takes the first of every core but the first, the host the first core), or -1.
    static int smt_spare_cpu() {
        return -1;
    }
    void loop() {
        // the job's memcpys stay off the host's core and the pool's: by default the spare SMT sibling of the last
        // two-thread core (Windows); STRATA_ADAPT_JOB_CPU=<logical cpu>, or -1 for no pinning.  Without either, --aux-cpus
        // (off by default) puts the thread on the spare CPUs; an explicit STRATA_ADAPT_JOB_CPU is the user's word over it.
        const char* c = std::getenv("STRATA_ADAPT_JOB_CPU");
        const int cpu = c != nullptr ? std::atoi(c) : smt_spare_cpu();
        if (cpu >= 0) (void) strata::kernels::cpu::pin_current_thread(cpu);
        if (c != nullptr) strata::aux_cpus::note_owned_thread();
        else if (cpu < 0) strata::aux_cpus::pin_current_thread();
        for (;;) {
            std::function<void()> f;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return has_ || quit_; });
                if (!has_) return;
                f = std::move(job_);
                has_ = false;
            }
            f();
            done_.store(true, std::memory_order_release);
        }
    }
    std::mutex mu_;
    std::condition_variable cv_;
    std::function<void()> job_;
    bool has_ = false, quit_ = false;
    std::atomic<bool> done_{true};
    std::thread th_;   // last: started once the members above exist
};

// ---- --serve's conversation cache.  A chat or an agent sends the whole conversation again with every request, and
// reading it again is what made a long session wait minutes for every turn.  What a sequence leaves behind splits in
// two, and only one half needs copying:
//   * POSITIONAL state - the KV cache of the 12 QSA layers and their pooled indexer keys, the draft layer's KV.  A
//     cell is written once for its position and read only by later positions (the block scores take `dead` for the
//     block being filled, never its pooled row), so rewinding to a position just means writing from there again.
//   * RUNNING state - the 36 GDN recurrences and conv histories, each QSA layer's indexer tail (the unfinished
//     block's raw keys) and the PLE's normalized history.  Each describes "everything so far" and cannot be
//     rewound, so a checkpoint is a copy of exactly these: ~118 MB, the same set the verifier snapshots to roll
//     back rejected drafts.
// A checkpoint is only valid while the positional cells below it still hold ITS tokens, so the serve loop keeps
// just the checkpoints that are prefixes of the tokens the session holds now.
using ImgKey = strata::core::ConversationImageKey;
using ConvCheckpoint = strata::core::ConversationCheckpoint;

uint64_t fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* p = (const uint8_t*) data;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

using ConvStateSizes = strata::core::ConversationStateSizes;

ConvStateSizes conv_state_sizes(const strata::core::ModelGeometry& g, const strata::core::SessionState& ss) {
    ConvStateSizes z;
    std::string error;
    // a split stage's session owns only its layer range's state (see SessionState's carve note); the geometry
    // and the carve have already passed engine validation
    strata::core::conversation_session_sizes(g, ss, z, error);
    return z;
}

/// Copies the running state out (this session's carve only).  The caller has synchronized the device.
bool checkpoint_save(ConvCheckpoint& c, const strata::core::SessionState& ss, const strata::core::ModelGeometry& g) {
    std::string error;
    if (strata::core::conversation_checkpoint_save(c, ss, g, error)) return true;
    std::fprintf(stderr, "strata serve: checkpoint save: %s\n", error.c_str());   // the caller's ERR has no reason
    return false;
}

/// Puts a checkpoint's running state back; the positional cells below it are the caller's to guarantee.
bool checkpoint_restore(const ConvCheckpoint& c, strata::core::SessionState& ss, const strata::core::ModelGeometry& g) {
    std::string error;
    if (strata::core::conversation_checkpoint_restore(c, ss, g, error)) return true;
    std::fprintf(stderr, "strata serve: checkpoint restore: %s\n", error.c_str());
    return false;
}

/// STRATA_CKPT_POOL: the storage of dropped checkpoints, for the next save to reuse.  `give` keeps a checkpoint's
/// vectors (its tokens, pictures and marks cleared), `take` hands one back (an empty checkpoint when none is kept).
/// At most `cap` are kept, the rest are freed as before.
class CheckpointPool {
public:
    explicit CheckpointPool(bool on, size_t cap = 2) : on_(on), cap_(cap) {}
    bool on() const { return on_; }
    void give(ConvCheckpoint&& c) {
        if (!on_ || c.gdn.empty()) return;
        clear_marks(c);
        std::lock_guard<std::mutex> lk(mu_);
        if (free_.size() < cap_) free_.push_back(std::move(c));
    }
    ConvCheckpoint take() {
        std::lock_guard<std::mutex> lk(mu_);
        if (!on_ || free_.empty()) return ConvCheckpoint{};
        ConvCheckpoint c = std::move(free_.back());
        free_.pop_back();
        return c;
    }
    /// v.erase(remove_if(v, drop)) with the dropped ones given to the pool (the kept ones keep their order)
    template <class Drop> void erase_if(std::vector<ConvCheckpoint>& v, Drop drop) {
        size_t w = 0;
        for (size_t i = 0; i < v.size(); ++i) {
            if (drop(static_cast<const ConvCheckpoint&>(v[i]))) { give(std::move(v[i])); continue; }
            if (w != i) v[w] = std::move(v[i]);
            ++w;
        }
        v.erase(v.begin() + (std::ptrdiff_t) w, v.end());
    }
private:
    static void clear_marks(ConvCheckpoint& c) {
        c.ids.clear(); c.imgs.clear(); c.used = 0; c.pinned = false;
        for (ConvCheckpoint& part : c.stage_parts) clear_marks(part);
    }
    bool on_;
    size_t cap_;
    std::mutex mu_;
    std::vector<ConvCheckpoint> free_;
};

// The effective host->device bandwidth of the PCIe link: copies from pinned host memory, as the expert arena's
// reads are.  The native default share (0.55) was measured on x16 links (~26-28 GB/s); a x8 card in a x8 slot
// carries about half of that.  Returns < 0 when the probe cannot run (then the caller keeps the default).
//
// #485: a single timed burst read an x16 PCIe 4 link (RTX A3000 laptop) at 18.5, 6.9 and 5.8 GB/s in three starts -
// a link still in a low-power state, other DMA, a context not yet up to speed - and the low reading set the share.
// Such a disturbance only ever slows a copy: nothing makes a correctly timed copy faster than the link carries.  So
// the same 1 GiB is now copied as four bursts of 256 MiB, timed one by one, and the fastest is the link's figure (the
// median would still follow a disturbance that lasts through half the bursts).  Each burst takes ~10 ms on an x16
// PCIe 4 link, so the probe takes no longer than the one 1 GiB burst did.  `samples`, when given, gets every
// burst's reading for the log.
double probe_pcie_h2d_gbps(std::string* samples = nullptr) {
    constexpr size_t kBytes = 256ull << 20;
    constexpr int kBursts = 4;
    void* h = nullptr;
    void* d = nullptr;
    if (cudaMallocHost(&h, kBytes) != cudaSuccess) return -1.0;
    if (cudaMalloc(&d, kBytes) != cudaSuccess) {
        cudaFreeHost(h);
        return -1.0;
    }
    std::memset(h, 0, kBytes);   // fault the pages in before timing
    cudaMemcpyAsync(d, h, kBytes, cudaMemcpyHostToDevice);   // warmup: context up, copy engine primed
    float ms[kBursts] = {};
    // the bursts run back to back on the stream, an event between each two
    cudaEvent_t ev[kBursts + 1];
    int n_ev = 0;
    while (n_ev <= kBursts && cudaEventCreate(&ev[n_ev]) == cudaSuccess) ++n_ev;
    bool ok = n_ev == kBursts + 1;
    if (ok) {
        cudaEventRecord(ev[0]);
        for (int b = 0; b < kBursts; ++b) {
            cudaMemcpyAsync(d, h, kBytes, cudaMemcpyHostToDevice);
            cudaEventRecord(ev[b + 1]);
        }
        ok = cudaEventSynchronize(ev[kBursts]) == cudaSuccess;
        for (int b = 0; b < kBursts && ok; ++b) ok = cudaEventElapsedTime(&ms[b], ev[b], ev[b + 1]) == cudaSuccess;
    }
    for (int i = 0; i < n_ev; ++i) cudaEventDestroy(ev[i]);
    double bw = -1.0;
    if (samples != nullptr) samples->clear();
    for (int b = 0; b < kBursts && ok; ++b) {
        const double s = ms[b] > 0.01f ? (double) kBytes / (ms[b] * 1e-3) / 1e9 : -1.0;
        bw = std::max(bw, s);
        if (samples != nullptr) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%s%.1f", b == 0 ? "" : " ", s);
            *samples += buf;
        }
    }
    if (!ok) (void) cudaGetLastError();
    cudaFree(d);
    cudaFreeHost(h);
    return bw;
}

// The PCIe share of the missed experts for a link measured at `gbps`: `base` (the share measured on x16 links) from
// 20 GB/s up, and below that in proportion to the bandwidth, so the time the link spends on its share stays about
// what the x16 share costs.  Continuous (#485): before, 19.9 GB/s gave 0.42 and 20.0 the full 0.55 (and 4.0 GB/s
// gave 0.08, 3.9 none), so a reading near either edge moved the share by a quarter of its range.  From 20 GB/s up
// (an x16 PCIe 4/5 link: ~26-28 GB/s) the share is unchanged.
double pcie_frac_for_gbps(double gbps, double base) {
    return gbps <= 0.0 ? base : base * std::min(1.0, gbps / 20.0);
}

// --pool-affinity measure: `all` against `p-cores` on a hybrid CPU, timed on this pack's own experts.
//
// The pool's default layout puts a worker on every physical core, and a layer's experts are done when the slowest
// worker is.  Measured on a Core i7-13700K (8 P + 8 E), IQ3_XXS, RTX 5090, same prompt and build: `all` (15 workers)
// spent 6.7 ms per verify window in the pool and decoded 148 tok/s, `p-cores` (7 workers) 2.2 ms and 188-191 tok/s;
// in an earlier run of the same day `all` took 2.4 ms, so the difference is not constant.  Hence a measurement at
// start and not a rule.
//
// What is timed is the decode's own call: two missed experts of four tokens per layer (run_split_multi_native on
// blobs from the arena), every layer in turn, with the host spinning ~150 us between two layers as it does on the
// GPU's flags.  The two layouts take turns in kBlocks blocks and the medians are compared; `p-cores` is taken
// unless `all` is more than 10% faster (fewer threads wait on fewer cores when the two are close).  The layout
// changes no result, only where threads run.  It is one reading at start and is not repeated: on the same PC one
// server start in four read `all` as the faster layout (6.6 ms against 8.6), where the others read it 2.4 to 4 times
// slower.
strata::kernels::cpu::PoolAffinity measure_pool_affinity(strata::core::ExpertSource* src, int workers, bool host_works,
                                                         int tasks) {
    namespace cpu = strata::kernels::cpu;
    using Clock = std::chrono::steady_clock;
    const cpu::ExpertLayout& lay = cpu::expert_layout();
    const cpu::CpuTopology topo = cpu::detect_cpu_topology(true, cpu::PoolAffinity::PCores);
    if (!topo.is_hybrid) {
        std::fprintf(stderr, "strata generate: --pool-affinity measure: this CPU is not a hybrid one; the layout stays all\n");
        return cpu::PoolAffinity::All;
    }
    constexpr int kExperts = 2, kTok = 4, kRounds = 10, kBlocks = 3;
    const int64_t nl = lay.n_layers;
    std::vector<const uint8_t*> blobs((size_t) nl * kExperts, nullptr);
    bool have = src != nullptr && lay.native && (int64_t) lay.fmt.size() >= nl && nl > 0;
    uint32_t seed = 12345u;
    for (int64_t l = 0; have && l < nl; ++l)
        for (int j = 0; j < kExperts; ++j) {
            seed = seed * 1664525u + 1013904223u;
            const uint8_t* b = src->blob(l, (int64_t) ((seed >> 8) % (uint32_t) lay.n_expert));
            if (b == nullptr) { have = false; break; }
            blobs[(size_t) l * kExperts + (size_t) j] = b;
        }
    if (!have) {
        std::fprintf(stderr, "strata generate: --pool-affinity measure needs a native pack with its experts in RAM; "
                             "the layout stays all\n");
        return cpu::PoolAffinity::All;
    }
    std::vector<uint8_t> nact((size_t) nl * kTok * cpu::kNativeActBytes);
    std::vector<cpu::ActQ> qact((size_t) nl * kTok);
    std::vector<float> x((size_t) cpu::H), out((size_t) kExperts * kTok * cpu::H);
    for (int64_t l = 0; l < nl; ++l)
        for (int t = 0; t < kTok; ++t) {
            for (int i = 0; i < cpu::H; ++i) x[(size_t) i] = std::sin((float) (i + 13 * t + 7 * (int) l) * 0.031f);
            const size_t at = (size_t) l * kTok + (size_t) t;
            if (cpu::q2_native_kernels(lay.fmt[(size_t) l].gu_type)) cpu::act_quant_any(x.data(), cpu::H, qact[at]);
            else cpu::native_quant_act(lay.fmt[(size_t) l], x.data(), nact.data() + at * cpu::kNativeActBytes);
        }
    int n_workers[2] = {0, 0};
    auto block = [&](cpu::PoolAffinity aff, int which) -> double {
        cpu::ExpertPool pool(workers, /*pin=*/true, host_works, aff, tasks);
        n_workers[which] = pool.workers();
        const int host = cpu::planned_host_core(aff);
        cpu::ThreadAffinity prev;
        if (host >= 0) prev = cpu::pin_current_thread(host);
        cpu::ExpertJobMulti jobs[kExperts];
        double ms = 0.0;
        for (int r = -1; r < kRounds; ++r)   // round -1 is not timed: the workers start and the rows are touched
            for (int64_t l = 0; l < nl; ++l) {
                for (int j = 0; j < kExperts; ++j) {
                    jobs[j].blob = blobs[(size_t) l * kExperts + (size_t) j];
                    jobs[j].nt = kTok;
                    for (int t = 0; t < kTok; ++t) {
                        const size_t at = (size_t) l * kTok + (size_t) t;
                        jobs[j].act[t] = &qact[at];
                        jobs[j].nact[t] = nact.data() + at * cpu::kNativeActBytes;
                        jobs[j].out[t] = out.data() + ((size_t) j * kTok + (size_t) t) * cpu::H;
                    }
                }
                const Clock::time_point t0 = Clock::now();
                pool.run_split_multi_native(lay.fmt[(size_t) l], jobs, kExperts);
                const Clock::time_point t1 = Clock::now();
                if (r >= 0) ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                while (std::chrono::duration<double, std::micro>(Clock::now() - t1).count() < 150.0) {}
            }
        if (host >= 0) cpu::restore_thread_affinity(prev);
        return ms / kRounds;
    };
    double all_ms[kBlocks], p_ms[kBlocks];
    for (int b = 0; b < kBlocks; ++b) {
        all_ms[b] = block(cpu::PoolAffinity::All, 0);
        p_ms[b] = block(cpu::PoolAffinity::PCores, 1);
    }
    std::sort(all_ms, all_ms + kBlocks);
    std::sort(p_ms, p_ms + kBlocks);
    const double a = all_ms[kBlocks / 2], p = p_ms[kBlocks / 2];
    const bool take_p = p <= a * 1.10;
    std::fprintf(stderr, "strata generate: --pool-affinity measure: %d experts x %d tokens per layer over %lld layers: "
                         "all (%d workers) %.2f ms, p-cores (%d workers) %.2f ms (medians of %d) -> %s\n",
                 kExperts, kTok, (long long) nl, n_workers[0], a, n_workers[1], p, kBlocks, take_p ? "p-cores" : "all");
    return take_p ? cpu::PoolAffinity::PCores : cpu::PoolAffinity::All;
}

// 0.1.42, CUDA single-GPU serve, no --pcie-frac: the share refined from what the CPU pool measures in the first decode
// windows.  The balance share of the missed experts is link / (link + CPU rate).  The link is the start-up probe; the CPU
// rate is the expert bytes over the pool's wall time per missed expert (ExpertDispatch::ms_run / multi_misses), which the
// engine already times.  The best shares measured with the route tail skip on (RTX 3060 0.15-0.20, Tesla P100 0.20, RTX
// A4000 flat from 0.15 to 0.40; the 0.1.41 rule gave 0.55, 0.29, 0.30) lie within one 0.05 step of link / (link + 1.85 x pool rate).
// Three updates at most (after 48, then 96 and 96 more windows), each rounded to 0.05, so a noisy window cannot move
// the share by less than a step and it stops moving after the third.  STRATA_PCIE_FRAC_DEFAULT=old keeps the 0.1.41 rule.
struct PcieAuto {
    static constexpr double kPoolFactor = 1.85;
    bool on = false;
    double link_gbps = 0.0, blob_bytes = 0.0;
    double mark_ms = 0.0;
    int64_t mark_jobs = 0;
    double acc_ms = 0.0;
    int64_t acc_jobs = 0;
    int acc_win = 0, updates = 0;
    void mark(double ms_run, int64_t jobs) { mark_ms = ms_run; mark_jobs = jobs; }
    /// After a decode window: the new share, or -1 while there is nothing to change.  `tc_us`/`rate` get the reading.
    double window(double ms_run, int64_t jobs, double cur, double* tc_us, double* rate) {
        if (!on) return -1.0;
        acc_ms += ms_run - mark_ms;
        acc_jobs += jobs - mark_jobs;
        mark(ms_run, jobs);
        ++acc_win;
        if (acc_win < (updates == 0 ? 48 : 96)) return -1.0;
        if (acc_jobs < 300) {          // too few missed experts to read the pool's speed (everything hits): look again
            if (acc_win >= 2000) on = false;
            return -1.0;
        }
        const double tc = acc_ms * 1000.0 / (double) acc_jobs;
        acc_ms = 0.0;
        acc_jobs = 0;
        acc_win = 0;
        if (++updates >= 3) on = false;
        if (!(tc > 1.0) || !(blob_bytes > 0.0) || !(link_gbps > 0.0)) return -1.0;
        const double r = blob_bytes / (tc * 1e-6) / 1e9;
        const double f = std::min(0.60, std::max(0.10, std::round(link_gbps / (link_gbps + kPoolFactor * r) * 20.0) / 20.0));
        *tc_us = tc;
        *rate = r;
        return std::fabs(f - cur) < 0.025 ? -1.0 : f;
    }
};

}  // namespace

// #1425 #1549 #1629: --ple-io direct (the default) does unbuffered random reads of the n-gram table; on some drives
// (DRAM-less NVMe, Windows unbuffered I/O) they run at 34 MB/s or 4-8 tok/s and a prompt reads layer 1 for minutes.
// Rows per second of a short cold probe: a few batches of random rows through the same batched reader a prompt chunk
// uses, stopped after ~0.6 s. Returns <= 0 when nothing could be measured.
static double ple_direct_rows_per_s(strata::kernels::PleTable& t) {
    const uint64_t n = t.rows();
    if (n < 4096) return -1.0;
    constexpr size_t kTok = 4;   // 64 rows a batch
    std::vector<uint32_t> rows(kTok * 16);
    std::vector<float> out(kTok * 2560);
    uint64_t x = 0x9E3779B97F4A7C15ull ^ n;
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t done = 0;
    double el = 0;
    for (int b = 0; b < 64; ++b) {
        for (auto& r : rows) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; r = (uint32_t) (x % n); }
        std::string e;
        if (!t.gather_batch(rows.data(), kTok, out.data(), e)) return -1.0;
        done += rows.size();
        el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (el > 0.6) break;
    }
    return el > 0 ? (double) done / el : -1.0;
}

namespace {

// `next` in parse_options: an option given without its value.
struct MissingValue {
    const char* option;
};

struct ScratchFree {
    strata::core::SessionLoopScratch* p = nullptr;
    ~ScratchFree() { if (p != nullptr) p->free(); }
};
struct TokenGraphFree {
    strata::core::TokenGraph* p = nullptr;
    ~TokenGraphFree() { if (p != nullptr) strata::core::token_graph_free(*p); }
};

// --kv-grow's state (see kvg_start, kvg_ensure and kvg_trim in start-up)
struct KvGrow {
    bool on = false;
    int64_t top = 0, lo = 0;          // slots [lo, top) hold no expert; their whole chunks may be with the K/V
    int64_t floor = 128;              // the cache keeps at least these slots
    int64_t step = 8192;              // the K/V grows in whole steps of cells
    int64_t cells = 0;                // what every K/V pool holds now
    std::vector<strata::core::VmmChunk> spare;   // out of the cache, not (yet) in the K/V
    int64_t grows = 0, trims = 0, fresh = 0, evicted = 0, refilled = 0;
};
// The auto scan used to walk a fixed list of sizes - 32768, 16384, 8192, 6144, ... - and take the first that
// fit.  That list is coarse exactly where a rig needs it: this one affords ~8,700 tokens and was handed 8192,
// and 8704 is not on it.  `bytes_needed` is a sum of (T x positive constant) terms plus a max of such sums,
// so it rises monotonically with T, and so does every test the scan applies - which makes the largest size
// that fits a bisection on the 256-token grid the prompt path already works on (`request_chunk` rounds up to
// it).  log2(32768/256) = 7 probes, against the list's 10, at one `bytes_needed` per probe.
const auto biggest_chunk = [](int64_t ceiling, auto&& ok) -> int64_t {
    if (ceiling < 256) return 0;
    int64_t lo = 1, hi = ceiling / 256, best = 0;   // n = T / 256, and ok() is monotone in n
    while (lo <= hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        if (ok(mid * 256)) { best = mid * 256; lo = mid + 1; } else hi = mid - 1;
    }
    return best;
};
std::string string_printf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
std::string string_printf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list measure;
    va_copy(measure, ap);
    const int n = std::vsnprintf(nullptr, 0, fmt, measure);
    va_end(measure);
    std::string s(n > 0 ? (size_t) n : 0, '\0');
    std::vsnprintf(s.data(), s.size() + 1, fmt, ap);
    va_end(ap);
    return s;
}
// the penalty-history buffer: one row per verify-window row (`penalty_rows`), each the last
// `penalty_last_n` tokens that row's pick follows, -1 padded in front.  Allocated once at the cap for
// the widest window; a request without penalties gets a null buffer and takes the byte-for-byte
// neutral path (no upload, no buffer handed to the sampler).
constexpr int kPenaltyWindowCap = 4096;
constexpr size_t kHistSlots = (size_t) kPenaltyWindowCap * (size_t) strata::kernels::kVerifyMaxT;
constexpr size_t kResPatchCap = 16384;   // the residency entries res_sync patches in one kernel
// ---- WHO BORROWS, AND FROM WHOSE CACHE.  One entry per prompt path: CUDA0's (layers [0, split_at[0]),
// which is the whole model without a split) borrowing the tail of CUDA0's cache, then one per stage
// borrowing the tail of ITS OWN cache.  A loan is sized by the exact `Prefill::bytes_needed` for the
// chunk, is laid out by `Prefill::relayout`, and is refilled before any window reads - so outside the
// prompt the whole cache is expert cache.  THIS IS THE POINT OF THE STRUCT: the loan used to exist only
// for CUDA0, and `no_prefill_borrow` made every stage instead withhold a chunk-sized reserve from its
// cache for the entire session, which is what cost the 4-way its context (see the note at the top of the
// layer-split block).  A stage may only lend the rows for ITS OWN layers: `host_res` is one table whose
// slot values are indices into whichever cache owns the layer, so a loan that marked rows by slot number
// alone would hand CUDA0 a slot belonging to another stage's cache.
struct PfPart {
    strata::core::ExpertCache* cache = nullptr;
    const strata::core::SessionState* ses = nullptr;
    strata::prefill::Prefill* sp = nullptr;
    int dev = -1;                  // -1: leave the device alone (CUDA0)
    int64_t lb = 0, le = 0;        // the layers whose rows this cache holds - the only rows it may lend
    int32_t first = -1;            // the first slot it may lend, for the chunk that was chosen
    int32_t first_now = -1;        // where its buffers are laid out now
    int64_t lent_chunk = 0;
    std::vector<std::pair<int32_t, int32_t>> lent;
};
struct ReqTimes {
    int ckpts = 0;
    double sync_ms = 0.0, save_ms = 0.0, evict_ms = 0.0, read_ms = 0.0, windows_ms = 0.0;
};
// the look-ahead thread and the exchanges must not touch the swap space once it is gone (declared after it:
// destroyed first)
struct FsDetach {
    strata::core::RouterLookahead* la = nullptr;
    strata::core::FileExpertSource* src = nullptr;
    ~FsDetach() {
        if (la != nullptr) la->set_foresight(nullptr);
        if (src != nullptr) src->set_foresight_swap(nullptr);
    }
};
struct PreLaunchOff {   // the hooks' captures die before the verifiers
    strata::core::Verifier* v = nullptr;
    strata::core::ExpertDispatch* d = nullptr;
    int32_t* pairs = nullptr;
    ~PreLaunchOff() {
        if (v != nullptr) v->set_pre_launch(nullptr);
        if (d != nullptr) d->dres_check = nullptr;
        if (pairs != nullptr) cudaFreeHost(pairs);
    }
};
struct ASwap { int32_t layer, in, out, slot; int home; int64_t j; bool exchange; };
struct AHome { strata::core::ExpertCache* cache; cudaStream_t stream; cudaEvent_t ev; int dev; bool used; };
enum class AState { Idle, CopyBack, SwapIn, RamFence, Commit };
// the progress state marks a request from its sync until its finish, whichever way it ends
struct BusyScope {
    BusyScope() {
        strata::core::progress().allow_until_ms.store(0);
        strata::core::progress().busy.store(true);
        strata::core::progress_at("request");
    }
    ~BusyScope() {
        strata::core::progress().busy.store(false);
        strata::core::progress().allow_until_ms.store(0);
        strata::core::progress_at("idle");
    }
};
struct DecSnap {
    double wait, pool, host, plan, actq, jobs, run;
    int64_t misses, entries, hits, pcie;
};
using strata::program::CancelFn;
using strata::program::ImageSpan;
using strata::program::kDead;
using strata::program::kImagePad;
using strata::program::kInterrupted;
using strata::program::kInvalidated;
using strata::program::kNoToken;
using strata::program::kOk;
using strata::program::kRefused;
using strata::program::ProgressFn;
using strata::program::RequestStats;
using strata::program::Sampling;
using strata::program::SyncProgress;
// Everything `strata --serve` builds before its first request (start-up, then the serve set-up), and the one
// session's state.  The members are the old main()'s locals, declared in the order it declared them, so they are
// destroyed in the order its `return` destroyed them.  Members hold pointers to each other (and the library holds
// some: set_native_embed, ss.db, ss.ple.table), so an EngineState is never moved.
struct EngineState {
    EngineState() = default;
    EngineState(const EngineState&) = delete;
    EngineState& operator=(const EngineState&) = delete;

    int start_up(int argc, char** argv);
    int parse_options(int argc, char** argv);
    int serve_setup();
    int sync_request(const int32_t* tokens, int64_t n, const ImageSpan* images, size_t image_count);
    int decode_window(int32_t token, const Sampling& sampling, int budget, int32_t* out, int cap);
    int finish_request();
    void invalidate_session();
    // QUIT or the end of input: the open request ends, the round in flight lands, the profile is saved.  False when
    // the engine is dead.
    bool shut_down();
    // a dead engine at close: the host memory that is cheap to give back
    void drop_host_state();
    // STRATA_TRACE=1: one stderr line per step of a request (the log shows where a request stops)
    void tr(const char* what, long long a = -1, long long b = -1);
    // between windows: `drain` finishes the round in flight and starts none; `count` false moves it on without
    // counting a window.  False when a copy failed.
    bool adapt_tick(bool drain, bool count = true);
    DecSnap dec_snap() const;
    void report_progress(int64_t done);
    bool stop_requested() const;
    int refuse(std::string message);   // kRefused, with the message for Session::error
    int die(std::string message);      // the engine is dead from here on: kDead
    // Prints the message to stderr as start-up did, keeps it (without the trailing newline) for Engine::open's
    // error, and returns status.
    int fail(int status, const char* fmt, ...) __attribute__((format(printf, 3, 4)));

    bool nothing_to_serve = false;   // --version or --help was printed
    std::string error_message;

    Options o;
    bool pcie_given = false;
    const char* kv_rot = nullptr;
    strata::kernels::RopeScaling rope_cfg;   // type filled at the option checks; the rest at the resolution below them
    const char* isa_floor = nullptr;
    std::string err;
    bool native_pack = false;
    int64_t slot_guard = 0;
    std::function<bool()> pcie_balance;
    strata::core::ModelGeometry g;
    int64_t K = 0;
    strata::core::NativeEmbed native_embed;
    std::set<std::string> skip;
    bool hc_req8_env = false;
    bool hc_q8 = false;
    uint64_t pool_bytes = 0;
    void* arena = nullptr;
    std::chrono::steady_clock::time_point load_t0;
    strata::core::WeightTable wt;
    strata::core::NativeDense native_dense;
    int32_t* d_mrope = nullptr;
    std::vector<int32_t> mrope_host;
    strata::core::SessionState ss;
    void* sbuf = nullptr;   // allocated with the session
    cudaStream_t main_stream = nullptr;
    void* main_cs = nullptr;
    strata::core::Doorbell db;
    strata::kernels::PleTable ple_table;
    std::vector<float> ple_emb_host;
    float* ple_emb_dev = nullptr;
    float* ple_scratch = nullptr;
    float* d_parts = nullptr;
    std::vector<std::pair<int32_t, int32_t>> profile;
    std::vector<std::pair<int32_t, int32_t>> profile_loaded;
    bool pf_borrow = false;
    strata::core::MtpDrafter mtp;
    const strata::core::WeightRef* wo = nullptr;
    int64_t n_vocab = 0;
    strata::core::NativeHead native_head;
    std::vector<float> logits;
    float* d_logits = nullptr;
    bool prefill_head_alias_bound = false;
    const void* prefill_head_alias_at = nullptr;   // where the head came back the first time
    strata::core::FileExpertSource src;
    strata::core::ArenaExpertSource arena_src;
    strata::core::ExpertSource* srcp = nullptr;
    std::optional<strata::kernels::cpu::ExpertPool> pool;   // its threads start where start-up built it
    strata::core::ExpertCache xcache;
    bool auto_cache = false;
    bool reserve_adapted = false;   // #496: the auto sizing lowered the reserve so a small card's cache fits
    std::vector<int64_t> sized_slots;
    uint64_t per_layer_bytes = 0;
    int64_t prefilled = 0;
    Drive drive;
    strata::core::RouterLookahead lookahead;
    bool fs_ahead = false;
    bool la_stats = false;
    void* hit_scratch = nullptr;
    int32_t* d_hit_slot = nullptr;
    int32_t* d_hit_dst = nullptr;
    uint8_t* d_hit_q8 = nullptr;
    float* d_hit_q8_scale = nullptr;   ///< R4.2h: the fp32 activation scales the CPU path also uses
    float* d_hit_out = nullptr;
    strata::core::HitFn hit_fn = nullptr;
    strata::core::SessionGraphs gr;
    float* d_emb = nullptr;
    int* d_next = nullptr;
    strata::core::SessionLoopScratch loop_scratch;
    ScratchFree scratch_free;
    strata::core::TokenGraph tgraph;
    TokenGraphFree tgraph_free;
    std::vector<int32_t> host_res;
    std::function<cudaError_t(int32_t*)> res_put;
    int32_t* d_res = nullptr;
    int32_t* d_hit_count = nullptr;
    strata::core::TokenHits thits;
    bool graph_hits = false;
    std::function<int64_t(int64_t, int64_t)> request_chunk;
    std::function<int64_t(int64_t, int64_t)> equal_chunk;
    int64_t kAutoLendPct = 0;
    int64_t auto_ceiling = 0;
    KvGrow kvg;
    std::function<void(int64_t)> kvg_start;
    std::function<bool(int64_t, const std::function<void()>&)> kvg_ensure;
    std::function<bool(int64_t)> kvg_trim;
    std::function<bool()> share_default_ok;

    // ---- serve_setup: what every request uses, in the old serve block's declaration order
    bool use_mtp = false;
    std::optional<strata::prefill::Prefill> sp;
    bool share_pool = false;
    void* borrow = nullptr;
    uint64_t borrow_bytes = 0;
    int32_t lend_first = -1;          // the first slot the prompt path may borrow (its largest chunk)
    std::function<int64_t(const strata::core::ExpertCache&, uint64_t)> cache_slots_for;
    std::function<int64_t(const PfPart&, int64_t)> part_slots;
    std::function<uint64_t(const PfPart&, int32_t)> part_bytes;
    std::vector<PfPart> pf_parts;
    std::function<int()> init_prompt_paths;
    int32_t* d_hist = nullptr;
    std::vector<int32_t> hist_stage;
    strata::core::Verifier ver;
    strata::core::VerifyHits vh;
    // the pool the verify windows call
    strata::core::PoolMultiFn win_pool_fn = nullptr;
    void* win_pool_user = nullptr;
    std::vector<int64_t> cur;
    // ---- the conversation cache (see ConvCheckpoint).  `live` is what the session holds right now: the tokens
    // it has consumed, so a request that starts with exactly them continues without any copy.  `checks` are the
    // saved points; every one of them is a prefix of `live` (the loop drops the rest), so they form a chain -
    // the radix cache's tree collapsed onto the one branch of history whose cells the session holds.  The
    // chain's root is the deepest point every request so far shared (the end of the system prompt, in
    // practice); the retention policy pins it and rotates the rest LRU (conv_cache.hpp), so a NEW chat that
    // shares that prefix mounts through it instead of reading it again.
    std::vector<int32_t> live;
    std::vector<ImgKey> live_imgs, req_imgs;
    bool live_ok = false;
    // #879: a window met non-finite logits: the next request drops every cached state first and reads from token 0
    bool nan_flush = false;
    std::vector<ConvCheckpoint> checks;
    // STRATA_CKPT_POOL: the dropped checkpoints' storage, reused
    std::optional<CheckpointPool> ckpt_pool;
    uint64_t check_clock = 0;   // the checkpoints' LRU clock; creation and every use advance it
    int64_t tail_ckpt_len = -1;   // --prompt-cache-tail: the length of the one tail checkpoint alive (-1 = none)
    // --prefill-pipe: the plan of the prompt segment being read (read_part sets it around its runs), for the
    // checkpoints taken inside it
    ConvCheckpoint::PromptPlan seg_plan;
    int64_t pp_total = 0, pp_from = 0, pp_next_check = 0;
    bool pp_tail_saved = false;
    // #471: the position the prompt pass has read up to (a chunk's or a window's end): what a request cancelled
    // mid-read reports as read, instead of the whole prompt
    int64_t pp_reached = 0;
    // #1620: the session holds exactly the tokens [0, pp_reached) of the prompt (no chunk or window is in flight, none
    // failed): a request cancelled here keeps them as the live conversation instead of dropping what it restored
    bool pp_exact = false;
    Clock::time_point pp_t0;
    // #879: this request met a window whose head logits were not all finite (nonfinite_hit): it ends with
    // DONE ... nonfinite, emitting none of that window's tokens
    bool nf_hit = false;
    std::function<bool(const strata::core::Verifier&, const char*)> nonfinite_hit;
    std::function<std::vector<ImgKey>(const std::vector<ImgKey>&, int64_t)> imgs_below;
    // #613: why the last checkpoint_at failed - a failed device sync is the GPU itself (a hang Windows then resets),
    // not the checkpoint, and the error says so
    std::string ckpt_why;
    // STRATA_TRACE: what each checkpoint cost - the device syncs, the copies into RAM (the vectors' first fill
    // included), the eviction that frees one - one line per checkpoint
    bool ckpt_trace = false;
    ReqTimes req_times;
    std::function<bool()> gpu_sync_ok;
    std::function<bool(int64_t, bool)> checkpoint_at;
    // Foresight swap space (STRATA_FS_SLOTS; unset = null and nothing changes)
    std::unique_ptr<strata::core::ForesightSwap> fs_swap;
    FsDetach fs_detach;
    PcieAuto pcie_auto;
    std::function<void(double&)> pcie_auto_step;
    // #477 --expert-profile-save: what the adaptive tier learned, kept across restarts (opt-in; off: `heat` stays
    // empty and nothing below runs).  It needs the adaptive tier's counts and the residency table.
    std::vector<double> heat;
    Clock::time_point profile_saved_at;
    cudaStream_t adapt_stream = nullptr;
    // plan v0.3 P6: swaps in flight - (residency index, slot) admitted when adapt_ev has completed
    std::vector<std::pair<int32_t, int32_t>> pending;
    int pending_age = 0;   // the windows the pending swaps have waited (adapt_lag)
    std::vector<void*> pin_live;   // the swaps' locked arena pages (pin_blob), unlocked once they have landed
    cudaEvent_t adapt_ev = nullptr;
    // 0.1.43 STRATA_ADAPT_OVERLAP=1 (adapt_overlap above): the single-GPU, plain-RAM-arena decode loop only
    strata::core::AdaptGate agate;
    bool aflipped = false;   // the pending swaps are in the tables and their copies may still be landing
    bool aov = false;
    int64_t aov_flips = 0;
    std::function<cudaError_t()> res_upload;
    std::function<void(bool)> apply_pending;
    bool res_checking = false;
    std::vector<int32_t> res_back;
    int64_t res_windows = 0, res_bad = 0, res_bad_windows = 0;
    int32_t* res_pairs = nullptr;
    std::function<bool(bool, std::string&)> res_sync;
    PreLaunchOff pre_launch_off;
    int64_t adapt_nswaps = 0;   // the blocking tier's swaps chosen, cumulative (the decode timing line's swaps/window)
    std::function<bool()> adapt;
    AState astate = AState::Idle;
    std::vector<ASwap> aswaps;
    std::vector<float> a_usage;
    std::vector<int32_t> a_res;
    int64_t a_win = 0, a_last = 0, a_rounds = 0, a_swapped = 0;
    double a_ms = 0;                  // wall time from a round's start to its flip, summed (the "ms/round" stat)
    Clock::time_point a_t0;
    std::atomic<bool> a_err{false};
    double a_pick_ms = 0;             // the arena's inline choice, summed (the stats line)
    size_t a_lo = 0, a_hi = 0, a_bi = 0;   // the batch being copied in (a_copy_in), and where the next one starts
    std::vector<AHome> ahomes;   // [0] = CUDA0's cache
    std::unique_ptr<JobThread> ajob;
    bool a_arena = false;
    int64_t a_cap = 0;
    bool a_inline = false;
    bool a_wait = false;
    size_t a_batch = 0;
    std::function<void(AHome&)> a_mark;
    std::function<void()> a_pick;
    std::function<void(int)> a_back;
    std::function<const uint8_t*(const ASwap&)> a_from;
    std::function<void(const ASwap&, const uint8_t*)> a_in_one;
    std::function<void()> a_choose;
    std::function<void()> a_copy_in;
    std::function<void()> a_post_batch;
    std::function<bool(bool)> a_ready;
    std::function<void()> adapt_quiesce;
    std::function<void(const char*)> save_profile;
    bool trace = false;
    std::string info;   // the INFO line's fields
    int64_t rounds = 0;
    int S = 0;
    int S_mtp = 0;   // the MTP's windows; suffixes go up to S
    std::optional<strata::spec::SuffixDrafter> sfx;
    std::optional<strata::spec::DraftPolicy> policy;   // MTP or lookup window, learned over the whole process
    // The vision path (--vision): GENI <max_new> <embeddings file> <id,id,...> carries images, one ImageSpan each;
    // each image's rows go to its run of <|image_pad|> tokens, whose M-RoPE positions are mtmd's: t = p,
    // h = p + y, w = p + x, and the text after the image continues at p + max(nx, ny).
    bool mrope_identity = true;
    std::vector<const float*> row_ptr;
    bool ready = false;   // serve_setup finished

    // ---- the session and its request in flight (sync_request, decode_window, finish_request)
    int64_t session_ctx = 0;
    ProgressFn progress_fn;   // the sync call's callbacks, set for its duration only
    CancelFn cancel_fn;
    std::string session_error;
    bool dead = false;   // every later call returns kDead; close skips the destructors
    bool request_open = false;   // sync started a request that finish_request has not ended
    bool decode_ok = false;      // the state is ready for the next decode window
    int req_ckpt = 1;
    double req_pcie_frac = 0.0;
    double req_spec_min_p = 0.0;
    std::optional<BusyScope> busy_scope;
    int64_t streamed0 = 0;   // the DONE line's experts_streamed= is the difference
    int64_t resume = 0;
    int64_t req_chunk = 0, req_chunk_len = 0;   // the DONE line's chunk=: the longest batched segment's first chunk
    bool cancelled = false;
    double prompt_ms = 0.0;
    int64_t p = 0;
    int32_t x = 0;
    bool drafts_ok = false;   // drafts[] follow x: the last window drafted, and x is its own sample
    std::vector<int32_t> drafts, window, outv;
    std::vector<float> dprob;
    std::vector<int32_t> sfx_out;
    bool sfx_on = false;
    bool first_window = true;
    int64_t produced_n = 0, sfx_windows = 0, sfx_drafts = 0, sfx_ok = 0;
    int64_t chain_windows = 0, chain_drafts = 0, chain_ok = 0;   // --lookup-chain's own counts
    int64_t t2_rej[8] = {}, t2_hit[8] = {};   // STRATA_MTP_TOP2: rejections inside the MTP drafts by depth, runner-up hits
    int64_t dp_tested[8] = {}, dp_ok[8] = {};   // STRATA_SPEC_DEPTH=1: MTP drafts tested / accepted by depth
    std::vector<int32_t> cbuf, ctail;
    std::optional<strata::spec::PromptLookupSource> lookup_src;
    int64_t draft_offered = 0, draft_accepted = 0;
    // what the session holds once this request is done: the prompt read so far, then every committed token
    std::vector<int32_t> consumed;
    Clock::time_point d0;
    DecSnap ds0{};
    double dt_run = 0, dt_commit = 0, dt_draft = 0;
    double dt_adapt_wait = 0, dt_adapt_join = 0;   // the tier's wait before a window / join after the draft (ms, summed)
    int64_t dt_sw0 = 0;
    int64_t dec_windows = 0, dec_T = 0;
    int64_t decode_hits0 = 0;
    // CS-T: the RAM and file tiers of this request (the mmap source; 0 with the arena)
    int64_t ram0 = 0, files0 = 0;
    uint64_t file_bytes0 = 0;
    int64_t decode_look0 = 0;
    int64_t offload0 = 0;   // #588
    int hist_n = 0;
    bool sampling_applied = false;   // applied_sampling is on the verifier (cleared by each sync)
    Sampling applied_sampling;
    RequestStats stats;
};

int EngineState::fail(int status, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list measure;
    va_copy(measure, ap);
    const int n = std::vsnprintf(nullptr, 0, fmt, measure);
    va_end(measure);
    std::string msg(n > 0 ? (size_t) n : 0, '\0');
    std::vsnprintf(msg.data(), msg.size() + 1, fmt, ap);
    va_end(ap);
    std::fputs(msg.c_str(), stderr);
    while (!msg.empty() && msg.back() == '\n') msg.pop_back();
    error_message = std::move(msg);
    return status;
}

}  // namespace

int EngineState::parse_options(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) throw MissingValue{what};
            return argv[++i];
        };
        bool parsed = true;
        if (a == "--help" || a == "-h") { usage(); nothing_to_serve = true; return 0; }
        else if (a == "--pack") o.pack = next("--pack");
        else if (a == "--max-context") o.max_context = std::atoll(next("--max-context"));
        else if (a == "--rope-scaling") o.rope_scaling = next("--rope-scaling");
        else if (a == "--rope-scale") o.rope_scale = std::atof(next("--rope-scale"));
        else if (a == "--rope-freq-base") o.rope_freq_base = std::atof(next("--rope-freq-base"));
        else if (a == "--rope-freq-scale") o.rope_freq_scale = std::atof(next("--rope-freq-scale"));
        else if (a == "--yarn-orig-ctx") o.yarn_orig_ctx = std::atof(next("--yarn-orig-ctx"));
        else if (a == "--yarn-ext-factor") o.yarn_ext_factor = std::atof(next("--yarn-ext-factor"));
        else if (a == "--yarn-attn-factor") o.yarn_attn_factor = std::atof(next("--yarn-attn-factor"));
        else if (a == "--yarn-beta-fast") o.yarn_beta_fast = std::atof(next("--yarn-beta-fast"));
        else if (a == "--yarn-beta-slow") o.yarn_beta_slow = std::atof(next("--yarn-beta-slow"));
        else if (a == "--ple-gguf") o.ple_gguf = next("--ple-gguf");
        else if (a == "--no-ple") o.no_ple = true;
        else if (a == "--ple-io") { o.ple_io = next("--ple-io"); o.ple_io_explicit = true; }
        else if (a == "--ple-row-cache") o.ple_row_cache = std::atoll(next("--ple-row-cache"));
        else if (a == "--ple-inflight") o.ple_inflight = std::atoi(next("--ple-inflight"));
        else if (a == "--ple-delay-us") o.ple_delay_us = std::atof(next("--ple-delay-us"));
        else if (a == "--ple-sync-submit") o.ple_sync_submit = true;
        else if (a == "--kv") o.kv = next("--kv");
        else if (a == "--kv-resident") o.kv_resident = std::atoll(next("--kv-resident"));
        else if (a == "--kv-grow") o.kv_grow = true;
        else if (a == "--no-kv-grow") o.kv_grow = false;
        else if (a == "--stream-token") o.stream_token = true;
        else if (a == "--gr-fp32-activations") o.gr_fp32_activations = true;
        else if (a == "--gr-native-mmvf") o.gr_native_mmvf = true;
        else if (a == "--native-bf16") o.native_bf16 = true;
        else if (a == "--native-bf16-extra") o.native_bf16_extra = true;
        else if (a == "--native-ple-key") o.native_ple_key = true;
        else if (a == "--native-moe-combine") o.native_moe_combine = true;
        else if (a == "--native-gdn") o.native_gdn = true;
        else if (a == "--native-flash-attn-short") o.native_flash_attn_short = true;
        else if (a == "--native-qsa-indexer") o.native_qsa_indexer = true;
        else if (a == "--native-qsa") o.native_qsa = true;
        else if (a == "--native-rope") o.native_rope = true;
        else if (a == "--native-ple-postops") o.native_ple_postops = true;
        else if (a == "--native-router") o.native_router = true;
        else if (a == "--cpu-oracle-q8-0") o.cpu_oracle_q8_0 = true;
        else if (a == "--native") o.native_preset = next("--native");
        else if (a == "--native-head-gguf") o.native_head_gguf = next("--native-head-gguf");
        else if (a == "--embd-gguf") o.embd_gguf = next("--embd-gguf");
        else if (a == "--native-dense-gguf") o.native_dense_gguf.push_back(next("--native-dense-gguf"));
        else if (a == "--no-capture") o.no_capture = true;
        else if (a == "--no-pool") o.no_pool = true;
        else if (a == "--pool-workers") o.pool_workers = std::atoi(next("--pool-workers"));
        else if (a == "--pool-tasks") {
            const char* v = next("--pool-tasks");
            char* end = nullptr;
            const long tasks = std::strtol(v, &end, 10);
            if (end == v || *end != '\0' || tasks < 0 || tasks > strata::kernels::cpu::ExpertPool::kMaxTasks) {
                return fail(2, "strata generate: --pool-tasks expects an integer in 0..4096 (0 = automatic)\n");
            }
            o.pool_tasks = (int) tasks;
        }
        else if (a == "--host-core") {
            o.host_core = next("--host-core");
            if (o.host_core != "first" && o.host_core != "last" && o.host_core != "sibling") {
                return fail(2, "strata generate: unknown --host-core value '%s' (expected first, last or sibling)\n",
                            o.host_core.c_str());
            }
        }
        else if (a == "--aux-cpus") {
            o.aux_cpus = next("--aux-cpus");
            std::string aerr;
            strata::aux_cpus::Config acfg;
            if (!strata::aux_cpus::parse_spec(o.aux_cpus, acfg, aerr)) {
                return fail(2, "strata generate: %s\n", aerr.c_str());
            }
        }
        else if (a == "--pool-affinity") {
            const std::string v = next("--pool-affinity");
            if (v == "auto") o.pool_affinity = strata::kernels::cpu::PoolAffinity::Auto;
            else if (v == "p-cores" || v == "pcores") o.pool_affinity = strata::kernels::cpu::PoolAffinity::PCores;
            else if (v == "all") o.pool_affinity = strata::kernels::cpu::PoolAffinity::All;
            else if (v == "measure") o.pool_measure = true;
            else {
                return fail(2, "strata generate: unknown --pool-affinity value '%s' (expected auto, p-cores, all or measure)\n", v.c_str());
            }
        }
        else if (a == "--no-host-worker") o.no_host_worker = true;
        else if (a == "--no-ple-prefetch") o.no_ple_prefetch = true;
        else if (a == "--coupled-draft") o.coupled_draft = true;
        else if (a == "--no-coupled-draft") o.coupled_draft = false;
        else parsed = false;
        // The chain continues here in a second statement: one chain of 120+ `else if` passed MSVC's limit of 128
        // nested blocks (C1061).  The order of the tests and what each does are unchanged.
        if (!parsed) {
        if (a == "--expert-cache") {
            const std::string v = next("--expert-cache");
            o.expert_cache = (v == "auto") ? -1 : std::atoi(v.c_str());
        }
        else if (a == "--vram-reserve-mib") { o.vram_reserve_mib = std::atoi(next("--vram-reserve-mib")); o.vram_reserve_given = true; }
        else if (a == "--prefill") {
            const std::string v = next("--prefill");
            o.prefill_auto = v == "auto" || v.rfind("auto:", 0) == 0;
            // #282: auto:N (N = 16384 or 32768) or STRATA_PREFILL_AUTO_MAX lets auto take chunks above 8192
            const char* env_max = std::getenv("STRATA_PREFILL_AUTO_MAX");
            const long long want_max = v.rfind("auto:", 0) == 0 ? std::atoll(v.c_str() + 5)
                                     : (o.prefill_auto && env_max != nullptr ? std::atoll(env_max) : 8192);
            o.prefill_auto_max = want_max >= 32768 ? 32768 : want_max >= 16384 ? 16384 : 8192;
            o.prefill_chunk = o.prefill_auto ? o.prefill_auto_max : std::atoll(v.c_str());
        }
        else if (a == "--no-split-rows") o.no_split_rows = true;
        else if (a == "--no-prefill-borrow") o.no_prefill_borrow = true;
        else if (a == "--spec") o.spec = std::atoi(next("--spec"));
        else if (a == "--mtp") o.mtp = next("--mtp");
        else if (a == "--mtp-window") o.mtp_window = std::atoll(next("--mtp-window"));
        else if (a == "--pcie-frac") o.pcie_frac = std::atof(next("--pcie-frac"));
        else if (a == "--adapt-every") o.adapt_every = std::atoi(next("--adapt-every"));
        else if (a == "--adapt-decay") o.adapt_decay = (float) std::atof(next("--adapt-decay"));
        else if (a == "--adapt-min-gain") o.adapt_min_gain = (float) std::atof(next("--adapt-min-gain"));
        else if (a == "--adapt-async") o.adapt_async = std::atoi(next("--adapt-async")) != 0 ? 1 : 0;
        else if (a == "--spec-min-p") o.spec_min_p = std::atof(next("--spec-min-p"));
        else if (a == "--spec-split") o.spec_split = true;
        else if (a == "--pcie-mode") o.pcie_mode = next("--pcie-mode");
        else if (a == "--serve") o.serve = true;
        else if (a == "--vision") o.vision = true;
        else if (a == "--prompt-cache") o.prompt_cache = std::max(0, std::atoi(next("--prompt-cache")));
        else if (a == "--memory-limit-mib") {
            const std::string value = next(a.c_str());
            int64_t number = 0;
            const auto result = std::from_chars(value.data(), value.data() + value.size(), number);
            const int64_t limit = INT64_MAX / (1024 * 1024);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || number < 0 || number > limit) {
                return fail(2, "%s needs a nonnegative integer within range\n", a.c_str());
            }
            o.memory_limit_mib = number;
        }
        else if (a == "--prompt-cache-tail") o.prompt_cache_tail = true;
        else if (a == "--prompt-cache-spread") o.prompt_cache_spread = true;
        else if (a == "--prompt-cache-every") o.prompt_cache_every = std::max(0LL, std::atoll(next("--prompt-cache-every")));
        else if (a == "--prompt-cache-root") o.prompt_cache_root = std::max(0LL, std::atoll(next("--prompt-cache-root")));
        else if (a == "--turn-token") o.turn_token = std::atoll(next("--turn-token"));
        else if (a == "--tail-role-token") o.tail_role_token = std::atoll(next("--tail-role-token"));
        else if (a == "--short-read") o.short_read = std::max(0LL, std::atoll(next("--short-read")));
        else if (a == "--prefill-head-alias") o.prefill_head_alias = true;
        else if (a == "--suffix-draft") o.suffix_draft = std::max(0, std::atoi(next("--suffix-draft")));
        else if (a == "--mtp-max-t") o.mtp_max_t = std::max(0, std::atoi(next("--mtp-max-t")));
        else if (a == "--lookup-chain") o.lookup_chain = std::clamp(std::atoi(next("--lookup-chain")), 0, 7);
        else if (a == "--lookup-chain-min") o.lookup_chain_min = std::max(3, std::atoi(next("--lookup-chain-min")));
        else if (a == "--mtp-hnorm") {
            const std::string v = next("--mtp-hnorm");
            if (v != "pooled" && v != "stream") {
                return fail(2, "strata generate: --mtp-hnorm takes pooled or stream\n");
            }
            o.mtp_hnorm_stream = v == "stream";
        }
        else if (a == "--mtp-q4") {
            o.mtp_q4 = next("--mtp-q4");
            if (o.mtp_q4 != "all" && o.mtp_q4 != "proj" && o.mtp_q4 != "head") {
                return fail(2, "strata generate: --mtp-q4 takes all, proj or head\n");
            }
        }
        else if (a == "--mtp-draft-vocab") o.mtp_draft_vocab = next("--mtp-draft-vocab");
        else if (a == "--no-spec-split") o.spec_split = false;
        else if (a == "--eos-ids") {
            std::string e;
            if (!parse_i64_list(next("--eos-ids"), o.eos_ids, e)) { return fail(2, "--eos-ids: %s\n", e.c_str()); }
        }
        else if (a == "--adapt-swaps") o.adapt_swaps = std::atoi(next("--adapt-swaps"));
        else if (a == "--expert-cache-cpu-order") o.expert_cache_cpu_order = true;
        else if (a == "--expert-cache-per-layer") o.expert_cache_per_layer = true;
        else if (a == "--no-hit-poke") o.no_hit_poke = true;
        else if (a == "--expert-profile") o.expert_profile = next("--expert-profile");
        else if (a == "--expert-profile-save") o.expert_profile_save = next("--expert-profile-save");
        else if (a == "--expert-profile-save-every")
            o.expert_profile_save_min = std::atof(next("--expert-profile-save-every"));
        else if (a == "--mmap-experts") o.mmap_experts = true;
        else if (a == "--shared-expert-arena") o.shared_expert_arena = next("--shared-expert-arena");
        else if (a == "--resident-cpu-experts") o.resident_cpu_experts = o.resident_cpu_explicit = true;
        else if (a == "--resident-experts") {
            o.mmap_experts = o.resident_cpu_experts = o.resident_pin = o.resident_soft = true;
            o.resident_headroom = 4ull << 30;
            // A/B arms: STRATA_RESIDENT_PIN=0 keeps the copy pageable (the --resident-cpu-experts form);
            // STRATA_RESIDENT_HEADROOM_GIB=N leaves N GiB of the available RAM free instead of 4
            if (const char* v = std::getenv("STRATA_RESIDENT_PIN"); v != nullptr && std::string(v) == "0")
                o.resident_pin = false;
            if (const char* v = std::getenv("STRATA_RESIDENT_HEADROOM_GIB"); v != nullptr && std::atof(v) >= 0.0)
                o.resident_headroom = (uint64_t) (std::atof(v) * 1073741824.0);
        }
        else if (a == "--resident-budget-gib") {
            const double gib = std::atof(next("--resident-budget-gib"));
            if (!(gib > 0.0)) { return fail(2, "strata generate: --resident-budget-gib needs N > 0\n"); }
            o.resident_budget = (uint64_t) (gib * 1073741824.0);
            o.mmap_experts = o.resident_cpu_experts = o.resident_pin = true;
            o.resident_headroom = 4ull << 30;
            if (const char* v = std::getenv("STRATA_RESIDENT_PIN"); v != nullptr && std::string(v) == "0")
                o.resident_pin = false;
            if (const char* v = std::getenv("STRATA_RESIDENT_HEADROOM_GIB"); v != nullptr && std::atof(v) >= 0.0)
                o.resident_headroom = (uint64_t) (std::atof(v) * 1073741824.0);
        }
        else if (a == "--shared-late") o.shared_late = true;
        else if (a == "--keep-canonical") o.keep_canonical = true;
        else if (a == "--no-token-graph") o.no_token_graph = true;
        else if (a == "--no-fused-gr") o.no_fused_gr = true;
        else if (a == "--no-fast-attn") o.no_fast_attn = true;
        else if (a == "--no-publish-kernel") o.no_publish_kernel = true;
        else if (a == "--no-fused-gdn") o.no_fused_gdn = true;
        else if (a == "--no-fast-select") o.no_fast_select = true;
        else {
            // An unknown flag is an ERROR and not a warning: a typo'd `--max-neww` that silently generated 16
            // tokens would look like a working run.
            fail(2, "unknown argument: %s\n", a.c_str());
            usage();
            return 2;
        }
        }
    }
    return 0;
}

int EngineState::start_up(int argc, char** argv) {
    // `strata --version` / `-V` (also `strata generate --version`): the version and nothing else, before any GPU is touched
    for (int i = 1; i < argc && i < 3; ++i) {
        if (std::strcmp(argv[i], "--version") == 0 || std::strcmp(argv[i], "-V") == 0) {
            std::printf("strata %s\n", STRATA_VERSION);
            nothing_to_serve = true;
            return 0;
        }
    }
    // Load every CUDA kernel when the context is created, before the expert cache takes the free VRAM.  With the
    // default lazy loading, a kernel first used mid-prompt (MMQ for IQ3_XXS at 64K+ on a 12 GB card) found no VRAM
    // left for its code and the engine ended ("out of memory: cudaFuncSetAttribute").  Costs ~30 MB of VRAM.
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) {
        setenv("CUDA_MODULE_LOADING", "EAGER", 0);
    }
    if (launch_blocking())
        std::fprintf(stderr, "warning: CUDA_LAUNCH_BLOCKING=1: the verify windows cannot run with blocking launches, so "
                             "the first one after a prompt hangs (#1341). Unset it; "
                             "STRATA_PF_STEP_SYNC=1 narrows down a failing prompt step without it\n");
    try {
        if (const int status = parse_options(argc, argv); status != 0 || nothing_to_serve) return status;
    } catch (const MissingValue& m) {
        return fail(2, "%s needs a value\n", m.option);
    }
    if (!o.serve) {
        fail(2, "strata generate: --serve is required (this engine only serves requests on stdin)\n");
        usage();
        return 2;
    }
    strata::core::set_coupled_draft(o.coupled_draft);
    {   // --host-core / STRATA_HOST_CORE, before the pool and the session pin any thread
        std::string hc = o.host_core;
        if (hc.empty())
            if (const char* e = std::getenv("STRATA_HOST_CORE")) hc = e;
        if (hc == "last") strata::kernels::cpu::set_host_core(strata::kernels::cpu::HostCore::Last);
        if (hc == "sibling") strata::kernels::cpu::set_host_core(strata::kernels::cpu::HostCore::Sibling);
    }
    {   // --aux-cpus / STRATA_AUX_CPUS: read the CPUs this process may use now, before any thread is pinned; the plan
        // follows once the pool exists (it needs the workers' CPUs)
        std::string ac = o.aux_cpus;
        if (ac.empty())
            if (const char* e = std::getenv("STRATA_AUX_CPUS")) ac = e;
        std::string aerr;
        if (!strata::aux_cpus::configure(ac, aerr)) {
            return fail(2, "strata generate: STRATA_AUX_CPUS: %s\n", aerr.c_str());
        }
    }
    {   // the RAM limit: --memory-limit-mib, else STRATA_MEMORY_LIMIT_MIB, else none
        int64_t mib = o.memory_limit_mib;
        if (mib < 0) {
            const char* env = std::getenv("STRATA_MEMORY_LIMIT_MIB");
            if (env != nullptr && env[0] != '\0') {
                const size_t len = std::strlen(env);
                const auto parsed = std::from_chars(env, env + len, mib);
                if (parsed.ec != std::errc{} || parsed.ptr != env + len || mib < 0 || mib > INT64_MAX / (1024 * 1024)) {
                    return fail(2, "STRATA_MEMORY_LIMIT_MIB needs a nonnegative integer within range (got \"%s\")\n", env);
                }
            } else mib = 0;
        }
        strata::core::set_memory_limit_bytes((uint64_t) mib << 20);
    }
    pcie_given = o.pcie_frac >= 0.0;
    if (o.mmap_experts && !o.shared_expert_arena.empty()) {
        return fail(2, "strata generate: --shared-expert-arena backs the resident arena and cannot be used with --mmap-experts\n");
    }
    if (o.resident_cpu_experts && (!o.mmap_experts || o.expert_profile.empty())) {
        return fail(2, "strata generate: --resident-cpu-experts requires --mmap-experts and a static --expert-profile\n");
    }
    if (o.prefill_auto && (o.no_prefill_borrow || o.expert_profile.empty())) {
        o.prefill_auto = false;       // nothing to lend from: the buffers are reserved for the session, so keep them small
        o.prefill_chunk = 2048;
    }
    // plan v0.3 P8: requests bring their own tokens
    o.tokens = {248045};
    o.max_new = 1;

    if ((o.ple_io != "direct" && o.ple_io != "mmap" && o.ple_io != "ram") || o.ple_row_cache < 0 || o.ple_inflight < 1 ||
        o.ple_inflight > 1024 || !(o.ple_delay_us >= 0)) {
        return fail(2, "strata generate: invalid --ple-io/--ple-row-cache/--ple-inflight/--ple-delay-us\n");
    }
    if (o.kv == "q4") o.kv = "q4_0";
    if (o.kv != "fp16" && o.kv != "int8" && o.kv != "q4_0" && o.kv != "k8v4") {
        return fail(2, "strata generate: --kv must be fp16, int8, q4_0 or k8v4\n");
    }
    strata::core::qsa_set_kv_int8(o.kv == "int8");
    strata::core::qsa_set_kv_q4(o.kv == "q4_0");   // PR #21: 4-bit codes after a Hadamard rotation (kv_q4.hpp)
    // STRATA_KV_ROT=1: INT8 K/V through the Hadamard rotation --kv q4_0 already uses. Opt-in: first-token KL to
    // fp16 K/V improved on an NVFP4 pack (0.0066 -> 0.0051) but not on IQ2_XS (0.0022 -> 0.0054)
    kv_rot = std::getenv("STRATA_KV_ROT");
    strata::core::qsa_set_kv_int8_rotate(kv_rot != nullptr && kv_rot[0] == '1');
    if (kv_rot != nullptr && kv_rot[0] == '1' && o.kv == "int8")
        std::fprintf(stderr, "strata generate: STRATA_KV_ROT=1: INT8 K/V through the Hadamard rotation (opt-in)\n");
    strata::core::qsa_set_kv_hybrid(o.kv == "k8v4");   // K8V4: INT8 K + rotated Q4_0 V, 816 B/cell
    if (o.kv_resident < 0) {
        return fail(2, "strata generate: --kv-resident must be >= 0\n");
    }
    strata::core::qsa_set_kv_resident(o.kv_resident);
    // Prompt lookup (the suffix drafter, on by default): the MTP keeps its --spec windows and a lookup window may be
    // up to 2 tokens longer; the draft policy (strata/spec/draft_policy.hpp) takes one only where it pays. Code
    // edits +6-11%, ordinary text unchanged (bench/results/2026-09-27-spec). --suffix-draft 0 turns it off.
    if (o.suffix_draft > 0 && o.spec >= 2 && o.mtp_max_t == 0) {
        o.mtp_max_t = o.spec;
        o.spec = std::min(o.spec + 2, 8);   // kVerifyMaxT
    }
    // --lookup-chain (opt-in): the MTP keeps its windows; a lookup chained after its drafts may lengthen one by up to K.
    // Sized from the cap actually in force: the suffix block above may already have raised --spec to --spec+2, so sizing
    // from mtp_max_t left K=1 and K=2 with a window identical to not passing the flag at all.
    if (o.lookup_chain > 0 && o.spec >= 2) {
        if (o.mtp_max_t == 0) o.mtp_max_t = o.spec;
        const int chain_base = o.spec;
        o.spec = std::max(chain_base, std::min(std::max(chain_base, o.mtp_max_t) + o.lookup_chain, 8));  // kVerifyMaxT
    }
    strata::core::layer_set_shared_early(!o.shared_late);
    if (!o.native_preset.empty()) {
        try {
            // every shard of the model (<name>-0000N-of-0000M.gguf beside --native).  A missing shard is an error
            // here: it used to be skipped, leaving a model with some tensors absent and a later error, or none.
            o.native_shards = strata::gguf_split_paths(o.native_preset);
            // a model family Strata recognises but has no engine path for yet (Qwen3.6-35B-A3B) is refused here,
            // with a message that says so, before anything else about the file is judged
            {
                const strata::GgufFile meta(o.native_shards.front());
                const std::string why = strata::model_family_not_supported(strata::model_family(meta));
                if (strata::model_family(meta) != strata::ModelFamily::Unknown && !why.empty()) {
                    return fail(2, "strata generate: --native %s: %s\n", o.native_preset.c_str(), why.c_str());
                }
            }
            // --ple-gguf defaults to the shard that holds the PLE table, found by name: shard 2 of the ISTA files
            // and of Unsloth's UD-Q4_K_XL, shard 1 of Swift's
            if (o.ple_gguf.empty() && !o.no_ple) {
                const strata::GgufModel model(o.native_shards);
                size_t at = 0;
                if (model.find("per_layer_token_embd.weight", &at) != nullptr) o.ple_gguf = o.native_shards[at];
            }
        } catch (const std::exception& e) {
            return fail(2, "strata generate: --native %s: %s\n", o.native_preset.c_str(), e.what());
        }
        if (o.no_ple || o.ple_gguf.empty()) {
            return fail(2, "strata generate: --native requires --ple-gguf (the PLE key is native too), and no "
                           "shard of the model holds per_layer_token_embd.weight\n");
        }
        o.stream_token = true;
        o.gr_native_mmvf = true;
        o.native_bf16 = o.native_bf16_extra = true;
        o.native_ple_key = o.native_moe_combine = o.native_gdn = o.native_router = true;
        o.native_qsa = o.native_qsa_indexer = o.native_rope = o.native_ple_postops = true;
        if (o.native_head_gguf.empty()) o.native_head_gguf = o.native_preset;
        if (o.native_dense_gguf.empty()) {
            // every shard of the model (<name>-0000N-of-0000M.gguf beside --native), then the PLE shard: a split
            // may put any layer in any shard (Swift's GGUFs: layers 13-47 in shard 2, the PLE table in shard 1)
            o.native_dense_gguf = o.native_shards;
            // a PLE-only table (tools/ple_fp8_pack.py: architecture strata-ple) holds no projections
            bool ple_only = false;
            try {
                strata::GgufFile pg(o.ple_gguf);
                if (const strata::MetaValue* v = pg.get("general.architecture")) ple_only = v->s == "strata-ple";
            } catch (const std::exception&) {}
            if (!ple_only &&
                std::find(o.native_dense_gguf.begin(), o.native_dense_gguf.end(), o.ple_gguf) == o.native_dense_gguf.end())
                o.native_dense_gguf.push_back(o.ple_gguf);
        }
        // Plan v0.3 (24 Sep): the CPU experts stay on the VNNI kernel.  The llama.cpp-CPU-exact q8_0 contract
        // cost 27.0 vs 17.2 ms/token of pool time and G-C does not need it; `--cpu-oracle-q8-0` still selects it.
    }
    if (o.no_ple && !o.ple_gguf.empty()) {
        return fail(2, "strata generate: --no-ple and --ple-gguf are mutually exclusive\n");
    }
    if (o.native_ple_postops && o.no_ple) {
        return fail(2, "strata generate: --native-ple-postops requires PLE enabled\n");
    }
    if (!o.no_ple && o.ple_gguf.empty()) {
        return fail(2, "strata generate: --ple-gguf is required; --no-ple explicitly enables a diagnostic ablation\n");
    }
    // P7 audit: positions, cells and pooled-block indices are cast to int32 on the device path.
    if (o.max_context > 2147483647LL - 8) {
        return fail(2, "strata generate: --max-context must be below 2^31\n");
    }
    if (o.max_new <= 0 || o.max_context <= 0 || o.max_new > o.max_context ||
        o.tokens.size() > (size_t) (o.max_context - o.max_new)) {
        return fail(2, "strata generate: positive --max-new and --max-context must fit the prompt and generation\n");
    }
    // THE ROPE KNOBS (rope_scaling.hpp).  Anything invalid dies here, at second zero, rather than becoming a
    // NaN angle inside one of the twelve QSA layers.  Only the RANGES are checked - the config itself is
    // resolved after the model file has had its say, right before session_init.
    {
        using RST = strata::kernels::RopeScalingType;
        // an absent --rope-scaling (the empty default) leaves the type to the model file's rope keys,
        // resolved below; anything present must be one of the three names
        if (o.rope_scaling == "none") rope_cfg.type = RST::None;
        else if (o.rope_scaling == "linear") rope_cfg.type = RST::Linear;
        else if (o.rope_scaling == "yarn") rope_cfg.type = RST::YaRN;
        else if (!o.rope_scaling.empty()) {
            return fail(2, "strata generate: --rope-scaling must be none, linear or yarn (got '%s')\n",
                        o.rope_scaling.c_str());
        }
        // every knob FINITE first: `atof("nan")` is NaN, and a NaN passes every range comparison below
        for (const double v : {o.rope_scale, o.rope_freq_base, o.rope_freq_scale, o.yarn_orig_ctx, o.yarn_ext_factor,
                               o.yarn_attn_factor, o.yarn_beta_fast, o.yarn_beta_slow})
            if (!std::isfinite(v)) {
                return fail(2, "strata generate: a rope scaling knob is not a finite number (%g)\n", v);
            }
        // 0 is the absent default; an explicit factor must extend, not shrink
        if (o.rope_scale != 0 && o.rope_scale < 1.0) {
            return fail(2, "strata generate: --rope-scale %g must be >= 1 (it extends the context, not shrinks it)\n",
                        o.rope_scale);
        }
        if (o.rope_freq_base != 0 && o.rope_freq_base <= 1.0) {
            return fail(2, "strata generate: --rope-freq-base must be a base above 1 (0 = the model's)\n");
        }
        if (o.rope_freq_scale < 0 || o.yarn_orig_ctx < 0 || o.yarn_ext_factor < -1.0 || o.yarn_attn_factor <= 0 ||
            o.yarn_beta_fast <= 0 || o.yarn_beta_slow <= 0) {
            return fail(2, "strata generate: invalid rope scaling knob (see usage: --yarn-ext-factor <0 = auto, "
                           "--yarn-orig-ctx 0 = default, the rest positive)\n");
        }
    }
    if (o.expert_cache < -1 || o.pool_workers < 0) {
        return fail(2, "strata generate: invalid sampling or resource parameter\n");
    }

    if (o.native_flash_attn_short && o.max_context > 256) {
        return fail(2, "strata generate: --native-flash-attn-short requires --max-context <=256\n");
    }
    if (!o.native_head_gguf.empty()) {
        try {
            o.native_head_shards = o.native_head_gguf == o.native_preset ? o.native_shards
                                                                         : strata::gguf_split_paths(o.native_head_gguf);
        } catch (const std::exception& e) {
            return fail(2, "strata generate: --native-head-gguf %s: %s\n", o.native_head_gguf.c_str(), e.what());
        }
    }
    if (o.native_ple_key && (o.native_dense_gguf.empty() || o.no_ple)) {
        return fail(2, "strata generate: --native-ple-key requires PLE and --native-dense-gguf\n");
    }
    if (o.cpu_oracle_q8_0 && (o.expert_cache != 0 || !o.expert_profile.empty())) {
        return fail(2, "strata generate: --cpu-oracle-q8-0 cannot be combined with --expert-cache or --expert-profile until the GPU expert contract matches\n");
    }

    // **BEFORE ANYTHING ELSE.**  The CPU expert kernel is AVX-512 (VNNI + VBMI) and its translation unit is
    // compiled `/arch:AVX512`, so on a CPU without those features it does not fail - it executes an illegal
    // instruction at some unpredictable token.  Refusing at second zero is the whole point of P2.S3's check.
    strata::kernels::cpu::expert_set_oracle_q8_0(o.cpu_oracle_q8_0);
    // ... and nothing runs on a CPU without AVX2: every CPU expert kernel is AVX2 at least (the AVX-512 ones are
    // chosen above it), and so is ggml-cpu in the release build, which the native pack's layout load initializes
    // next.  Refused here, by name, rather than an illegal instruction in the first expert.
    // The experimental older-CPU build (STRATA_ISA_FLOOR=avx|none, compiled on that PC; #394 #595 #623) has ggml-cpu
    // for that floor, so a native pack's experts run there on ggml-cpu (every AVX2 kernel is behind cpu_avx2_ok).
    isa_floor = strata::kernels::cpu::isa_floor_build();
    if (!strata::kernels::cpu::cpu_avx2_ok() && isa_floor[0] == '\0') {
        return fail(2, "strata generate: this CPU (%s) does not support AVX2 with FMA and F16C, which this engine's "
                       "CPU expert kernels need; Strata runs on Intel Haswell (2013), AMD Zen (2017) or newer. "
                       "Older CPUs are EXPERIMENTAL and slow: setup compiles an engine for them on this PC "
                       "(STRATA_ISA_FLOOR, see \"Older CPUs\" in docs/INSTALL.md)\n",
                    strata::kernels::cpu::cpu_name().c_str());
    }
    if (isa_floor[0] != '\0') {
        const bool avx_floor = std::strcmp(isa_floor, "avx") == 0;
        if (avx_floor ? !strata::kernels::cpu::cpu_avx1_ok() : !strata::kernels::cpu::cpu_sse42_ok()) {
            return fail(2, "strata generate: this engine is the older-CPU build for %s, which this CPU (%s) does "
                           "not have%s\n", avx_floor ? "AVX" : "SSE4.2 with POPCNT",
                        strata::kernels::cpu::cpu_name().c_str(),
                        avx_floor ? "; compile it with STRATA_ISA_FLOOR=none" : "");
        }
        if (!strata::kernels::cpu::cpu_avx2_ok())
            std::fprintf(stderr, "strata generate: EXPERIMENTAL older-CPU build (ggml-cpu for %s): this CPU has no AVX2, "
                                 "so the CPU's experts run on ggml-cpu's kernels - expect it to be slow\n", isa_floor);
        else
            std::fprintf(stderr, "strata generate: this is the older-CPU build (ggml-cpu for %s); this CPU has AVX2, "
                                 "and the normal build is faster on it\n", isa_floor);
    }

    if (!o.native_head_gguf.empty() && !o.stream_token) {
        return fail(2, "--native-head-gguf requires --stream-token\n");
    }
    // Plan v0.3 P6: where the experts live.  A native pack (tools/iq_pack.py: the IQ2_XS / IQ3_XXS files) keeps
    // every quantized tensor in its GGUF form, so it needs --native (the dense projections, head and embedding
    // come from the model file) and runs its experts in verify windows only (--spec).
    {
        const strata::core::ModelGeometry g0;
        if (!strata::kernels::cpu::expert_layout_load(o.pack, g0.n_layers, g0.n_expert, err)) {
            return fail(1, "strata generate: %s\n", err.c_str());
        }
        // Every layer's formats must have GPU expert kernels and a prompt-path dequantizer, checked here, before
        // anything is allocated: an unsupported down type used to exit from inside the first verify window, and
        // an unsupported dequant type left the prompt path's fp16 buffer unwritten.
        const auto& lay = strata::kernels::cpu::expert_layout();
        for (int64_t l = 0; lay.native && l < (int64_t) lay.fmt.size(); ++l) {
            const auto& f = lay.fmt[(size_t) l];
            if (!strata::kernels::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff)) {
                return fail(1, "strata generate: layer %lld's experts are %s/%s (ggml types %d/%d), which this "
                               "engine has no GPU kernels for\n", (long long) l,
                            strata::ggml_type_name((uint32_t) f.gu_type), strata::ggml_type_name((uint32_t) f.d_type),
                            f.gu_type, f.d_type);
            }
#if defined(STRATA_PREFILL_MMQ)
            // MODEL-UPDATE Phase 1: the K-quants' prompt kernels (STRATA_MMQ_KQ=1) are a library of their own on CUDA
            // (strata_kq): asking for the formats opens it now, before the expert cache takes the free VRAM
            (void) strata::prefill::mmq::supported(f.gu_type);
            (void) strata::prefill::mmq::supported(f.d_type);
#endif
        }
    }
    native_pack = strata::kernels::cpu::expert_layout().native;
    // STRATA_MMQ_DIRECT: the room each sized expert slot keeps after its blob (0 unless it is on: the slot costs and
    // the arena are then today's, byte for byte)
    slot_guard = native_pack ? (int64_t) strata::prefill::Prefill::slot_guard_bytes() : 0;
    // plan v0.3 P6: the PCIe share of the missed experts, measured per kind of pack (the paper, finding on PCIe).
    // PR #44: a x8 link carries half of what the native default assumes - the GPU's SMs read that share over the
    // link (the copy kernel, since 0.1.14), so on a slower link it must shrink or the window waits for it.  The
    // real H2D bandwidth is probed once; from 20 GB/s up (x16 PCIe 4/5) the measured default stays.  The canonical
    // pack keeps 0.2 as its start value (0.1.43: PcieAuto refines it on CUDA).  `--calibrate` measures it outright.
    if (o.pcie_frac < 0.0) {
        const double base = native_pack ? 0.55 : 0.2;
        std::string bursts;
        const char* pfd = std::getenv("STRATA_PCIE_FRAC_DEFAULT");
        const bool pf_old = pfd != nullptr && std::strcmp(pfd, "old") == 0;
        const bool pf_old_canon = pf_old;
        // the canonical pack keeps its 0.2 as the START value, but is probed too (CUDA, unless
        // STRATA_PCIE_FRAC_DEFAULT=old) so that PcieAuto can refine the share from the CPU pool's speed
        const double bw = (native_pack || !pf_old_canon) ? probe_pcie_h2d_gbps(&bursts) : -1.0;
        if (!native_pack && !(bw > 0.0)) {
            o.pcie_frac = base;
        } else if (!native_pack) {
            o.pcie_frac = base;
            std::fprintf(stderr, "strata generate: PCIe probe: %.1f GB/s host->device (best of %s); the canonical pack "
                                 "starts at pcie_frac %.2f and refines it from the CPU pool's speed\n", bw, bursts.c_str(), base);
            o.pcie_frac_auto = true;
            o.pcie_probe_gbps = bw;
        } else if (bw > 0.0) {
            o.pcie_frac = pcie_frac_for_gbps(bw, base);
            std::fprintf(stderr, "strata generate: PCIe probe: %.1f GB/s host->device (best of %s) -> pcie_frac %.2f "
                                 "(default %.2f)\n", bw, bursts.c_str(), o.pcie_frac, base);
            // STRATA_PCIE_FRAC_DEFAULT=old keeps the start-up rule everywhere.
            o.pcie_frac_auto = !pf_old;
            o.pcie_probe_gbps = bw;
        } else {
            o.pcie_frac = base;
            std::fprintf(stderr, "strata generate: PCIe probe failed -> pcie_frac default %.2f\n", base);
        }
    }
    // STRATA_PCIE_BALANCE=1 (opt-in): each verify layer's PCIe count from measured costs (Verifier::set_pcie_balance)
    // instead of the probe's fixed share above - the CPU's speed sets the right share as much as the link's.  IQ2_XS,
    // one AVX2 pool worker (emulated on an RTX 5090 + 9950X3D): 16.8-17.2 -> 12.8-13.0 ms a round; the 9950X3D's own
    // 16 workers: 12.2 either way.  One GPU, no batch slots, no --pcie-frac.
    pcie_balance = [&]() {
        const char* e = std::getenv("STRATA_PCIE_BALANCE");
        return !pcie_given && native_pack && e != nullptr && std::strcmp(e, "1") == 0;
    };
    // the canonical Q2_0 pack's CPU kernels are AVX-512 only; a native pack runs on AVX2 CPUs as well
    if (!native_pack) {
        // #2347: an NVFP4 / i-quant / Unsloth pack IS a native pack and runs on an AVX2 CPU (its rows have AVX2
        // kernels).  This refusal is the canonical Q2_0 pack's, reached when the --pack folder has no
        // native_experts.txt - a hand-written config pointing --pack at the wrong folder reads as "this CPU is too
        // old".  Say so before the refusal, and where the file is when it sits one level down (a pack/ subfolder).
        if (!strata::kernels::cpu::cpu_features().usable()) {
            std::error_code ec;
            const std::filesystem::path pk(o.pack);
            std::fprintf(stderr, "strata generate: %s has no native_experts.txt, so it is read as the canonical Q2_0 "
                                 "pack, whose CPU kernel needs AVX-512.\n"
                                 "        An NVFP4, i-quant or Unsloth pack has that file in the folder given as --pack "
                                 "and runs on an AVX2 CPU as well;\n",
                         o.pack.c_str());
            if (std::filesystem::exists(pk / "pack" / "native_experts.txt", ec))
                std::fprintf(stderr, "        it is in %s: use that folder as --pack.\n", (pk / "pack").generic_string().c_str());
            else
                std::fprintf(stderr, "        if this is one of those packs, the folder is incomplete or the wrong one "
                                     "(setup writes it; run setup again).\n");
        }
        strata::kernels::cpu::cpu_require_expert_support();
    } else if (!strata::kernels::cpu::cpu_avx512_ok())
        std::fprintf(stderr, "strata generate: this CPU has no AVX-512: the expert kernels run on %s "
                             "(multi-token for the i-quant gate/up rows)\n",
                     !strata::kernels::cpu::cpu_avx2_ok()
                         ? (strata::kernels::cpu::cpu_avx1_ok() && std::getenv("STRATA_NO_IQ128") == nullptr
                            ? "AVX1 128-bit (iq_avx1, the older-CPU build)"
                            : "ggml-cpu vec_dot (no AVX2: the older-CPU build)")
                     : std::getenv("STRATA_NO_IQ256") == nullptr ? "AVX-2" : "ggml-cpu vec_dot (STRATA_NO_IQ256 set)");
    // the compiled shape; a native model's file replaces it below with its own metadata (read_geometry), which a
    // pruned variant (GSQ-RCO Coder, fewer than 512 experts) differs from
    K = g.n_expert_used;
    // THE ROPE CONFIG RESOLVES HERE, BEFORE ANY WEIGHT MOVES - the CLI and the model file have both spoken,
    // and `session_init` below builds the rope table from it and captures the kernels reading its constants
    // (rope_scaling.hpp); the only hard constraint is "set before that", and dying on a bad rope key beats
    // scanning gigabytes of shards first.  Precedence: an EXPLICIT flag over the model file's rope keys over
    // the struct defaults.  The empty --rope-scaling and the 0 --rope-scale mean the flag is absent, so the
    // model file decides; an explicit value - `none` and `1` included, the opt-outs - wins over the model file.
    {
        // The model file's rope keys (llama.cpp's names under the arch prefix), when it carries any - the
        // artifact today ships none, so this is a no-op defaults channel for future fine-tunes.
        std::string gguf_rope_type;
        double gguf_rope_base = 0, gguf_rope_factor = 0, gguf_rope_orig_ctx = 0;
        if (!o.native_preset.empty()) {
            // The geometry is the model file's (core::geometry_from_gguf: every "<arch>.*" shape key), checked
            // against the shapes this engine's kernels are built for.  The expert count and top-k are run-time
            // sizes: a pruned variant (GSQ-RCO Coder) ships fewer experts than the canonical 512x10.
            try {
                strata::GgufFile model_gguf(o.native_shards.front());   // the metadata shard
                std::string gerr;
                if (!strata::core::geometry_from_gguf(model_gguf, g, K, gerr)) {
                    return fail(1, "strata generate: %s\n", gerr.c_str());
                }
                const std::string mismatch = strata::core::compiled_geometry_mismatch(g);
                if (!mismatch.empty()) {
                    return fail(1, "strata generate: %s: %s\n", o.native_preset.c_str(), mismatch.c_str());
                }
                const std::string p = std::string(strata::model_family_arch(g.family)) + ".";
                if (const strata::MetaValue* v = model_gguf.get(p + "rope.freq_base")) gguf_rope_base = v->num();
                if (const strata::MetaValue* v = model_gguf.get(p + "rope.scaling.type")) gguf_rope_type = v->s;
                if (const strata::MetaValue* v = model_gguf.get(p + "rope.scaling.factor")) gguf_rope_factor = v->num();
                if (const strata::MetaValue* v = model_gguf.get(p + "rope.scaling.original_context_length"))
                    gguf_rope_orig_ctx = v->num();
            } catch (const std::exception& e) {
                return fail(1, "strata generate: reading the model's geometry from %s: %s\n",
                            o.native_preset.c_str(), e.what());
            }
        }
        using RST = strata::kernels::RopeScalingType;
        if (!o.rope_scaling.empty()) {
            // the early validation pinned the spelling; `none` here is the CLI opting OUT of the model file's keys
            if (o.rope_scaling == "linear") rope_cfg.type = RST::Linear;
            else if (o.rope_scaling == "yarn") rope_cfg.type = RST::YaRN;
            else rope_cfg.type = RST::None;
        } else if (!gguf_rope_type.empty()) {
            if (gguf_rope_type == "linear") rope_cfg.type = RST::Linear;
            else if (gguf_rope_type == "yarn") rope_cfg.type = RST::YaRN;
            else if (gguf_rope_type != "none") {
                return fail(2, "strata generate: %s carries rope.scaling.type '%s' - none, linear or yarn only\n",
                            o.native_preset.c_str(), gguf_rope_type.c_str());
            }
        }
        if (o.rope_scale > 0) rope_cfg.factor = o.rope_scale;            // an explicit factor, 1 included
        else if (gguf_rope_factor > 1.0) rope_cfg.factor = gguf_rope_factor;
        if (o.rope_freq_base > 0) rope_cfg.freq_base = o.rope_freq_base;
        else if (gguf_rope_base > 1.0) rope_cfg.freq_base = gguf_rope_base;
        if (o.yarn_orig_ctx > 0) rope_cfg.orig_ctx = o.yarn_orig_ctx;
        else if (gguf_rope_orig_ctx >= 1) rope_cfg.orig_ctx = gguf_rope_orig_ctx;
        rope_cfg.freq_scale_in = o.rope_freq_scale;
        rope_cfg.ext_factor = o.yarn_ext_factor >= 0 ? o.yarn_ext_factor
                                                     : (rope_cfg.type == RST::YaRN ? 1.0 : 0.0);
        rope_cfg.attn_factor = o.yarn_attn_factor;
        rope_cfg.beta_fast = o.yarn_beta_fast;
        rope_cfg.beta_slow = o.yarn_beta_slow;
        if (rope_cfg.type == RST::None) {
            // none is the trained rotation, exactly: the scaling knobs are inert (the table builder and
            // `kernel_args` ignore them), and resetting them keeps the logged/queried config honest.  Only the
            // frequency base survives - it is the rotation itself, not a scaling knob.
            const bool knobs = o.rope_scale > 1.0 || o.rope_freq_scale > 0 || o.yarn_ext_factor > 0 ||
                               o.yarn_attn_factor != 1.0;
            const double base = rope_cfg.freq_base;
            rope_cfg = strata::kernels::RopeScaling{};
            rope_cfg.freq_base = base;
            if (knobs)
                std::fprintf(stderr, "strata generate: note: no rope scaling is active (none), so --rope-scale, "
                                     "--rope-freq-scale and the --yarn-* knobs have no effect\n");
        }
        // THE RESOLVED CONFIG IS VALIDATED AS A WHOLE, with the one rule every rotation site also applies
        // (rope_scaling.hpp): the CLI ranges above cannot see a model-file value, nor a combination such as a
        // --rope-freq-scale that turns the resolved factor non-finite.
        if (const char* why = strata::kernels::rope_scaling_invalid(rope_cfg)) {
            return fail(2, "strata generate: invalid rope scaling configuration: %s (type %s, factor %g, "
                           "freq_scale %g, base %g, original context %g)\n",
                        why, rope_cfg.type == RST::YaRN ? "yarn" : rope_cfg.type == RST::Linear ? "linear" : "none",
                        rope_cfg.factor, rope_cfg.freq_scale(), rope_cfg.freq_base, rope_cfg.orig_ctx);
        }
        strata::kernels::rope_scaling_set(rope_cfg);
        if (rope_cfg.type != RST::None) {
            const char* tn = rope_cfg.type == RST::YaRN ? "yarn" : "linear";
            std::fprintf(stderr,
                         "strata generate: rope scaling %s, factor %.6g (freq_scale %.6g, base %.6g, mscale %.6f), "
                         "--max-context %lld against a trained context of %.0f\n",
                         tn, rope_cfg.factor, rope_cfg.freq_scale(), rope_cfg.freq_base, rope_cfg.mscale(),
                         (long long) o.max_context, rope_cfg.orig_ctx);
            if ((double) o.max_context <= rope_cfg.orig_ctx)
                std::fprintf(stderr,
                             "strata generate: note: the context is within the trained %.0f - no position needs the "
                             "extension, and the resolved scaling still applies to every angle\n",
                             rope_cfg.orig_ctx);
            // only when there IS a magnitude correction: YaRN's log term (ext_factor != 0) or an explicit
            // --yarn-attn-factor; plain linear (mscale 1) has none, and saying otherwise was TODO 22
            if (rope_cfg.mscale() != 1.0)
                std::fprintf(stderr,
                             "strata generate: note: the %s magnitude correction scales cos and sin by %.6f "
                             "at every position\n",
                             rope_cfg.type == RST::YaRN ? "YaRN" : "--yarn-attn-factor", rope_cfg.mscale());
        }
    }
    if (native_pack) {
        if (o.native_preset.empty() || o.spec < 2 || o.keep_canonical ||
            (o.prefill_chunk <= 0 && o.tokens.size() > 1)) {
            return fail(2, "strata generate: %s is a native (IQ) pack: it needs --native SHARD1, --spec T (T >= 2) "
                           "and --prefill CHUNK\n", o.pack.c_str());
        }
        const auto embed_t0 = std::chrono::steady_clock::now();
        if (!native_embed.load(o.embd_gguf.empty() ? o.native_shards : std::vector<std::string>{o.embd_gguf}, g.n_embd,
                               g.n_vocab, err)) {
            return fail(1, "strata generate: %s\n", err.c_str());
        }
        strata::core::set_native_embed(&native_embed);
        std::fprintf(stderr, "strata generate: native pack: %s experts (largest blob %.2f MB), token embedding "
                             "%s in mapped host memory (%.0f MiB, %.1f s)\n",
                     o.pack.c_str(), (double) strata::kernels::cpu::expert_layout().max_blob / 1e6,
                     strata::ggml_type_name((uint32_t) native_embed.type()), (double) native_embed.bytes() / 1048576.0,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - embed_t0).count());
    }
    // Plan v0.3 P1: tensors served in native form are not also loaded in canonical form (~2.7 GB of VRAM back
    // to the expert cache with --native).  `--keep-canonical` loads both, as before.
    if (!o.keep_canonical) {
        if (!o.native_dense_gguf.empty() &&
            !strata::core::NativeDense::served_names(o.native_dense_gguf, o.native_ple_key, skip, err)) {
            return fail(1, "strata generate: %s\n", err.c_str());
        }
        if (!o.native_head_gguf.empty()) skip.insert("output.weight");
        // the PLE module validates its canonical key at construction (8 MB); a native pack has none to load
        if (!native_pack) skip.erase("blk.1.ple_key.weight");
        // #326: a --compat-bf16 pack (OrcaRouter IQ3_XXS) keeps its BF16 key in the arena
        if (native_pack && !strata::core::NativeDense::keep_unquantized_ple_key(o.pack, skip, err)) {
            return fail(1, "strata generate: %s\n", err.c_str());
        }
        if (native_pack) skip.insert("token_embd.weight");
        // STRATA_HC_Q8=1 (single GPU): the layers' hyper-connection down / up projections live only as Q8_0 (NativeDense
        // quantizes them from the pack file), so their bf16 copy is not loaded
        if (!o.native_dense_gguf.empty() && strata::core::NativeDense::hc_q8_requested())
            strata::core::NativeDense::add_hc_skip(o.pack, skip);
    }
    // STRATA_HC_REQ8=1 (opt-in, changes the output): the hyper-connection projections are not loaded as BF16;
    // load_hc_q8 below requantizes them to int8 + an fp32 scale per 32 values (~0.56x the bytes), and the freed VRAM
    // goes to the expert cache like any other.  (Upstream's STRATA_HC_Q8 is a different thing: the GGUF's Q8_0 bytes.)
    hc_req8_env = std::getenv("STRATA_HC_REQ8") != nullptr && std::atoi(std::getenv("STRATA_HC_REQ8")) != 0;
    hc_q8 = hc_req8_env;
    if (hc_q8) strata::core::WeightTable::hc_q8_names(g.n_layers, skip);
    if (!strata::core::WeightTable::pool_bytes(o.pack, pool_bytes, err, skip.empty() ? nullptr : &skip)) {
        return fail(1, "strata generate: %s\n", err.c_str());
    }
    if (const cudaError_t ce = cudaMalloc(&arena, pool_bytes); ce != cudaSuccess) {
        // #486: the arena is the first large allocation and its size does not depend on the context, so what is
        // missing is held by something else: say how much was free
        cudaGetLastError();
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        return fail(1, "strata generate: cudaMalloc(%llu) for the weight arena failed (%s): %llu MiB of %llu "
                       "MiB VRAM free on this GPU. The arena is allocated first, before the KV and expert "
                       "caches: another program (or an engine that is still exiting) holds the rest - "
                       "nvidia-smi / rocm-smi lists them\n",
                    (unsigned long long) pool_bytes, cudaGetErrorString(ce), (unsigned long long) (free_b >> 20),
                    (unsigned long long) (total_b >> 20));
    }
    mem_mark("the CUDA context and the weight arena");
    load_t0 = std::chrono::steady_clock::now();
    auto load_s = [&] {
        const auto t = std::chrono::steady_clock::now();
        const double s = std::chrono::duration<double>(t - load_t0).count();
        load_t0 = t;
        return s;
    };
    if (!wt.load(o.pack, arena, pool_bytes, err, skip.empty() ? nullptr : &skip)) {
        return fail(1, "strata generate: %s\n", err.c_str());
    }
    if (hc_q8) {
        uint64_t qb = 0;
        if (!wt.load_hc_q8(o.pack, qb, err)) { return fail(1, "strata generate: %s\n", err.c_str()); }
        std::fprintf(stderr, "strata generate: STRATA_HC_REQ8: hyper-connection projections as int8 + fp32/32 scales, "
                             "%llu MiB\n", (unsigned long long) (qb >> 20));
    }
    std::fprintf(stderr, "strata generate: %llu MiB of weights loaded from %s in %.1f s (%zu canonical tensors "
                         "skipped: served natively)\n",
                 (unsigned long long) (pool_bytes >> 20), o.pack.c_str(), load_s(), skip.size());
    mem_mark("the pack weights");

    if (!o.native_dense_gguf.empty()) {
        if (!native_dense.load(o.native_dense_gguf, wt, err, o.native_ple_key)) {
            return fail(1, "strata generate: native dense projections: %s\n", err.c_str());
        }
        std::fprintf(stderr, "strata generate: %zu native projection matrices, %.2f MiB of weights, in %.1f s\n",
                     native_dense.tensor_count(), (double) native_dense.weight_bytes() / (1024.0 * 1024.0), load_s());
    }

    strata::kernels::gr_set_fp32_activations(o.gr_fp32_activations);
    strata::kernels::gr_set_native_mmvf(o.gr_native_mmvf);
    // Plan v0.3 P3: the fused hyper-connection read rides the native (FP32-activation) contract.
    strata::core::layer_set_fast_attn(!o.no_fast_attn);
    strata::core::layer_set_publish_kernel(!o.no_publish_kernel);
    strata::core::layer_set_fused_gdn(!o.no_fused_gdn);
    strata::core::layer_set_fast_select(!o.no_fast_select);
    strata::core::layer_set_fused_gr(o.gr_native_mmvf && !o.no_fused_gr);
    strata::core::layer_set_native_bf16(o.native_bf16);
    strata::core::layer_set_native_flash_attn_short(o.native_flash_attn_short);
    strata::kernels::ple_set_native_bf16(o.native_bf16_extra);
    strata::kernels::shared_expert_set_native_bf16(o.native_bf16_extra);
    strata::kernels::native_moe_combine_set_enabled(o.native_moe_combine);
    strata::kernels::native_gdn_set_enabled(o.native_gdn);
    strata::kernels::native_router_set_enabled(o.native_router);
    strata::kernels::native_qsa_set_enabled(o.native_qsa);
    strata::kernels::native_qsa_indexer_set_enabled(o.native_qsa_indexer);
    strata::kernels::native_rope_set_enabled(o.native_rope);
    // The vision path: every rope kernel reads a cell's (t, h, w) from this table (strata/kernels/mrope.hpp).  It is
    // the identity until an image request, and it is set here, before any CUDA graph captures a rope kernel.
    if (o.vision) {
        const int64_t cells = o.max_context + 64;
        mrope_host.resize((size_t) cells * 3);
        for (int64_t c = 0; c < cells; ++c)
            mrope_host[(size_t) c * 3] = mrope_host[(size_t) c * 3 + 1] = mrope_host[(size_t) c * 3 + 2] = (int32_t) c;
        if (cudaMalloc(&d_mrope, mrope_host.size() * sizeof(int32_t)) != cudaSuccess ||
            cudaMemcpy(d_mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice) !=
                cudaSuccess) {
            return fail(1, "strata generate: cannot allocate the image position table\n");
        }
        strata::kernels::mrope_table_set(d_mrope);
    }
    strata::kernels::ple_set_native_postops(o.native_ple_postops);
    if (o.max_context < (int64_t) o.tokens.size() + o.max_new) {
        return fail(2, "strata generate: --max-context %lld cannot hold %zu prompt + %lld new tokens\n",
                    (long long) o.max_context, o.tokens.size(), (long long) o.max_new);
    }

    // **THE ENGINE RAN ON THE LEGACY DEFAULT STREAM, WHICH ON WDDM IS THE SLOW PATH.**  All four session
    // calls - `session_capture`, `session_replay`, `session_token` and `session_loop` - were handed `nullptr`,
    // i.e. stream 0.  `bench/micro/kernel_costs.cu` measures what that costs: EVERY kernel it launches through
    // a wrapper comes back at 28-31 us REGARDLESS OF SIZE, `scale_inplace` on 2,048 floats and `silu_inplace`
    // on 10,240 floats being indistinguishable, which is a fixed per-launch cost and not execution.
    // `bench/micro/graph_node_cost.cu` measures the same kernels on a real stream at 3.63 us ungrapped and
    // 0.805 us inside a graph.  **That is an ~8x penalty on every launch in the engine.**
    if (cudaStreamCreateWithFlags(&main_stream, cudaStreamNonBlocking) != cudaSuccess) {
        return fail(1, "strata generate: cannot create the main stream\n");
    }
    main_cs = (void*) main_stream;

    if (strata::core::doorbell_init(g, K, db) == 0) {
        return fail(1, "strata generate: doorbell_init failed\n");
    }
    ss.db = &db;

    // ================================ THE PLE ================================
    //
    // **ITS ABSENCE IS WHY GATE C1 FAILED** (LEDGER L123): layer 1 carries six `blk.1.ple_*` tensors, the whole
    // module was built and parity-tested, and nothing called it.  Everything below is construction - the table
    // is a mapping of the ORIGINAL second GGUF shard, the six weights are already loaded in the arena, and the
    // three buffers are the only allocation.
    ple_emb_host = std::vector<float>((size_t) strata::kernels::NG_N_EMBD);
    if (!o.ple_gguf.empty()) {
        strata::kernels::PleIoOptions pio;
        pio.mode = o.ple_io == "mmap" || o.ple_io == "ram" ? strata::kernels::PleIo::Mmap : strata::kernels::PleIo::Direct;
        pio.lock = o.ple_io == "ram";
        const auto tpl = Clock::now();
        pio.max_inflight = (uint32_t) o.ple_inflight;
        pio.cache_rows = (uint64_t) o.ple_row_cache;
        pio.io_thread = !o.ple_sync_submit;
        // Keep the SSD awake while rows are being asked for (PleReader::set_keepalive): some SSDs stall the first
        // reads 50-150 ms after ~250 ms without a command.  STRATA_SSD_KEEPALIVE = ms without a read before one
        // page is read anyway (default 100, 0 = off), STRATA_SSD_KEEPALIVE_WINDOW = seconds after the last row
        // request that this goes on (default 60; then the SSD may sleep until the next request).
        {
            const char* ka = std::getenv("STRATA_SSD_KEEPALIVE");
            const char* kw = std::getenv("STRATA_SSD_KEEPALIVE_WINDOW");
            pio.keepalive_ms = ka != nullptr && *ka ? std::clamp(std::atof(ka), 0.0, 10000.0) : 100.0;
            pio.keepalive_window_s = kw != nullptr && *kw ? std::clamp(std::atof(kw), 1.0, 86400.0) : 60.0;
            if (pio.mode != strata::kernels::PleIo::Direct || !pio.io_thread) pio.keepalive_ms = 0;
        }
        if (!ple_table.open(o.ple_gguf, err, pio)) {
            return fail(1, "strata generate: %s\n", err.c_str());
        }
        // Recommend, never force: when the default direct reads measure slow, switch to the mapped reads (40-100x
        // faster on the reporters' drives); an explicit --ple-io direct is kept and only warned about.
        if (pio.mode == strata::kernels::PleIo::Direct && std::getenv("STRATA_PLE_PROBE") == nullptr) {
            constexpr double kSlowRowsPerS = 15000.0;   // ~60 MB/s of 4 KiB pages; a healthy NVMe is >100k
            const double rps = ple_direct_rows_per_s(ple_table);
            if (rps >= kSlowRowsPerS)
                std::fprintf(stderr, "strata generate: PLE direct read probe: %.0f rows/s (fine)\n", rps);
            if (rps > 0 && rps < kSlowRowsPerS) {
                // #1984 #1971 #2040: on no platform is the switch made on its own.  Windows: the mapped reads stalled prompts (the first chunk waits for 16 serial page faults
                // a token from a cold table with no progress mark).  Linux SATA SSD: the 15K rows/s bar is NVMe-calibrated, a SATA SSD
                // tops out at 10-14K with direct's 16 reads in flight and mapped reads were 7x slower.  The switch is only made when asked for
                // (STRATA_PLE_AUTO_MMAP=1, the 0.1.42 behaviour); otherwise the slow probe warns and direct is kept.
                bool auto_mmap = !o.ple_io_explicit;
                if (const char* am = std::getenv("STRATA_PLE_AUTO_MMAP"); am == nullptr || std::atoi(am) == 0) auto_mmap = false;
                if (auto_mmap) {
                    std::fprintf(stderr, "strata generate: --ple-io direct measures slow here (%.0f rows/s, about %.0f MB/s of "
                                         "random 4 KiB reads): using --ple-io mmap instead (the OS file cache holds the pages; "
                                         "--ple-io direct forces the old mode, --ple-io ram keeps the table in RAM)\n",
                                 rps, rps * 4096.0 / 1e6);
                    ple_table.close();
                    pio.mode = strata::kernels::PleIo::Mmap;
                    pio.keepalive_ms = 0;
                    if (!ple_table.open(o.ple_gguf, err, pio)) {
                        return fail(1, "strata generate: %s\n", err.c_str());
                    }
                } else {
                    std::fprintf(stderr, "strata generate: WARNING: --ple-io direct measures slow here (%.0f rows/s, about %.0f "
                                         "MB/s of random 4 KiB reads): a prompt can spend minutes reading layer 1 "
                                         "(#1425, #1629). --ple-io mmap was 40-100x faster on some DRAM-less NVMe drives but 7x slower on a SATA SSD (#2040), so it is not chosen for you; "
                                         "--ple-io ram needs the RAM for the table%s\n", rps, rps * 4096.0 / 1e6,
                                 o.ple_io_explicit ? "" : " (kept direct: pass --ple-io mmap or --ple-io ram to change it; in a config file's \"args\" they are two entries: \"--ple-io\", \"mmap\")");
                }
            }
        }
        if (pio.mode == strata::kernels::PleIo::Direct) {
            // #605: --ple-io direct's random reads (up to --ple-inflight at once) are for SSDs; a rotational disk
            // can take minutes for one prompt chunk's rows.  Recommended, not forced.
            std::string dev;
            if (!under_wddm() && on_rotational_disk(o.ple_gguf, dev) == 1)   // (WSL's virtual disk says rotational)
                std::fprintf(stderr, "strata generate: WARNING: the n-gram table (%s) is on a rotational disk (%s): "
                                     "--ple-io direct reads it at random and can stall a prompt for minutes (#605). "
                                     "--ple-io ram (it then needs RAM for the table) or the model on an SSD is "
                                     "recommended\n", o.ple_gguf.c_str(), dev.c_str());
        }
        if (pio.lock)
            std::fprintf(stderr, "strata generate: PLE table %s (--ple-io ram) in %.1f s\n",
                         ple_table.locked() ? "locked in RAM" : "loaded (not locked)",
                         std::chrono::duration<double>(Clock::now() - tpl).count());
        if (pio.keepalive_ms > 0)
            std::fprintf(stderr, "strata generate: the SSD is kept awake while rows are read: one page of the table after "
                                 "%.0f ms without a read, until %.0f s after the last request "
                                 "(STRATA_SSD_KEEPALIVE=0 turns it off)\n", pio.keepalive_ms, pio.keepalive_window_s);
        else if (pio.mode == strata::kernels::PleIo::Direct)
            std::fprintf(stderr, "strata generate: SSD keep-alive off: the SSD may fall asleep between reads\n");
        const strata::core::WeightRef* wk = wt.find("blk.1.ple_key.weight");
        const strata::core::WeightRef* wv = wt.find("blk.1.ple_value.weight");
        const strata::core::WeightRef* wnk = wt.find("blk.1.ple_norm_key.weight");
        const strata::core::WeightRef* wnq = wt.find("blk.1.ple_norm_query.weight");
        const strata::core::WeightRef* wnc = wt.find("blk.1.ple_norm_conv.weight");
        const strata::core::WeightRef* wc = wt.find("blk.1.ple_conv1d.weight");
        if (!wk || !wv || !wnk || !wnq || !wnc || !wc) {
            return fail(1, "strata generate: the pack has no blk.1.ple_* tensors, so the PLE cannot be "
                           "wired - and running without it is a DIFFERENT MODEL (LEDGER L123)\n");
        }
        // `ple_key` is S2 and the loader has already widened its scales to f32, so the two planes are located
        // by the sizes the `WeightRef` records rather than re-derived - the same rule `plane_ptrs` follows.
        if (!wk->quantized()) {
            // plan v0.3 P6: the IQ model files' BF16 key (the pack's extra.bin, raw BF16)
            ss.ple.w.key_bf16 = (const uint16_t*) wk->data;
        } else if (wk->data != nullptr) {
            ss.ple.w.key_codes = (const uint8_t*) wk->data;
            ss.ple.w.key_scales = (const float*) ((const uint8_t*) wk->data + wk->codes_bytes);
        }
        if (o.native_ple_key && wk->quantized()) {
            if (!wk->native_data || (wk->native_type != 42 && wk->native_type != 18 && wk->native_type != 23 &&
                                     wk->native_type != 8) || !wk->native_q8_1) {
                return fail(1, "strata generate: native PLE key is absent or incompatible\n");
            }
            ss.ple.w.key_native_data = wk->native_data;
            ss.ple.w.key_native_type = wk->native_type;
            ss.ple.w.key_native_q8_1 = wk->native_q8_1;
        }
        ss.ple.w.value_bf16 = (const uint16_t*) wv->data;
        ss.ple.w.norm_key = (const float*) wnk->data;
        ss.ple.w.norm_query = (const float*) wnq->data;
        ss.ple.w.norm_conv = (const float*) wnc->data;
        // The conv1d kernel reads F16, so this cast is a claim about the pack's storage.  A checkpoint that keeps
        // the tensor F32 (Q8_0, UD-Q4_K_XL) would hand the kernel the low halves of the f32 words - not an error,
        // a plausible wrong layer-1 routing.  tools/iq_pack.py narrows it (index kind 3); a pack that did not is
        // refused here (#255, gopinath87607).  F16 is index kind 5 or 3 (F16InF32) in a native pack and kind 0
        // (verbatim 2-byte F16) in the canonical Q2_0 pack; BF16 (kind 4) has the same size and is not F16.
        const bool f16 = wc->kind == strata::core::WeightKind::F16InF32 ||
                         (wc->kind == strata::core::WeightKind::Verbatim && wc->code_bits == 0);
        if (!f16 || wc->bytes != (uint64_t) wc->elements * 2) {
            return fail(1, "strata generate: blk.1.ple_conv1d.weight is not stored as F16 (pack index kind %d, "
                           "%llu B for %lld values); the PLE conv1d kernel reads F16 - repack with "
                           "tools/iq_pack.py\n",
                        (int) wc->kind, (unsigned long long) wc->bytes, (long long) wc->elements);
        }
        ss.ple.w.conv1d_f16 = (const uint16_t*) wc->data;
        ss.ple.consts = strata::kernels::ple_artifact_consts();
        if (o.ple_delay_us > 0) ple_table.set_injected_delay_us(o.ple_delay_us);
        ss.ple.table = &ple_table;
        ss.ple.token = &ss.ple_token;
        ss.ple.prev = ss.ple_prev;
        // `ss.ple.hist` and the ready() check wait for `session_init`, which carves the history; the wiring lands
        // with the session below
        ss.ple.emb_host = ple_emb_host.data();
        if (cudaMalloc((void**) &ple_emb_dev, (size_t) strata::kernels::NG_N_EMBD * 4) != cudaSuccess ||
            cudaMalloc((void**) &ple_scratch, strata::core::ple_run_scratch_bytes()) != cudaSuccess) {
            return fail(1, "strata generate: the PLE buffers failed\n");
        }
        ss.ple.emb_dev = ple_emb_dev;
        ss.ple.scratch = ple_scratch;
    } else {
        std::fprintf(stderr,
                     "strata generate: PLE OFF by explicit --no-ple diagnostic request.\n"
                     "  The tokens below are NOT this model's; this is only useful for A/B measurement.\n");
    }

    if (cudaMalloc(&d_parts, (size_t) K * g.n_embd * 4) != cudaSuccess ||
        cudaMemset(d_parts, 0, (size_t) K * g.n_embd * 4) != cudaSuccess) {
        return fail(1, "strata generate: the parts buffer failed\n");
    }

    strata::core::Verifier::set_commit_async(true);   // see Verifier::set_commit_async
    if (!o.expert_profile.empty()) {
        int64_t pslots = 0;
        if (!strata::core::read_expert_profile(o.expert_profile, g.n_layers, g.n_expert, profile, pslots, err)) {
            return fail(1, "strata generate: %s\n", err.c_str());
        }
        // An explicit number truncates the ranked list ("what would 2,000 slots give" without rebuilding the
        // file).  `--expert-cache 0` used to take the count the profile was built for; the profile now ranks
        // every pair (issue #46: a card that holds more than the old 8,000 used to stop there), so it means auto.
        if (o.expert_cache == 0) o.expert_cache = -1;
        std::fprintf(stderr, "strata generate: profile %s: %zu ranked pairs, built for %lld slots\n",
                     o.expert_profile.c_str(), profile.size(), (long long) pslots);
    }
    // #477: the whole ranking as loaded, the prior of --expert-profile-save's order.  Empty without
    // --expert-profile-save.
    if (!o.expert_profile_save.empty()) profile_loaded = profile;
    pf_borrow = !o.no_prefill_borrow && !o.expert_profile.empty();

    // ---- the session
    {
        const int64_t hi0 = -1;
        // the elastic K/V (--kv-grow, see kvg_ensure): one GPU, the whole K/V in VRAM (no streaming), a profiled cache
        // that can give slots up, and every expert in RAM for the CPU to compute the ones it gives up
        {
            const char* ev = std::getenv("STRATA_KV_GROW");
            const bool asked = ev != nullptr && ev[0] != '\0' ? ev[0] != '0' : o.kv_grow;
            const bool on = asked && o.kv_resident <= 0 &&
                            !o.expert_profile.empty() && !o.resident_cpu_experts && o.expert_cache != 0 &&
                            strata::core::vmm_available();
            if (asked && !on)
                std::fprintf(stderr, "strata generate: --kv-grow is off (one GPU, a profile, the whole K/V in VRAM, "
                                     "every expert in RAM)\n");
            const char* iv = std::getenv("STRATA_KV_GROW_INIT");
            strata::core::qsa_set_kv_elastic(on, iv != nullptr && std::atoll(iv) > 0 ? std::atoll(iv) : 16384);
            strata::core::ExpertCache::set_vmm(on);
        }
        if (cudaMalloc(&sbuf, strata::core::session_bytes(g, o.max_context, K, 0, hi0)) != cudaSuccess) {
            return fail(1, "strata generate: session state allocation failed\n");
        }
        if (strata::core::session_init(g, o.max_context, K, sbuf, ss, 0, hi0) == 0) {
            return fail(1, "strata generate: session_init failed\n");
        }
        if (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1)
            std::fprintf(stderr, "strata generate: KV streaming: %lld of %lld cells per QSA layer in VRAM, the K/V in "
                                 "%.2f GiB of pinned RAM\n",
                         (long long) (ss.qsa_states[ss.qsa_primary()].n_slots * 4),
                         (long long) o.max_context, (double) strata::core::qsa_kv_host_bytes() / 1073741824.0);
        // the PLE block above built everything but the history, which session_init has just carved
        ss.ple.hist = ss.ple_hist;
        // (#167) nothing else zeroes this state before the prompt path or the verifier reads it (a request zeroes it
        // again when nothing is reused)
        strata::core::session_zero(ss, g, nullptr, main_cs);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            return fail(1, "strata generate: zeroing the session state failed\n");
        }
        if (!o.ple_gguf.empty()) {
            if (!ss.ple.ready()) {
                return fail(1, "strata generate: the PLE run is not ready after construction\n");
            }
            std::fprintf(stderr, "strata generate: PLE on, table %llu rows of %s\n",
                         (unsigned long long) ple_table.rows(), o.ple_gguf.c_str());
        }
    }
    mem_mark("the session (KV, GDN state, PLE history)");

    // Secure MTP's CUDA0 allocations before the large host arena is registered with both CUDA contexts.
    // In particular WDDM can refuse the draft weights after mapping tens of GiB of host pages.
    // the draft layer's geometry (the canonical model's MTP head), `static` because MtpDrafter keeps a reference
    static const strata::core::ModelGeometry draft_geometry{};
    if (!o.mtp.empty()) {
        if (o.spec < 2) {
            std::fprintf(stderr, "strata generate: --mtp is ignored without --spec T (T >= 2)\n");
            o.mtp.clear();
        }
        if (!o.mtp.empty()) mtp.set_prompt_len((int64_t) o.tokens.size());
        mtp.set_hnorm_per_stream(o.mtp_hnorm_stream);   // --mtp-hnorm stream (opt-in), before any capture
        mtp.set_q4(o.mtp_q4 == "all" || o.mtp_q4 == "proj", o.mtp_q4 == "all" || o.mtp_q4 == "head");   // before load
        if (!o.mtp_draft_vocab.empty()) mtp.set_draft_vocab(o.mtp_draft_vocab);
        // the draft layer is the canonical model's MTP head (512 experts) even when the target is pruned,
        // so it always sees the canonical geometry; `static` because MtpDrafter keeps a reference
        if (!o.mtp.empty() && !mtp.load(o.mtp, draft_geometry, ss, o.spec, err, o.mtp_window)) { return fail(1, "strata generate: %s%s\n", err.c_str(), vram_free_note().c_str()); }
        mtp.set_ple_session(&ss);
    }
    // THE HEAD BEFORE THE CACHE, AND BEFORE THE ARENA.  The expert cache takes what is free minus the reserve, so
    // everything allocated after it comes out of the reserve.  The native head (~0.5 GB with IQ3_S) was loaded after
    // it and ate most of the 700 MiB: 128K IQ3_S ended with 30 MiB free, the driver paged, and a request stalled for
    // good at its first verify window.  Loaded first, the cache is sized around it.  #620: and before the expert
    // arena registers tens of GiB of host pages, like the drafter above - WDDM then refused the head's cudaMalloc on
    // a 16 GB card with the desktop on the iGPU ("native head upload: out of memory" with GiBs free).
    wo = wt.find("output.weight");
    if (wo == nullptr) { return fail(1, "strata generate: output.weight is missing\n"); }
    n_vocab = wo->ne1;
    if (!o.native_head_gguf.empty()) {
        const auto head_t0 = std::chrono::steady_clock::now();
        if (!native_head.load(o.native_head_shards, g.n_embd, n_vocab, err)) {
            return fail(1, "strata generate: %s%s\n", err.c_str(), vram_free_note().c_str());
        }
        std::fprintf(stderr, "strata generate: experimental native Q5_K head, %llu bytes, in %.1f s\n",
                     (unsigned long long) native_head.weight_bytes(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - head_t0).count());
    }
    logits = std::vector<float>((size_t) n_vocab);
    mem_mark("the native head");
    if (cudaMalloc(&d_logits, (size_t) n_vocab * 4) != cudaSuccess) {
        (void) cudaGetLastError();
        return fail(1, "strata generate: the logits buffer failed%s\n", vram_free_note().c_str());
    }
    if (o.prefill_head_alias) {
        const char* why = o.short_read > 0 ? "it needs --short-read 0"
                        : o.expert_cache != 1 && o.expert_cache >= 0 ? "it needs --expert-cache 1 or auto"
                        : o.mtp.empty() ? "it needs --mtp"
                        : !native_head.loaded() ? "it needs the native output head"
                        : nullptr;
        if (why != nullptr) {
            return fail(2, "strata generate: --prefill-head-alias is unavailable: %s\n", why);
        }
        if (!mtp.bind(wt, &native_head, nullptr, err)) {
            return fail(1, "strata generate: --prefill-head-alias: %s\n", err.c_str());
        }
        native_head.unload();
        prefill_head_alias_bound = true;
        std::fprintf(stderr, "strata generate: --prefill-head-alias: the %llu-byte output head is prompt workspace\n",
                     (unsigned long long) native_head.weight_bytes());
    }

    // ---- the CPU expert pool
    //
    // R2.1: the experts are loaded into a RESIDENT ARENA by default.  The mmap path is kept behind
    // `--mmap-experts` because it is the A/B arm, not because it is competitive.
    //
    // The reasoning is the review's C1 and it is now measured on both sides.  `FileExpertSource` maps the 34 GB
    // file, and mapped file pages are the first thing the OS reclaims; the engine's rate then depends on whether
    // the standby list happens to hold `experts.bin`, which is why two consecutive runs of the SAME BINARY with
    // the SAME FLAGS measured 71.97 and 34.78 ms/token in the pool (7.54 vs 12.18 tok/s).  The arena is
    // anonymous memory the engine owns, and the pool runs at 19.41 ms/token - 1.79x better than the warm mmap
    // and 3.7x better than the cold one.
    //
    // IT IS NOT PINNED, and that is reported rather than hidden: `cudaHostRegister` on 31.64 GiB fails with
    // "out of memory" (you cannot pin 34 of 63 GB) and the arena falls back to 4 KB anonymous pages.  That is
    // fine for the CPU pool - which is all that exists today - and NOT fine for Phase 3, whose cache fills and
    // CPU/PCIe miss split need the GPU to DMA out of this arena.  Read `note()` when that lands.
    //
    // The earlier "the arena does not fit" conclusion was WRONG and is worth recording: the failure was a stale
    // CUDA error left set by the failed `cudaHostRegister` and read later by `gr_read`'s launch check.  See the
    // note in `pinned.cu`.
    {   // the card, and whether this build has code for it (a binary built for other GPUs fails at its first kernel
        // otherwise, after the whole expert arena has loaded) - before the arena starts loading
        int dev = 0;
        cudaDeviceProp p{};
        const bool named = cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess;
        if (!named) cudaGetLastError();
        const char* name = named && p.name[0] ? p.name : "(an unnamed GPU)";
        std::fprintf(stderr, "strata generate: GPU %d: %s, compute capability %d.%d%s\n", dev, name,
                     strata::cc_major_of(p.major), strata::cc_minor_of(p.minor),
                     strata::emulated_cc() ? " (STRATA_EMULATE_CC: a test mode, the card is emulated)" : "");
        {   // #542: a build whose libcudart is older than its headers (a CUDA 13 kit with a dangling libcudart.so that
            // CMake resolved to the system's CUDA 12 one) reads cudaDeviceProp shifted - silently, and slowly
            int rt = 0;
            if (cudaRuntimeGetVersion(&rt) == cudaSuccess && rt / 1000 != CUDART_VERSION / 1000)
                std::fprintf(stderr, "strata generate: WARNING: this engine was compiled with CUDA %d.%d headers but "
                                     "loaded a CUDA %d.%d runtime (libcudart): GPU properties can read wrong and some "
                                     "kernels go unused. Rebuild it against one toolkit (cmake -DCUDAToolkit_ROOT=<the "
                                     "toolkit>, with its libcudart.so present) (#542)\n",
                             CUDART_VERSION / 1000, CUDART_VERSION % 1000 / 10, rt / 1000, rt % 1000 / 10);
        }
        const std::string e = strata::core::device_code_error();
        if (!e.empty()) {
            return fail(1, "strata generate: this engine has no code for %s (sm_%d%d): %s - rebuild it for this "
                           "card (setup does: START-HERE.bat --setup)\n", name, p.major, p.minor, e.c_str());
        }
    }
    if (o.mmap_experts) {
        // FileExpertSource maps the pack's experts.bin: a canonical pack has it; a native (IQ) pack has it when
        // built with `tools/iq_pack.py --experts-bin` (the per-layer blob sizes of its layout, PR #121).  The low-RAM
        // mode: the experts come from the file through the OS cache instead of a pinned copy in RAM, for a PC whose
        // GPU holds most of them but whose RAM cannot hold them all.
        // CS-T: a native pack without experts.bin maps the model's GGUF shards instead (native_experts.txt's spans,
        // checked against the files first) - no 30-77 GB copy of the experts on the disk
        src.set_gguf(o.native_preset);
        if (const char* v = std::getenv("STRATA_FETCH_THREADS"); v != nullptr && std::atoi(v) > 0)
            src.set_fetch_threads(std::atoi(v));
        if (!src.open(o.pack, g.n_layers, g.n_expert, err)) {
            return fail(1, "strata generate: %s\n", err.c_str());
        }
        std::fprintf(stderr, "strata generate: experts via mmap (--mmap-experts; %s)\n",
                     src.gguf_mode() ? "the GGUF shards in place, no experts.bin" : "the A/B arm of R2.1");
        // #286: with a RAM budget the hottest experts live in it, and the rest are read from the drive unbuffered
        // when the file cache could not keep them beside the budget anyway (a 32 GB PC) - the mapped reads' page
        // faults are small requests on the critical path, and their pages take the RAM the budget was sized for
        // #773: without a RAM budget there is no RAM copy and the file cache IS the expert tier (measured on a
        // 32 GB, 2 x 16 GB rig: forcing unbuffered reads there re-read 163-629 GB from the drive and halved the
        // speed), so STRATA_UNBUFFERED_LOAD=1 is not honoured in that mode
        if (o.resident_budget > 0) {
            std::string why;
            const bool ub = src.set_unbuffered(o.resident_budget, why);
            std::fprintf(stderr, "strata generate: the file tier reads %s (%s)\n",
                         ub ? "unbuffered" : "through the file cache", why.c_str());
        } else if (const char* env = std::getenv("STRATA_UNBUFFERED_LOAD"); env != nullptr && env[0] != '\0' &&
                   env[0] != '0') {
            std::fprintf(stderr, "strata generate: the file tier reads through the file cache (STRATA_UNBUFFERED_LOAD=%s "
                                 "ignored: without --resident-budget-gib the file cache holds the experts)\n", env);
        }
        // Linux I/O path (opt in): whole-blob preads instead of page faults, and the predicted experts of the next
        // layers read ahead on I/O threads while the current layer computes (see FileExpertSource::set_io_prefetch)
        if (const char* v = std::getenv("STRATA_IO_PREFETCH"); v != nullptr && std::atoi(v) != 0) {
            const char* th = std::getenv("STRATA_IO_PF_THREADS");
            src.set_io_prefetch(true, th != nullptr ? std::atoi(th) : 0);
            if (src.io_prefetch())
                std::fprintf(stderr, "strata generate: file tier io prefetch ON (STRATA_IO_PREFETCH=1): whole-blob preads, "
                                     "the layer's uncached experts and the predicted ones read by %s I/O threads (STRATA_IO_PF_THREADS, default 8)\n",
                             th != nullptr ? th : "8");
            else
                std::fprintf(stderr, "strata generate: STRATA_IO_PREFETCH=1 is not available here (Linux mapped file tier "
                                     "only, not with unbuffered reads); the file tier reads as before\n");
            if (src.io_prefetch()) {   // recommend, never force: the engine runs as asked
                const char* st = std::getenv("STRATA_IO_PF_STAGE");
                if (st != nullptr && std::atoi(st) != 0)
                    std::fprintf(stderr, "strata generate: note: STRATA_IO_PF_STAGE=1 (staging) measured faster only on a 32 GB-class "
                                         "box (+25%% on an RTX 3060, 32 GB); it was slower at 16 GB and on an iGPU (docs/DETAILS.md, file tier I/O)\n");
            }
        } else if (const char* sv = std::getenv("STRATA_IO_STATS"); sv != nullptr && std::atoi(sv) != 0) {
            src.set_io_stats(true);
        }
        srcp = &src;
    } else {
        arena_src.set_gguf(o.native_preset);   // plan v0.3 P6: a native pack may take its experts from shard 1
        // The whole arena is pinned (#253).  STRATA_ARENA_PIN_GIB caps it.
        const int pin_env = strata::core::arena_pin_cap_gib();   // -1 unset, -2 "auto" (#243, Windows sliced pin)
        const uint64_t pin_limit = pin_env >= 0 ? (uint64_t) pin_env << 30 : 0;
        if (pin_env >= 0)
            std::fprintf(stderr, "strata generate: STRATA_ARENA_PIN_GIB=%d: %s\n", pin_env,
                         pin_env == 0 ? "the whole expert arena is pinned" : "the expert arena's pinning is capped");
        // ---- STRATA_PIN_BY_NEED=1 (default off: the prefix pinning above, byte-identical).  WDDM does not let the
        // pin grow past ~RAM/2, and the prefix spends it on layers 0-31 whatever the GPU holds: layers 32-47 stay 100 %
        // CPU while ~10 GiB of pinned blobs sit in GPU-resident experts no DMA ever reads.  Pinning is registered
        // BEFORE the arena's pages are touched (pinned.cu), so the set must be chosen here, before arena_src.open: the
        // profile is already read, and the GPU cache's size - opened much later - is ESTIMATED from the same free-VRAM
        // reading its auto sizing uses (the weights, session and drafter are loaded by now, host registration takes no
        // VRAM): the first R ranked pairs are taken as GPU-resident, and the rest (then the unranked pairs, in layer
        // order) are what the arena pins, hottest first, until the driver refuses.  R is under-estimated a little
        // (x0.98): a resident pair pinned by mistake costs a few MiB, a hot non-resident one skipped costs DMA.
        // STRATA_PIN_BY_NEED_SKIP=<n> forces R.  Single GPU, profile given, no helper caches only.
        std::vector<std::pair<int32_t, int32_t>> pin_cand;   // the non-resident estimate, by decreasing rank, then the resident one
        size_t pin_nonres = 0;                               // (how many of them are the non-resident estimate)
        if (const char* pn = std::getenv("STRATA_PIN_BY_NEED"); pn != nullptr && pn[0] != '\0' && pn[0] != '0') {
            const char* off_why = profile.empty() ? "it needs --expert-profile (the ranking)" : nullptr;
            if (off_why != nullptr) {
                std::fprintf(stderr, "strata generate: pin by need: off (%s); the prefix pinning stays\n", off_why);
            } else {
                const auto& pl = strata::kernels::cpu::expert_layout();
                size_t skip = profile.size();
                if (const char* sk = std::getenv("STRATA_PIN_BY_NEED_SKIP"); sk != nullptr && sk[0] != '\0') {
                    skip = std::min<size_t>(profile.size(), (size_t) std::max<long long>(std::atoll(sk), 0));
                } else {
                    // the bytes the GPU cache will hold: --expert-cache N slots of the largest blob, or what auto sizing
                    // finds free (the same reserve terms as "expert cache auto" below, without its small-card rescue,
                    // which only ever grants MORE slots)
                    uint64_t cap_bytes;
                    if (o.expert_cache > 0) {
                        cap_bytes = (uint64_t) o.expert_cache * pl.max_blob;
                    } else {
                        const int64_t pf_mib = (o.prefill_chunk > 0 && !pf_borrow) ? 160 + (o.prefill_chunk * 680) / 1024 : 0;
                        const int64_t mtp_b = (!o.mtp.empty() && native_head.loaded())
                                                  ? (int64_t) mtp.bind_bytes(native_head.row_bytes(), n_vocab) : 0;
                        const int64_t room = (int64_t) strata::core::device_free_bytes() -
                                             ((((int64_t) o.vram_reserve_mib + pf_mib) << 20) + mtp_b);
                        cap_bytes = room > 0 ? (uint64_t) room : 0;
                    }
                    cap_bytes = cap_bytes / 100 * 98;
                    uint64_t used = 0;
                    skip = 0;
                    for (const auto& pr : profile) {
                        const uint64_t b = pl.native ? (pl.blob_bytes(pr.first) + 255) / 256 * 256 : (uint64_t) pl.max_blob;
                        if (used + b > cap_bytes) break;
                        used += b;
                        ++skip;
                    }
                }
                std::vector<uint8_t> seen((size_t) (g.n_layers * g.n_expert), 0);
                for (const auto& pr : profile)
                    if (pr.first >= 0 && pr.first < g.n_layers && pr.second >= 0 && pr.second < g.n_expert)
                        seen[(size_t) (pr.first * g.n_expert + pr.second)] = 1;
                pin_cand.assign(profile.begin() + (long) skip, profile.end());
                for (int64_t l = 0; l < g.n_layers; ++l)
                    for (int64_t e = 0; e < g.n_expert; ++e)
                        if (!seen[(size_t) (l * g.n_expert + e)]) pin_cand.emplace_back((int32_t) l, (int32_t) e);
                pin_nonres = pin_cand.size();   // (the report counts these)
                // what the cap leaves over goes to the pairs taken as GPU-resident (last in line): the adaptive tier
                // swaps them out as the conversation moves, and the prompt read streams whatever is not resident then
                pin_cand.insert(pin_cand.end(), profile.begin(), profile.begin() + (long) skip);
                std::fprintf(stderr, "strata generate: pin by need: %zu of the %zu ranked pairs taken as GPU-resident "
                                     "(STRATA_PIN_BY_NEED_SKIP=<n> sets it), %zu candidates to pin\n",
                             skip, profile.size(), pin_nonres);
                arena_src.set_pin_priority(pin_cand, pin_nonres);
            }
        }
        if (!arena_src.open(o.pack, g.n_layers, g.n_expert, /*threads=*/6, err, pin_limit,
                            o.shared_expert_arena)) {
            return fail(1, "strata generate: %s\n", err.c_str());
        }
        if (!pin_cand.empty()) {
            const auto& pl = strata::kernels::cpu::expert_layout();
            uint64_t cand_b = 0, pin_b = 0;
            for (size_t ci = 0; ci < pin_nonres; ++ci) {
                const auto& pr = pin_cand[ci];
                const uint64_t b = pl.blob_bytes(pr.first);
                cand_b += b;
                if (arena_src.pinned(pr.first, pr.second)) pin_b += b;
            }
            int64_t pcie_layers = 0;
            for (int64_t l = 0; l < g.n_layers; ++l) pcie_layers += arena_src.pcie_layer(l) ? 1 : 0;
            std::fprintf(stderr, "strata generate: pin by need: %zu ranges, %.2f GiB pinned, %.1f %% of the non-resident "
                                 "experts (%lld of %lld layers have a PCIe share)\n",
                         arena_src.pin_ranges(), (double) pin_b / 1073741824.0,
                         cand_b > 0 ? 100.0 * (double) pin_b / (double) cand_b : 0.0, (long long) pcie_layers,
                         (long long) g.n_layers);
        }
        if (!arena_src.ram_warning().empty())   // #633: said before the load's numbers, which it explains
            std::fprintf(stderr, "strata generate: WARNING: %s\n", arena_src.ram_warning().c_str());
        std::fprintf(stderr, "strata generate: expert arena: %s\n", arena_src.note().c_str());
        std::fprintf(stderr, "strata generate: loaded %.2f GiB at %.2f GiB/s\n",
                     (double) strata::kernels::cpu::expert_layout().total / (1024.0 * 1024 * 1024),
                     arena_src.load_gib_per_second());
        // A rate under ~0.2 GiB/s is not the hardware.  Task Scheduler / service contexts throttle this
        // read+fill about 24x (measured 0.05 vs 1.42 GiB/s for the same binary, args and cache state; the
        // scheduler's defaults - Below normal priority and a least-privilege token - were the only
        // difference between the runs).  Say so instead of letting the user blame the disk; see
        // docs/DETAILS.md, "Running it at startup (Task Scheduler)".
        srcp = &arena_src;
    }
    // 0.1.43.1: on by default (STRATA_ADAPT_CALLBACK_WARMUP=0 turns it off).  Linux threads inherit the creator's CPU mask, and
    // the host pins itself to one CPU later; a callback thread the driver creates lazily after that would run on the
    // host's CPU and fight the host's spin-wait (985 us instead of 4 us per callback, #1964 #1975).  Windows CPU Sets are
    // per thread and not inherited.
    if (const char* e = std::getenv("STRATA_ADAPT_CALLBACK_WARMUP"); !(e && e[0] == '0')) {
        // The driver creates its host-callback thread lazily. Create it before ExpertPool pins this thread.
        if (cudaLaunchHostFunc(nullptr, [](void*) {}, nullptr) != cudaSuccess ||
            cudaStreamSynchronize(nullptr) != cudaSuccess) {
            (void) cudaGetLastError();
            std::fprintf(stderr, "strata generate: could not warm up the GPU host-callback thread (continuing)\n");
        } else {
            std::fprintf(stderr, "strata generate: GPU host-callback thread started before host-core pinning\n");
        }
    }
    if (o.pool_measure)
        o.pool_affinity = measure_pool_affinity(srcp, o.pool_workers, !o.no_host_worker, o.pool_tasks);
    pool.emplace(o.pool_workers, /*pin=*/true, /*host_works=*/!o.no_host_worker, o.pool_affinity, o.pool_tasks);
    std::fprintf(stderr, "strata generate: CPU pool tasks/phase: %d%s, participating threads: %d\n",
                 o.pool_tasks ? o.pool_tasks : 3 * (pool->workers() + (pool->host_works() ? 1 : 0)),
                 o.pool_tasks ? " (capped by rows)" : " (automatic)",
                 pool->workers() + (pool->host_works() ? 1 : 0));
    if (pool->is_hybrid() && pool->affinity() != strata::kernels::cpu::PoolAffinity::All) {
        const char* aff_str = pool->affinity() == strata::kernels::cpu::PoolAffinity::PCores ? "p-cores" :
                              pool->affinity() == strata::kernels::cpu::PoolAffinity::All ? "all" : "auto";
        std::fprintf(stderr, "strata generate: hybrid CPU detected (%d P-cores / %d threads, %d E-cores), pool workers: %d, affinity: %s\n",
                     pool->p_cores(), pool->p_threads(), pool->e_cores(), pool->workers(), aff_str);
    }
    {   // where the threads run (--host-core)
        const strata::kernels::cpu::CpuTopology ht = strata::kernels::cpu::detect_cpu_topology(true, o.pool_affinity);
        std::string on;
        for (int i = 0; i < pool->workers() && i < (int) ht.worker_cores.size(); ++i)
            on += (i > 0 ? "," : "") + std::to_string(ht.worker_cores[(size_t) i]);
        std::fprintf(stderr, "strata generate: %d pool workers on logical processors %s, the host thread on %d%s "
                             "(--host-core %s)\n", pool->workers(), on.c_str(), ht.host_core,
                     pool->host_works() ? " (draining too)" : "",
                     strata::kernels::cpu::host_core_setting() == strata::kernels::cpu::HostCore::Last      ? "last"
                     : strata::kernels::cpu::host_core_setting() == strata::kernels::cpu::HostCore::Sibling ? "sibling"
                                                                                                             : "first");
        // --aux-cpus: the plan, now the workers' CPUs are known; the threads that exist already (the CUDA driver's) move
        // now when it is on, the ones created later pin themselves (strata/platform/aux_cpus.hpp)
        const size_t nw = std::min<size_t>((size_t) pool->workers(), ht.worker_cores.size());
        std::fprintf(stderr, "%s\n",
                     strata::aux_cpus::set_topology(ht.host_core, std::vector<int>(ht.worker_cores.begin(),
                                                                                   ht.worker_cores.begin() + (long) nw)).c_str());
    }
    if (o.no_ple_prefetch) strata::kernels::ple_prefetch_enable(false);
    // ---- R4's slot storage.  Allocated AFTER the weights and the session, so `cudaMemGetInfo` inside `open`
    // sees the memory this process actually has left rather than the card's idle figure - and refuses with both
    // numbers if the slots do not fit, instead of handing back a cache smaller than it was asked for.
    mem_mark("the weights, the session and the drafter");
    // THE HEAD BEFORE THE CACHE: the native head and the logits are allocated above, before the expert arena (#620)
    auto_cache = o.expert_cache < 0;
    // The MiB the prompt path's OWN buffers (no loan) take, which the cache sizing leaves out.  0.1.39's rule is
    // 160 + chunk * 680 / 1024 (~16 GiB at 24576 against ~2.4 GiB really allocated); STRATA_OWNED_PRICE=exact prices the
    // real allocation instead (Prefill::bytes_needed_owned: 2 MiB pages, the ring in one piece, as PR #796 measured)
    // and gives that cache the difference.  A recommendation to try, so it is not the default.
    auto owned_prefill_mib = [&]() -> int64_t {
        if (!(o.prefill_chunk > 0 && !pf_borrow)) return 0;
        static const bool exact = [] { const char* v = std::getenv("STRATA_OWNED_PRICE"); return v != nullptr && std::string(v) == "exact"; }();
        if (!exact) return 160 + (o.prefill_chunk * 680) / 1024;
        const int64_t mib = ((int64_t) strata::prefill::Prefill::bytes_needed_owned(g, ss, o.prefill_chunk, srcp != nullptr) + (1 << 20) - 1) >> 20;
        std::fprintf(stderr, "strata generate: STRATA_OWNED_PRICE=exact: the prompt path's own buffers for a %lld-token chunk: %lld MiB (the 0.1.39 rule: %lld MiB)\n",
                     (long long) o.prefill_chunk, (long long) mib, (long long) (160 + (o.prefill_chunk * 680) / 1024));
        return mib + 64;   // a margin for the allocator
    };
    if (o.expert_cache < 0) {
        size_t free_b = 0, total_b = 0;
        free_b = strata::core::device_free_bytes(); (void) total_b;
        if (free_b == 0 && o.prefill_head_alias) {
            (void) cudaGetLastError();
            free_b = strata::core::device_free_bytes();
        }
        // Plan v0.3 P5: the batched prompt path's chunk buffers are allocated later, so they are reserved here -
        // under WDDM an over-subscribed allocation does not fail, it pages to system memory and crawls.
        // (with borrowing - the default with a profile - the prompt path lends cache slots instead; `pf_borrow` is
        // the predicate a local `borrow` was here, hoisted above so both cache-size branches read the same one)
        const int64_t prefill_mib = owned_prefill_mib();
        // the draft layer's head and logits are allocated when it binds, after this: 0.1.27's CJK subset made them
        // ~110-180 MiB larger, and out of the reserve they left 16 GB cards below the stall line (#199)
        const int64_t mtp_bind = (!o.mtp.empty() && native_head.loaded())
                                     ? (int64_t) mtp.bind_bytes(native_head.row_bytes(), n_vocab) : 0;
        const int64_t reserve = (((int64_t) o.vram_reserve_mib + prefill_mib) << 20) + mtp_bind;
        const int64_t blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
        int64_t slots = ((int64_t) free_b - reserve) / blob;
        if (!profile.empty()) slots = std::min<int64_t>(slots, (int64_t) profile.size());
        o.expert_cache = (int) std::max<int64_t>(slots, 0);
        std::fprintf(stderr, "strata generate: expert cache auto: %.2f GiB free, %d MiB reserved (+%lld MiB for the "
                             "draft head) -> %d slots\n",
                     (double) free_b / 1073741824.0, o.vram_reserve_mib, (long long) (mtp_bind >> 20), o.expert_cache);
        // #496: the verify window cannot start without a cache (#174), and a cache too small to lend the prompt path
        // a 256-token chunk's buffers (plus the 128 slots a loan leaves; one slot without --prefill) makes it
        // allocate its own on top - more than the reserve.  When the default reserve leaves less than that (a 6 GB
        // card), the reserve shrinks to what leaves exactly that cache, down to kSmallReserveMib: what is allocated
        // after the cache - the prompt path's own part, the verify buffers, the draft head - comes out of the reserve,
        // and below ~550 MiB a card ends with less than the 256 MiB the serve check calls LOW (IQ3_XXS, 32K, a 300 MiB
        // reserve: 5 MiB left), so the cache gets no more than it needs, and the serve check says so plainly when it
        // ends LOW (`reserve_adapted`).  A reserve given on the command line is kept.  No slot at all: the start
        // stops, saying what is short and what makes room.  A card the default reserve leaves that much is sized as
        // before.
        constexpr int kSmallReserveMib = 300;
        const int64_t min_slots = (o.prefill_chunk > 0 && pf_borrow)
            ? ((int64_t) strata::prefill::Prefill::bytes_needed(g, ss, 256) + blob - 1) / blob + 128 : 1;
        if (o.expert_cache < min_slots && !o.vram_reserve_given && o.vram_reserve_mib > kSmallReserveMib) {
            // the largest reserve (in MiB) that still leaves min_slots
            const int64_t fit_mib = ((int64_t) free_b - mtp_bind - min_slots * blob) / (1 << 20) - prefill_mib;
            if (fit_mib >= kSmallReserveMib) {
                const int r = (int) std::min<int64_t>(fit_mib, o.vram_reserve_mib);
                int64_t s2 = ((int64_t) free_b - ((((int64_t) r + prefill_mib) << 20) + mtp_bind)) / blob;
                if (!profile.empty()) s2 = std::min<int64_t>(s2, (int64_t) profile.size());
                std::fprintf(stderr, "strata generate: expert cache auto: the %d MiB reserve leaves too few slots on "
                                     "this card (a working cache needs %lld): a %d MiB reserve instead -> %lld slots\n",
                             o.vram_reserve_mib, (long long) min_slots, r, (long long) s2);
                o.vram_reserve_mib = r;
                o.expert_cache = (int) s2;
                reserve_adapted = true;
            }
        }
        if (o.expert_cache == 0) {
            // what is short, and what makes room: the numbers a small card picks from
            const int64_t at_reserve = o.vram_reserve_given ? o.vram_reserve_mib
                                                            : std::min(o.vram_reserve_mib, kSmallReserveMib);
            const int64_t need_b = (((int64_t) at_reserve + prefill_mib) << 20) + mtp_bind + min_slots * blob;
            const int64_t short_mib = std::max<int64_t>(1, (need_b - (int64_t) free_b + (1 << 20) - 1) >> 20);
            const int64_t session_mib =
                (int64_t) (strata::core::session_bytes(g, o.max_context, K, 0, g.n_layers) >> 20);
            const std::string reserve_tip =
                o.vram_reserve_given && o.vram_reserve_mib > kSmallReserveMib
                    ? ", a smaller --vram-reserve-mib (" + std::to_string(o.vram_reserve_mib) + " now; " +
                          std::to_string(kSmallReserveMib) + " is enough on a small card)"
                    : std::string();
            std::fprintf(stderr, "strata generate: no VRAM is left for the expert cache: it needs at least %lld slots "
                                 "(%lld MiB), about %lld MiB more than this card has free. To make room: a smaller "
                                 "--max-context (the session, mostly its KV cache, takes %lld MiB at %lld tokens), "
                                 "--kv q4_0, the English draft subset (setup --draft-vocab en; the draft head takes "
                                 "%lld MiB now)%s, images on the CPU, or close other programs that use the GPU\n",
                         (long long) min_slots, (long long) ((min_slots * blob) >> 20), (long long) short_mib,
                         (long long) session_mib, (long long) o.max_context, (long long) (mtp_bind >> 20),
                         reserve_tip.c_str());
        }
    }
    // plan v0.3 P6: a native pack's blobs differ per layer, so the same VRAM holds more experts than slots of
    // the largest blob would. The shared cache below sizes slots in profile order.
    // #369: that list does not match --expert-cache-per-layer. A range is layer * quota + n, so a size cut for
    // one layer can land in another layer's slot. Every expert in a layer is one size, so the per-layer list is
    // `quota` copies of that layer's own blob, in layer order. `--expert-cache N` stays the budget of N largest
    // blobs, and the quota is however many copies of every layer fit in it.
    if (native_pack && o.expert_cache > 0 && o.expert_cache_per_layer) {
        const size_t free_b = strata::core::device_free_bytes();   // the same reading the shared-cache sizing uses
        const auto& lay = strata::kernels::cpu::expert_layout();
        const int asked = o.expert_cache;
        const uint64_t budget = (uint64_t) asked * lay.max_blob;
        size_t free_room = free_b > ((size_t) o.vram_reserve_mib << 20) ? free_b - ((size_t) o.vram_reserve_mib << 20) : 0;
        const uint64_t cap = std::min<uint64_t>(budget, (uint64_t) free_room);
        uint64_t sum = 0;
        for (int64_t l = 0; l < g.n_layers; ++l)
            sum += (uint64_t) strata::core::ExpertCache::slot_stride((int64_t) lay.blob_bytes(l), slot_guard);
        int64_t q = sum > 0 ? (int64_t) (cap / sum) : 0;
        if (q > g.n_expert) q = g.n_expert;
        if (q > 0) {
            per_layer_bytes = sum;
            sized_slots.reserve((size_t) q * (size_t) g.n_layers);
            for (int64_t l = 0; l < g.n_layers; ++l) {
                const int64_t b = (int64_t) lay.blob_bytes(l);
                for (int64_t s = 0; s < q; ++s) sized_slots.push_back(b);
            }
            o.expert_cache = (int) sized_slots.size();
            std::fprintf(stderr, "strata generate: per-layer slots use each layer's blob: %d uniform slots "
                                 "(%.2f GiB) -> %d slots, %lld per layer (%.2f GiB)\n",
                         asked, (double) budget / 1073741824.0, o.expert_cache, (long long) q,
                         (double) ((uint64_t) q * per_layer_bytes) / 1073741824.0);
        }
    } else if (native_pack && o.expert_cache > 0 && !profile.empty()) {
        size_t free_b = 0, total_b = 0;
        free_b = strata::core::device_free_bytes(); (void) total_b;
        const auto& lay = strata::kernels::cpu::expert_layout();
        // what the uniform sizing granted, in the 256-byte slots the loop below counts: the blob formats of 1280 / 2560
        // rows are whole multiples of 256 already; an NVFP4 blob's 16-byte scale tail is not, and the unrounded budget
        // held one slot fewer than asked (--expert-cache 1: none)
        const uint64_t budget = (uint64_t) o.expert_cache * ((lay.max_blob + 255) / 256 * 256);
        uint64_t used = 0;
        const size_t keep_free = (size_t) o.vram_reserve_mib << 20;
        size_t free_room = free_b > keep_free ? free_b - keep_free : 0;
        const uint64_t cap = std::min<uint64_t>(budget, (uint64_t) free_room);
        for (const auto& pr : profile) {
            const uint64_t b = (uint64_t) strata::core::ExpertCache::slot_stride((int64_t) lay.blob_bytes(pr.first), slot_guard);
            if (used + b > cap) break;
            used += b;
            sized_slots.push_back((int64_t) lay.blob_bytes(pr.first));
        }
        // #831: the sized-slots loop walks the profile's pairs, so the profile's length is the ceiling; an explicit
        // --expert-cache above it used to be cut with no word.  A warning only (what is allocated does not change).
        if (!auto_cache && o.expert_cache > 0 && sized_slots.size() == profile.size() && (size_t) o.expert_cache > profile.size())
            std::fprintf(stderr, "strata generate: WARNING: --expert-cache %d is more than the %zu pairs of the profile %s: "
                                 "%zu slots (the profile is the ceiling on a native pack; a profile that ranks every "
                                 "(layer, expert) pair lifts it - tools/make_profile.py)\n", o.expert_cache,
                         profile.size(), o.expert_profile.c_str(), profile.size());
        o.expert_cache = (int) sized_slots.size();
    }
    // STRATA_MMQ_DIRECT: the guard after each sized slot (kept across the retry loop's close/open)
    xcache.set_slot_guard(slot_guard);
    if (slot_guard > 0 && o.expert_cache > 0) {
        if (!sized_slots.empty())
            std::fprintf(stderr, "strata generate: +%lld B guard per sized slot (STRATA_MMQ_DIRECT)\n", (long long) slot_guard);
        else   // xcache.open below: every slot is the largest blob's size, and no guard is added
            std::fprintf(stderr, "strata generate: uniform expert slots, no guard (STRATA_MMQ_DIRECT): an expert is read "
                                 "in its slot where the slot's slack covers the over-read\n");
    }
    if (o.expert_cache > 0) {
        // keep the first `keep_bytes` of the cache (the profile's hottest experts first); false when nothing is left
        auto shrink_to = [&](int64_t keep_bytes) -> bool {
            if (keep_bytes <= 0) { o.expert_cache = 0; sized_slots.clear(); return false; }
            if (per_layer_bytes > 0) {
                // Cutting the list short would put one layer's blob in the next layer's range.
                const auto& lay = strata::kernels::cpu::expert_layout();
                int64_t q = keep_bytes / (int64_t) per_layer_bytes;
                if (q > g.n_expert) q = g.n_expert;
                sized_slots.clear();
                if (q <= 0) { o.expert_cache = 0; return false; }
                for (int64_t l = 0; l < g.n_layers; ++l) {
                    const int64_t b = (int64_t) lay.blob_bytes(l);
                    for (int64_t s = 0; s < q; ++s) sized_slots.push_back(b);
                }
                o.expert_cache = (int) sized_slots.size();
                return true;
            }
            if (!sized_slots.empty()) {
                int64_t used = 0;
                size_t keep = 0;
                while (keep < sized_slots.size() &&
                       used + strata::core::ExpertCache::slot_stride(sized_slots[keep], slot_guard) <= keep_bytes)
                    used += strata::core::ExpertCache::slot_stride(sized_slots[keep++], slot_guard);
                sized_slots.resize(keep);
                o.expert_cache = (int) keep;
            } else {
                o.expert_cache = (int) (keep_bytes / (int64_t) strata::kernels::cpu::expert_layout().max_blob);
            }
            if (o.expert_cache <= 0) { o.expert_cache = 0; sized_slots.clear(); return false; }
            return true;
        };
        auto cache_bytes = [&]() -> int64_t {
            if (sized_slots.empty()) return (int64_t) o.expert_cache * (int64_t) strata::kernels::cpu::expert_layout().max_blob;
            int64_t b = 0;
            for (const int64_t s : sized_slots) b += strata::core::ExpertCache::slot_stride(s, slot_guard);
            return b;
        };
        // With `--expert-cache auto` the reserve must still be free once the slots are WRITTEN: under WDDM an
        // allocation is not resident until it is touched, and the free figure read before it can be ~1 GB too
        // high.  A cache sized from it filled the card to 0 MiB, the driver then paged, and a request that needed a
        // page back while the verify graph spun on a host flag never finished.  So the slots are zeroed and the
        // free figure read again; while it is short of the reserve the cache is reopened smaller.
        // STRATA_TEST_CACHE_FAIL=N: the first N opens fail as an out-of-commit cudaMalloc does (tests the retry)
        int fake_fails = std::getenv("STRATA_TEST_CACHE_FAIL") ? std::atoi(std::getenv("STRATA_TEST_CACHE_FAIL")) : 0;
        int failed = 0;
        int zero_reads = 0;
        for (int attempt = 0;; ++attempt) {
            bool ok = false;
            if (fake_fails > 0) {
                --fake_fails;
                err = "ExpertCache: cudaMalloc failed: out of memory (STRATA_TEST_CACHE_FAIL)";
            } else {
                ok = sized_slots.empty()
                    ? xcache.open(o.expert_cache, g.n_layers, g.n_expert, (int64_t) strata::kernels::cpu::expert_layout().max_blob, err)
                    : xcache.open_sized(sized_slots, g.n_layers, g.n_expert, err);
            }
            if (!ok) {
                // Issue #60: on Windows a device allocation is also charged to the system commit (RAM + page file),
                // so with a small page file the cache's one big cudaMalloc fails while the VRAM is free.  An auto
                // cache then tries three quarters of the size, a few times, instead of stopping the engine.
                char commit[96] = "";
                if (auto_cache && failed < 8 && shrink_to(cache_bytes() / 4 * 3)) {
                    ++failed;
                    std::fprintf(stderr, "strata generate: %s%s; trying a smaller expert cache: %d slots\n", err.c_str(),
                                 commit, o.expert_cache);
                    continue;
                }
                return fail(1, "strata generate: %s\n", err.c_str());
            }
            if (!auto_cache || attempt - failed >= 6) break;
            cudaMemset(xcache.device_slot(0), 0, (size_t) xcache.bytes());
            cudaDeviceSynchronize();
            size_t free_b = 0, total_b = 0;
            free_b = strata::core::device_free_bytes(); (void) total_b;
            const int64_t want = (int64_t) o.vram_reserve_mib << 20;
            if ((int64_t) free_b >= want - (64ll << 20)) break;
            // short by (want - free); a figure of 0 only says "at least": the first two such reads give back 1 GiB
            // each (under WDDM the free figure read before the allocation runs ~0.7 GiB high), later ones a quarter
            int64_t give = want - (int64_t) free_b + (64ll << 20);
            if (free_b < ((size_t) 16 << 20))
                give = std::max<int64_t>(give, ++zero_reads <= 2 ? 1ll << 30 : xcache.bytes() / 4);
            const int64_t keep_bytes = xcache.bytes() - give;
            std::fprintf(stderr, "strata generate: only %lld MiB free once the slots are written (reserve %d MiB); "
                                 "shrinking the expert cache\n", (long long) (free_b >> 20), o.vram_reserve_mib);
            xcache.close();
            if (!shrink_to(keep_bytes)) break;
        }
        if (failed > 0 && o.expert_cache > 0)
            std::fprintf(stderr, "strata generate: expert cache: %d slots (%.2f GiB) after %d smaller tries - a bigger "
                                 "page file lets it use more of the free VRAM\n",
                         o.expert_cache, (double) xcache.bytes() / 1073741824.0, failed);
    }
    if (o.expert_cache > 0) {
        std::fprintf(stderr, "strata generate: expert cache %lld slots, %.2f GiB of VRAM; policy is\n",
                     (long long) xcache.slots(), xcache.gib());
        mem_mark("opening the expert cache");
        xcache.set_per_layer_admission(o.expert_cache_per_layer);
        // Round 328 warned here that the GPU hit path was wrong (tokens diverged from a cache-off run from
        // token 0). That fault was fixed long since (native_expert_parity, expert_parity, the grouped kernels'
        // tests), and the warning outlived it (issue #23). What remains is rounding: a GPU expert and the CPU's
        // compute the same quantized expert with different float order, so a near-tie can flip. Measured teacher-
        // forced on 2,557 tokens (bench/results/2026-09-27-cache-parity): 95-98% same top-1, and perplexity equal
        // (on - off = -0.005 +- 0.005 nats). Neither output is more correct than the other.
        std::fprintf(stderr,
                     "strata generate: the GPU computes the experts in the cache; it rounds differently from the CPU,\n"
                     "                 so a reply can differ slightly from a run without the cache (same quality:\n"
                     "                 bench/results/2026-09-27-cache-parity).\n");
        if (o.expert_cache_per_layer) {
            int64_t lo = 0, hi = 0;
            xcache.layer_slot_range(0, lo, hi);
            std::fprintf(stderr, "                 R4.2g PER-LAYER: each layer owns %lld slots (%lld..%lld).\n",
                         (long long) (hi - lo), (long long) lo, (long long) (hi - 1));
        } else if (profile.empty()) {
            std::fprintf(stderr, "                 compulsory-miss (fills with whatever the run routes first).\n");
        } else {
            std::fprintf(stderr, "                 PROFILE, ranked by routing frequency, no eviction.\n");
        }
    }
    // ---- R4.2e: fill the tier from the profile.  This is the only place the plan is applied, and it runs
    // ONCE: with `slots` pairs and `slots` slots the cache is full when this returns, so the decode-time
    // admission finds no room and every non-profiled expert stays a CPU miss.  That is what makes the profile
    // the policy rather than a hint.
    if (!profile.empty() && srcp != nullptr) {
        // #369 (dag08): per layer, a full layer skips only its own pairs - each layer takes its hottest experts until
        // its range is full (one full layer used to end the whole fill, leaving most layers empty)
        const bool per_layer = xcache.per_layer_admission();
        const int64_t want = per_layer ? (int64_t) profile.size()
                                       : std::min<int64_t>((int64_t) profile.size(), xcache.slots());
        // #286: an unbuffered file tier reads the pairs in batches of 64, the next batch while this one is copied
        std::future<void> ahead;
        auto read_batch = [&](int64_t at) { src.prefetch_pairs(profile.data() + at, std::min<int64_t>(64, want - at)); };
        // mapped reads: advise the next `fill_ahead` pairs so their reads overlap; STRATA_FILL_AHEAD=0 turns it off
        int64_t fill_ahead = 0;
        if (!per_layer && srcp == &src && !src.unbuffered()) {
            const char* v = std::getenv("STRATA_FILL_AHEAD");
            fill_ahead = v == nullptr ? 256 : std::max(0, std::atoi(v));
            if (fill_ahead > 0 && !src.advise_pairs(profile.data(), std::min<int64_t>(fill_ahead, want))) fill_ahead = 0;
            std::fprintf(stderr, "strata generate: the profile fill asks for %lld pairs ahead (STRATA_FILL_AHEAD)\n",
                         (long long) fill_ahead);
        }
        const auto fill_t0 = std::chrono::steady_clock::now();
        uint64_t fill_bytes = 0;
        for (int64_t i = 0; i < want; ++i) {
            if (!per_layer && srcp == &src && src.unbuffered() && i % 64 == 0) {
                if (ahead.valid()) ahead.get();
                else read_batch(i);
                if (i + 64 < want) ahead = std::async(std::launch::async, [&read_batch](int64_t b) { strata::aux_cpus::pin_current_thread(); read_batch(b); }, i + 64);
            }
            if (fill_ahead > 0 && i + fill_ahead < want) (void) src.advise_pairs(profile.data() + i + fill_ahead, 1);
            const int32_t slot = xcache.admit(profile[(size_t) i].first, profile[(size_t) i].second);
            if (slot == strata::core::kNotResident) {
                if (per_layer) continue;
                break;
            }
            const uint8_t* b = srcp->blob_stable(profile[(size_t) i].first, profile[(size_t) i].second);
            if (b == nullptr || !xcache.fill_slot_blocking(slot, b, err,
                    (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(profile[(size_t) i].first))) {
                return fail(1, "strata generate: the profile fill failed at pair %lld: %s\n",
                            (long long) i, err.c_str());
            }
            fill_bytes += strata::kernels::cpu::expert_layout().blob_bytes(profile[(size_t) i].first);
            ++prefilled;
        }
        const double fill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - fill_t0).count();
        // **AND ONE SLOT IS READ BACK AND COMPARED.**  A residency table that is right about indices and wrong
        // about bytes produces a plausible token, which is this project's most expensive failure mode; the
        // cache's own `verify_slot` is the check and it costs one 1.38 MB D2H at startup.
        if (prefilled > 0 && !xcache.verify_slot(xcache.slot_of(profile[0].first, profile[0].second),
                                srcp->blob_stable(profile[0].first, profile[0].second), err,
                                (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(profile[0].first))) {
            return fail(1, "strata generate: %s\n", err.c_str());
        }
        mem_mark("the profile fill");
        std::fprintf(stderr, "strata generate: pre-filled %lld of %lld slots from the profile in %.1f s (%.0f MB/s); "
                             "slot 0 verified\n",
                     (long long) prefilled, (long long) (per_layer ? xcache.slots() : want), fill_s,
                     fill_s > 0 ? (double) fill_bytes / 1e6 / fill_s : 0.0);
        if (std::getenv("STRATA_VERIFY_COMPLEMENT") != nullptr) {
            // tests: the filled slots' checksum (each slot's blob read back and hashed, then the hashes in slot order)
            std::vector<std::pair<int32_t, int64_t>> sl;   // (slot, blob bytes)
            for (int64_t l = 0; l < g.n_layers; ++l)
                for (int64_t e = 0; e < g.n_expert; ++e)
                    if (const int32_t s = xcache.slot_of(l, e); s >= 0)
                        sl.emplace_back(s, (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(l));
            std::sort(sl.begin(), sl.end());
            std::vector<uint64_t> sh(sl.size(), 0);
            std::atomic<size_t> next_slot{0};
            std::atomic<bool> copy_ok{true};
            auto hash_work = [&] {
                std::vector<uint8_t> hb;
                for (size_t i; (i = next_slot.fetch_add(1)) < sl.size();) {
                    hb.resize((size_t) sl[i].second);
                    if (cudaMemcpy(hb.data(), xcache.device_slot(sl[i].first), hb.size(), cudaMemcpyDeviceToHost) !=
                        cudaSuccess) copy_ok = false;
                    sh[i] = strata::core::fnv1a64(hb.data(), (uint64_t) hb.size());
                }
            };
            std::vector<std::thread> hth;
            for (int t = 1; t < 8; ++t) hth.emplace_back(hash_work);
            hash_work();
            for (auto& t : hth) t.join();
            uint64_t h = strata::core::fnv1a64(nullptr, 0);
            for (size_t i = 0; i < sl.size(); ++i) {   // field by field: a pair's padding is not part of it
                const int64_t v[2] = {(int64_t) sl[i].first, sl[i].second};
                h = strata::core::fnv1a64((const uint8_t*) v, sizeof v, h);
                h = strata::core::fnv1a64((const uint8_t*) &sh[i], sizeof sh[i], h);
            }
            std::fprintf(stderr, "strata generate: expert cache checksum %016llx (%zu slots%s)\n", (unsigned long long) h,
                         sl.size(), copy_ok ? "" : ", A READ-BACK FAILED");
        }
    }

    drive.d.hit_cpu_order = o.expert_cache_cpu_order;
    drive.d.split_rows = !o.no_split_rows;
    drive.d.pool = &*pool;
    drive.d.src = srcp;
    drive.d.n_expert = g.n_expert;
    drive.d.jobs.resize((size_t) K);
    // CS-T: routing-aware prefetch of the file tier (the GGUF in place): the next layer's router on this layer's MoE
    // input predicts its experts and their pages are warmed meanwhile.  It only warms pages; STRATA_LOOKAHEAD=0 is
    // the A/B arm, STRATA_LOOKAHEAD_K the experts per token (default 10).
    // #1348, one prefetch design: with the Foresight swap space on (STRATA_FS_SLOTS) the look-ahead also runs on the pinned-RAM
    // tiers, where there are no pages to warm, and feeds its predictions to the swap space (STRATA_FS_AHEAD=0: not)
    fs_ahead = srcp != nullptr && [] { const char* v = std::getenv("STRATA_FS_SLOTS"); return v != nullptr && std::atoi(v) > 0; }() &&
                          [] { const char* v = std::getenv("STRATA_FS_AHEAD"); return v == nullptr || std::atoi(v) != 0; }();
    // STRATA_LOOKAHEAD_STATS=1 (measurement only): run the predictor on any tier and score it against the real routing
    la_stats = srcp != nullptr && [] { const char* v = std::getenv("STRATA_LOOKAHEAD_STATS"); return v != nullptr && std::atoi(v) != 0; }();
    if (srcp != nullptr && ((srcp == &src && src.warms()) || fs_ahead || la_stats) &&
        [] { const char* v = std::getenv("STRATA_LOOKAHEAD"); return v == nullptr || std::atoi(v) != 0; }()) {
        std::vector<std::vector<uint16_t>> routers((size_t) g.n_layers);
        bool ok = true;
        for (int64_t l = 0; l < g.n_layers && ok; ++l) {
            const strata::core::WeightRef* w = wt.find("blk." + std::to_string(l) + ".ffn_gate_inp.weight");
            ok = w != nullptr && w->kind == strata::core::WeightKind::Bf16InF32 &&
                 w->bytes == (uint64_t) (g.n_expert * g.n_embd) * 2;
            if (!ok) break;
            routers[(size_t) l].resize((size_t) (g.n_expert * g.n_embd));
            ok = cudaMemcpy(routers[(size_t) l].data(), w->data, (size_t) w->bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        const char* kv = std::getenv("STRATA_LOOKAHEAD_K");
        if (ok && lookahead.start(std::move(routers), g.n_embd, g.n_expert, kv ? std::atoi(kv) : 10, srcp, err, fs_ahead || la_stats)) {
            drive.d.lookahead = &lookahead;
            if (la_stats) lookahead.enable_stats(g.n_layers);
            if (const char* dv = std::getenv("STRATA_IO_PREFETCH_DEPTH"); dv != nullptr && std::atoi(dv) > 0)
                lookahead.set_depth(std::atoi(dv));
            else if (srcp == &src && src.io_prefetch())
                lookahead.set_depth(2);
            std::fprintf(stderr, "strata generate: routing-aware prefetch on (the next layer's router%s)\n",
                         fs_ahead ? ", feeding the Foresight swap space" : ", file tier");
        } else {
            (void) cudaGetLastError();
            std::fprintf(stderr, "strata generate: routing-aware prefetch off (%s)\n",
                         ok ? err.c_str() : "the routers are not BF16 in the arena");
            err.clear();
        }
    }
    // ---- R4.2c: THE HIT PATH.  Every one of these is required for `hits_ready()`, which is all-or-nothing on
    // purpose: a half-configured hit path would compute some experts twice and others not at all, and a token
    // built on that is wrong rather than refused.
    if (o.expert_cache > 0 && !o.no_pool) {
        const uint64_t sb = strata::kernels::moe_hit_grouped_scratch_bytes(K, g.n_embd, strata::kernels::cpu::FF);
        if (cudaMalloc(&hit_scratch, (size_t) sb) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_slot, (size_t) K * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_dst, (size_t) K * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_q8, (size_t) (g.n_embd / 32) * 34) != cudaSuccess ||
            // R4.2h: the fp32 activation scales.  Without this the GPU's hits use the block's fp16 `d`
            // while the CPU's misses use `ActQ::scale`, which is fp32 - a 4.761e-04 relative disagreement on
            // every chunk, and the reason enabling the cache changed the tokens.
            cudaMalloc((void**) &d_hit_q8_scale, (size_t) (g.n_embd / 32) * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_out, (size_t) K * g.n_embd * 4) != cudaSuccess) {
            return fail(1, "strata generate: the R4 hit path could not allocate its device buffers\n");
        }
        drive.d.cache = &xcache;
        drive.d.cache_stream = main_cs;
        drive.d.cache_base = (const uint8_t*) xcache.device_slot(0);
        drive.d.cache_blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
        drive.d.cache_slot_off = xcache.slot_offsets();
        drive.d.hit_scratch = hit_scratch;
        drive.d.parts_out = d_parts;
        drive.d.hit_out = d_hit_out;
        drive.d.parts_elems = K * g.n_embd;
        drive.d.mixed = ss.block.mixed;
        drive.d.x_q8_0_hit = d_hit_q8;
        drive.d.x_q8_0_hit_scale = d_hit_q8_scale;
        drive.d.d_slot = d_hit_slot;
        drive.d.d_dst = d_hit_dst;
        drive.d.h_slot.resize((size_t) K);
        cudaEvent_t hit_done = nullptr;
        if (cudaEventCreate(&hit_done) != cudaSuccess) {
            return fail(1, "strata generate: the hit path could not create its probe event\n");
        }
        drive.d.hit_done = (void*) hit_done;
        drive.d.hit_poke = !o.no_hit_poke;
        drive.d.h_dst.resize((size_t) K);
        mem_mark("the R4 hit path");
        std::fprintf(stderr, "strata generate: R4 hit path ON - resident experts are computed on the GPU\n");
    }
    // The hit hook rides the same switch as the pool: with no pool there is no `parts` staging to
    // write into, and a hit path with nowhere to write is a wrong token rather than an error.
    hit_fn =
        (o.no_pool || o.expert_cache <= 0) ? nullptr : &strata::core::expert_hit_run;
    std::fprintf(stderr, "strata generate: %d expert-pool workers%s%s\n", pool->workers(),
                 pool->host_works() ? " + the host thread" : "",
                 o.no_pool ? " (UNUSED: --no-pool)" : "");

    // **THE MISALIGNMENT WARNING THAT STOOD HERE IS GONE, BECAUSE THE MISALIGNMENT IS FIXED.**
    //
    // It said the tokens were not the model's, and it was true: `session_loop` handed layer `l`'s expert
    // outputs to layer `l+1`, which multiplied them by layer `l+1`'s router weights (LEDGER L100).  The loop
    // now runs a captured PAIR per layer - `pre[l]` ending with the router and the doorbell, then the CPU
    // pool, then `post[l]` which combines those experts with THAT layer's weights - so layer `l`'s experts meet
    // layer `l`'s routing.  The generated ids changed the moment it landed, which is what a correctness fix
    // looks like from the outside.
    //
    // The cost is real and is recorded rather than hidden: the window for the CPU pool is now whatever GPU
    // work follows the ring inside `pre[l]`, which is the shared expert and nothing else - 0.038 ms against
    // 0.514 ms of CPU work per layer.  A per-layer CPU expert pool cannot be hidden behind a strictly serial
    // residual chain; the CPU term is answered by Phase 3's VRAM expert cache, not by this pipeline.

    // ---- the graphs
    if (!o.no_capture && !native_pack) {   // plan v0.3 P6: a native pack runs verify windows only
        if (!strata::core::session_capture(wt, g, ss, d_parts, gr, err, /*split=*/false, 0, -1)) {
            return fail(1, "strata generate: session_capture: %s\n", err.c_str());
        }
    }

    // **`--no-capture` AND THE EXPERTS ARE MUTUALLY EXCLUSIVE, AND SILENTLY SO.**
    //
    // The CPU expert pool is wired into `session_loop` - the host loop around the captured graphs - and
    // `session_token` has no pool hook at all.  So `--no-capture` did not merely change HOW the layers were
    // launched: it ran the whole model with `parts` left at whatever the buffer held, which is ZERO, and the
    // only symptom was `expert blobs 0` in a stats line nobody had to read.  A run that silently omits the
    // routed experts is not a slow measurement of this model, it is a measurement of a different model.
    //
    // Refusing is the fix.  `--no-pool` is the explicit way to say "I want the GPU-only floor".
    if (o.no_capture && !o.no_pool) {
        return fail(2,
                    "strata generate: --no-capture runs `session_token`, which has NO CPU expert pool hook, so "
                    "the routed experts would silently contribute nothing. Pass --no-pool as well if the "
                    "GPU-only floor is what you want.\n");
    }

    mem_mark("the expert cache and the graphs");
    std::fprintf(stderr, "strata generate: session is up (engine %s)\n", STRATA_VERSION);
    if (cudaMalloc(&d_emb, (size_t) g.n_embd * 4) != cudaSuccess) {
        return fail(1, "strata generate: the embedding buffer failed\n");
    }
    // **`sample_tokens` TAKES DEVICE POINTERS.**  It is a kernel launch; `logits` and `out` are both read and
    // written on the device.  Passing `logits.data()` - the host vector - faults inside the kernel and the
    // error surfaces at the NEXT synchronising call, which here was the next token's `embed_row`, reporting an
    // illegal access on a weight plane.  Nothing in the parameter names said device.
    if (cudaMalloc(&d_next, sizeof(int)) != cudaSuccess) {
        return fail(1, "strata generate: the sampler output buffer failed\n");
    }

    // ---- **THE TOKEN PATH ALLOCATES NOTHING (P2.T10, review finding H3).**
    //
    // `session_loop` used to allocate its pinned staging buffer, its probe event and its host pin ON EVERY
    // TOKEN, and `cudaFreeHost` at the end of each call implicitly synchronises the device - so every token
    // finished with a device-wide sync nobody asked for.  The scratch is created once here and reused; it also
    // owns the host pin for the whole session rather than taking and releasing it per token.
    scratch_free.p = &loop_scratch;
    // Initialised unconditionally, including under --no-pool: the loop validates the scratch it is handed, so
    // passing a default-constructed one is an error rather than a fallback.  (It was, and the guard caught it -
    // which is the point of the guard.)  One allocation at setup either way.
    if (!loop_scratch.init((size_t) K * g.n_embd * 4, err)) {
        return fail(1, "strata generate: %s\n", err.c_str());
    }
    // Plan v0.3 P3: the whole token as ONE graph whenever nothing needs a host step between the ring and post[l]
    // (the VRAM expert tier and the per-layer dumps do).  `--no-token-graph` keeps two graphs per layer.
    tgraph_free.p = &tgraph;
    // Plan v0.3 P4: with a PROFILE-filled cache the residency is static, so the hit decision moves onto the
    // device and the token graph keeps it.  (A cache filled on demand still needs the per-layer host path.)
    // The residency table is read by kernels on non-blocking streams, which do not wait for the legacy stream a plain
    // cudaMemcpy runs on, and a pageable copy can return before its DMA has landed: each upload waits for its own copy
    // (the current device's legacy stream, stream 0 - nothing else; HIP has no cudaStreamLegacy name, #550).
    res_put = [&host_res = host_res](int32_t* dst) -> cudaError_t {
        const cudaError_t e = cudaMemcpy(dst, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
        return e != cudaSuccess ? e : cudaStreamSynchronize(nullptr);
    };
    graph_hits = hit_fn != nullptr && !profile.empty() && !o.no_pool;
    if (graph_hits && !o.no_capture && !o.no_token_graph) {
        host_res.assign((size_t) (g.n_layers * g.n_expert), strata::core::kNotResident);
        int64_t resident = 0;
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const int32_t slot = xcache.slot_of(l, e);
                host_res[(size_t) (l * g.n_expert + e)] = slot;
                if (slot != strata::core::kNotResident) ++resident;
            }
        if (cudaMalloc((void**) &d_res, host_res.size() * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_count, sizeof(int32_t)) != cudaSuccess ||
            res_put(d_res) != cudaSuccess) {
            return fail(1, "strata generate: the device residency table could not be staged\n");
        }
        thits.d_res = d_res;
        thits.n_expert = g.n_expert;
        // a file-backed arena (STRATA_ARENA_MMAP): the experts no GPU holds are the ones the CPU pool and the
        // prompt path will read - start reading them now instead of faulting them in 4 KB at a time mid-request
        // ... and the ones a GPU holds are handed back first (STRATA_ARENA_RELEASE=0 keeps them): the fill read the
        // whole arena, and left mapped and referenced it crowds every other allocation into swap
        {
            static const bool keep = std::getenv("STRATA_ARENA_RELEASE") && std::getenv("STRATA_ARENA_RELEASE")[0] == '0';
            rss_probe("before the release");
            uint64_t released = 0;
            if (!keep)
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] != strata::core::kNotResident)
                        released += srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);
            if (released > 0)
                std::fprintf(stderr, "strata generate: expert arena: %.2f GiB of VRAM-held experts handed back to the OS\n",
                             (double) released / (1024.0 * 1024.0 * 1024.0));
            rss_probe("after the release");
            int64_t pf = 0;
            for (size_t i = 0; i < host_res.size(); ++i)
                if (host_res[i] == strata::core::kNotResident) {
                    srcp->prefetch((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);
                    ++pf;
                }
            (void) pf;
            rss_probe("after the prefetch hints");
        }
        thits.cache_base = drive.d.cache_base;
        thits.blob = drive.d.cache_blob;
        thits.d_slot = drive.d.d_slot;
        thits.d_dst = drive.d.d_dst;
        thits.d_count = d_hit_count;
        thits.x_q8 = drive.d.x_q8_0_hit;
        thits.x_scale = drive.d.x_q8_0_hit_scale;
        thits.scratch = drive.d.hit_scratch;
        thits.hit_out = drive.d.hit_out;
        drive.d.host_res = host_res.data();
        std::fprintf(stderr, "strata generate: token graph hit path: %lld resident experts, decided on the device\n",
                     (long long) resident);
    }
    if (!o.no_capture && !o.no_token_graph && (hit_fn == nullptr || thits.on()) && !native_pack) {
        if (!strata::core::session_capture_token(wt, g, ss, d_parts, loop_scratch.y_miss, loop_scratch.parts_bytes,
                                                 tgraph, err, thits.on() ? &thits : nullptr)) {
            return fail(1, "strata generate: %s\n", err.c_str());
        }
        std::fprintf(stderr, "strata generate: token graph captured (48 layers, one launch per token)\n");
    }

    // ================================ plan v0.3 P8: THE PERSISTENT ENGINE (--serve) ================================
    //
    // The weights, the expert arena and the VRAM tier load once; then requests arrive on stdin, one per line,
    //
    //     GEN <max_new> <id,id,...>
    //
    // and each generated token is written to stdout as `T <id>` as soon as its verify window is done, followed by
    //
    //     DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <stop|length|cancel|nonfinite> <drafts accepted>
    //          <drafts offered> <prompt tokens reused> ... <prompt tokens read>   (see the DONE line below; #471)
    //
    // `nonfinite` (#879): a window's head logits held a NaN or an infinity (Verifier::nonfinite_row).  None of that
    // window's tokens was written; every state that may carry the poison (the live session, the checkpoints) is
    // dropped at the next request's start, which reads its prompt from token 0.
    //
    // Before that, `RESUME <n>` (n prompt tokens are not read again), `PP <position> <prompt_tokens> <ms> <tok/s>`
    // after every prompt chunk, and `REUSED <n>` once the prompt is read.  (`ERR <message>` instead when a request
    // cannot run; `STOP` ends the running request at its next step; `QUIT` ends the process.)  A request continues
    // from the live session or the longest conversation checkpoint its prompt starts with (see ConvCheckpoint),
    // otherwise from an empty sequence (`session_zero`); the rest of the prompt goes through the batched prompt path
    // and its last token through the first verify window - the path all three model files share.  Decoding is greedy.
    // The expert-cache slots (from the end of the cache) that hold the prompt path's buffers for a chunk, and the
    // bytes from the first of them to the end.
    // the share of expert bytes the arena could pin (sizes the prompt path's streamed ring and its lend cap)
    if (srcp != nullptr && o.prefill_chunk > 0) {
        uint64_t pinned = 0, total = 0;
        const auto& lay = strata::kernels::cpu::expert_layout();
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const uint64_t b = lay.blob_bytes(l);
                total += b;
                if (srcp->pinned(l, e)) pinned += b;
            }
        double share = total ? (double) pinned / (double) total : 1.0;
        if (const char* t = std::getenv("STRATA_TEST_PINNED_SHARE")) share = std::atof(t);   // tests: the Windows-like ring rule
        strata::prefill::Prefill::set_pinned_share(share);
    }
    // bytes -> slots for CUDA0's cache: exact when it knows its per-slot offsets (a native pack's blobs differ
    // per layer), otherwise max_blob each.
    auto slots_from_bytes = [&](uint64_t need) -> int64_t {
        const int64_t blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
        int64_t k = (int64_t) ((need + (uint64_t) blob - 1) / (uint64_t) blob);
        if (xcache.slot_offsets() != nullptr) {   // sized slots: take slots from the end until they hold `need`
            k = 0;
            while (k < xcache.slots() &&
                   (uint64_t) (xcache.bytes() - (int64_t) xcache.slot_offsets()[xcache.slots() - k]) < need) ++k;
        }
        return k;
    };
    // ...and its inverse, for the ring's room: the bytes the cache's last `k` slots hold.  k can arrive <= 0: the
    // caller's lend budget is `min(slots - 128, pct * slots / 100)` and goes negative on a cache smaller than 128
    // slots, where `slots - k` would index slot_offsets() past its end.  Such a cache lends nothing, so say so.
    auto bytes_from_slots = [&](int64_t k) -> uint64_t {
        if (k <= 0) return 0;
        if (xcache.slot_offsets() != nullptr)
            return (uint64_t) (xcache.bytes() - (int64_t) xcache.slot_offsets()[xcache.slots() - k]);
        return (uint64_t) k * (uint64_t) strata::kernels::cpu::expert_layout().max_blob;
    };
    auto lend_slots = [&](int64_t c) -> int64_t {
        return slots_from_bytes(strata::prefill::Prefill::bytes_needed(g, ss, c, srcp != nullptr));
    };
    // `lend_bytes` went with the single-cache serve loan: a participant's loan is priced by `part_bytes` from its
    // OWN cache, and the only other user of the old helper was the serve path's own relayout.
    request_chunk = [](int64_t tokens, int64_t max_chunk) -> int64_t {
        if (tokens <= 0 || max_chunk <= 0) return 0;
        const int64_t rounded = tokens > std::numeric_limits<int64_t>::max() - 255
                                    ? tokens
                                    : ((tokens + 255) / 256) * 256;
        return std::min(max_chunk, rounded);
    };
    // STRATA_PREFILL_EQUAL=1 (opt-in, PR #693): a segment of `tokens` reads in n = ceil(tokens / max_chunk) chunks of EQUAL
    // size (ceil(tokens / n) rounded up to 256) instead of full chunks and a short last one.  A chunk of
    // stream_all_min_tokens() or more streams every expert the GPU does not hold, whatever its length, so a prompt's cost
    // is its number of such chunks and equal chunks borrow no more slots than that count needs.  A last chunk below
    // stream_all_min_tokens() stays as it is (it moves only the experts its own tokens route to).  Used ONLY where a
    // segment's chunk is chosen (the serve lend).  The chunk geometry changes the rounding, so it is off by default.
    static const bool equal_chunks = [] { const char* e = std::getenv("STRATA_PREFILL_EQUAL"); return e != nullptr && e[0] == '1'; }();
    equal_chunk = [&](int64_t tokens, int64_t max_chunk) -> int64_t {
        const int64_t plain = request_chunk(tokens, max_chunk);
        if (!equal_chunks || tokens <= 0 || max_chunk <= 0) return plain;
        const int64_t n = tokens / max_chunk + (tokens % max_chunk != 0);
        const int64_t last = tokens - (n - 1) * max_chunk;
        if (n <= 1 || last < strata::prefill::Prefill::stream_all_min_tokens()) return plain;
        const int64_t per = tokens / n + (tokens % n != 0);
        const int64_t rounded = per > std::numeric_limits<int64_t>::max() - 255 ? per : ((per + 255) / 256) * 256;
        return std::min(max_chunk, rounded);
    };
    // The prompt path's chunk and the slots it borrows for its buffers: the requested chunk halved until it fits,
    // or with --prefill auto the largest chunk whose buffers take at most kAutoLendPct % of the slots (a
    // lent slot's expert is streamed during the prompt and refilled after it; measured on a 12 GB card, 32K Q2_0
    // prompt: 4096 791 tok/s, 6144 878, 8192 973 with 69% of the slots lent).  A request lends only what its own
    // prompt needs (Prefill::relayout), so a big chunk costs short prompts nothing.  0 = none fits.
    // at 8192-token chunks nearly every expert streams anyway, so a lent slot costs little: 90% when the
    // copies are DMA from pinned RAM (Q2_0 8192 + a 384-slot ring: 1283 tok/s), 85% when host copies are the
    // limit (lending more only streams more through them).  STRATA_PREFILL_LEND_PCT overrides (tuning).
    kAutoLendPct = [] {
        const char* v = std::getenv("STRATA_PREFILL_LEND_PCT");
        return v ? (int64_t) std::atoi(v)
                 : (int64_t) (strata::prefill::Prefill::pinned_share() >= 0.9 ? 90 : 85);
    }();
    // The largest chunk --prefill auto may pick: the operator's ceiling (--prefill auto:N) and never past the
    // context, but a bare `auto` always reaches 8192.  32768 and 16384 stay opt-in (#282): a 32K prompt with
    // IQ2_XS (RTX 5090, 64K context) read at 5,624 tok/s in 8192-token chunks and 6,465 in one 32768 chunk (40K
    // -> 18K experts streamed; an NVFP4 pack at 262K: 3,535 -> 5,201).
    auto_ceiling = std::max<int64_t>(8192, std::min<int64_t>(o.prefill_auto_max, o.max_context));
    auto plan_lend = [&](int64_t& chunk) -> int64_t {
        auto slots_for = lend_slots;
        // 0.1.39's list with its ring (also the size up to which a prompt keeps that ring under #583)
        auto old_rule = [&]() -> int64_t {
            strata::prefill::Prefill::set_ring_budget(0, 0);
            static constexpr int64_t kAutoChunks[] = {32768, 16384, 8192, 6144, 4096, 3072, 2048, 1024, 512, 256};
            for (const int64_t c : kAutoChunks) {
                // above 8192: only when asked for, and only when a prompt of the context can use it
                if (c > 8192 && (c > o.prefill_auto_max || c > o.max_context)) continue;
                const int64_t k = slots_for(c);
                if (k + 128 <= xcache.slots() && k * 100 <= kAutoLendPct * xcache.slots()) return c;
            }
            return 0;
        };
        if (o.prefill_auto && !strata::prefill::Prefill::ring_bytes_enabled()) {   // STRATA_RING_BYTES=0: 0.1.39's list
            const int64_t c = old_rule();
            if (c > 0) { chunk = c; return slots_for(c); }
            return 0;
        }
        if (o.prefill_auto) {   // the default since 0.1.39b (#583)
            // The chunk and the ring are one budget, and the ring is the better buy.  Measured on this 4-way
            // IQ3_S rig (2x RTX 3060 + 2x RTX 5060, CUDA3 lending at its 90% cap) on a 120K prompt: 8960 tokens
            // with the 17-slot ring that leaves reads at 963 tok/s, 8192/130 at 1,008, 7168 with the ring full at
            // its 199-slot byte budget at 1,000, and 5632/199 at 915.  A ring slot is worth ~0.53 tok/s there and
            // a chunk token ~0.05, so the 69 slots between a full ring and 8192's 130 are worth more than the
            // 512 chunk tokens they cost - and once the ring IS full, shrinking the chunk further buys nothing.
            // So: the largest chunk that still leaves the ring full.  Only a rig where no chunk at all can afford
            // one falls back to the old rule, which takes the largest chunk whose ring clears kRingMin (the
            // value at or below which ring_slots() returns STAGE and streaming is off).
            constexpr int64_t kRingMin = 16;
            const int64_t small = old_rule();   // 0.1.39's chunk (and its ring for the prompts that fit it)
            const int64_t ring_max = strata::prefill::Prefill::ring_cap_for(small);
            const int64_t budget = std::min(xcache.slots() - 128, kAutoLendPct * xcache.slots() / 100);
            // The room is a BYTE budget.  A ring slot is max_blob bytes (`carve` lays out one whole blob each),
            // while these cache slots hold their own layer's blob, which is smaller than max_blob unless the cache
            // happens to hold the pack's biggest layer - on the 4-way IQ3_S rig that is 2.15 MiB a slot against a
            // 2.54 MiB max_blob.  Counting the room in slots priced the ring at 0.85x what it costs, `fits` then
            // rejected the whole chunk instead of shrinking the ring, and the scan stopped a step short: measured
            // against chunk 6144/ring 199, 8192 was refused where the same bytes afford a ~87-slot ring.
            const int64_t blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
            const uint64_t avail = bytes_from_slots(budget);
            auto room_of = [&](int64_t t) -> int64_t {
                const uint64_t nonring = strata::prefill::Prefill::bytes_needed_no_ring(g, ss, t, srcp != nullptr);
                return std::min((int64_t) ((avail - std::min(avail, nonring)) / (uint64_t) blob), ring_max);
            };
            // `room_of` is capped at ring_max, so "the ring is full" is exactly room == ring_max, and both that
            // test and the ones below it only get harder as t grows - the bisection stays valid.
            auto scan = [&](int64_t floor) -> int64_t {
                return biggest_chunk(auto_ceiling, [&](int64_t t) -> bool {
                    const int64_t room = room_of(t);
                    if (room < floor) return false;
                    strata::prefill::Prefill::set_ring_budget((int) room, 0);
                    // ring_slots() now returns `room` - unless STRATA_PREFILL_RING overrides it, in which case the
                    // chunk has to fit THAT ring, which is the pre-fix rule and the A/B arm
                    const int64_t k = slots_for(t);
                    return k + 128 <= xcache.slots() && k * 100 <= kAutoLendPct * xcache.slots();
                });
            };
            int64_t c = scan(ring_max);
            if (c == 0 && ring_max < strata::prefill::Prefill::ring_default_slots()) c = scan(kRingMin);
            // a prompt that fits 0.1.39's chunk keeps 0.1.39's ring (one chunk: the smaller ring only slowed it)
            if (small >= c) {   // the scan bought nothing: 0.1.39's choice
                strata::prefill::Prefill::set_ring_budget(0, 0);
                if (small > 0) { chunk = small; return slots_for(small); }
                return 0;
            }
            strata::prefill::Prefill::set_ring_budget((int) room_of(c), small);
            chunk = c;
            return slots_for(c);
        }
        for (int64_t c = chunk; c >= 256; c /= 2) {
            const int64_t k = slots_for(c);
            if (k + 128 <= xcache.slots()) { chunk = c; return k; }
        }
        return 0;
    };
    // ---- THE ELASTIC K/V (--kv-grow; vmm.hpp).  Allocated for the whole --max-context up front, the K/V of a 262K
    // context (3.35 GiB with the drafter's at int8) takes VRAM the expert cache could hold more experts in.  Now the K/V pools hold physical memory only for the cells
    // the requests reach.  When a request needs more, slots just below the prompt path's loan give up their experts
    // (the CPU computes those from RAM, like any miss) and their whole chunks are mapped into the K/V; a later short
    // request hands them back and the slots are refilled with the profile's hottest experts.  Neither side's
    // addresses move, so every captured graph stays valid.  The context is never smaller: --max-context cells always
    // fit, the cache simply holds fewer experts while a long one is live.
    kvg_start = [&](int64_t top) {
        kvg.on = strata::core::qsa_kv_elastic() && xcache.vmm_range() != nullptr && d_res != nullptr &&
                 !host_res.empty() && srcp != nullptr && top > kvg.floor;
        if (!kvg.on) return;
        kvg.top = kvg.lo = top;
        kvg.cells = strata::core::qsa_kv_elastic_cells();
        if (const char* v = std::getenv("STRATA_KV_GROW_STEP"); v != nullptr && std::atoll(v) > 0) kvg.step = std::atoll(v);
        // tests: the cache keeps this many slots (its size: the K/V grows into new VRAM only)
        if (const char* v = std::getenv("STRATA_KV_GROW_FLOOR"); v != nullptr && std::atoll(v) > 0) kvg.floor = std::atoll(v);
        std::fprintf(stderr, "strata generate: elastic K/V: %lld of %lld cells in VRAM (%.2f of %.2f GiB); the expert "
                             "cache (%lld slots) gives the K/V room below slot %lld as the context grows\n",
                     (long long) kvg.cells, (long long) o.max_context,
                     (double) strata::core::qsa_kv_elastic_mapped_bytes() / 1073741824.0,
                     (double) strata::core::qsa_kv_elastic_full_bytes() / 1073741824.0, (long long) xcache.slots(),
                     (long long) top);
    };
    // Room for `cells` cells (rounded up to a step).  `quiesce` must leave nothing running on the device and land
    // the adaptive tier's swaps (a swap in flight could still be writing a slot given up here).
    kvg_ensure = [&](int64_t cells, const std::function<void()>& quiesce) -> bool {
        if (!kvg.on || cells <= kvg.cells) return true;
        const int64_t target = std::min<int64_t>(o.max_context, (cells + kvg.step - 1) / kvg.step * kvg.step);
        const int64_t need = strata::core::qsa_kv_elastic_need(target);
        if (need == 0) { kvg.cells = strata::core::qsa_kv_elastic_cells(); return true; }
        quiesce();
        strata::core::VmmRange& r = *xcache.vmm_range();
        const uint64_t G = strata::core::vmm_granularity();
        std::vector<int32_t> owner;   // slot -> residency index
        // The slots given up are the ones just below the prompt path's loan, and the loan is most of the cache: they
        // hold experts of middling heat.  So an expert there that is hotter than the coldest one in the loan region
        // moves into that one's slot (a device copy), and the coldest is given up instead - the cache then holds what
        // a smaller cache would.  Heat: the adaptive tier's routing counts, then the profile's rank.
        std::vector<int32_t> rank;
        std::vector<std::pair<float, int32_t>> cold;   // the loan region's residents, hottest first
        auto heat = [&](int32_t i) -> float {
            const float u = drive.d.usage.empty() ? 0.0f : drive.d.usage[(size_t) i];
            return u * 1048576.0f - (float) rank[(size_t) i];
        };
        const auto& lay = strata::kernels::cpu::expert_layout();
        int64_t gave = 0, moved = 0;
        auto freeable = [&]() -> int64_t {   // mapped chunks wholly inside [lo, top): a chunk a live slot shares stays
            const int64_t c0 = (int64_t) ((xcache.slot_offset(kvg.lo) + G - 1) / G);
            const int64_t c1 = (int64_t) (xcache.slot_offset(kvg.top) / G);
            int64_t k = 0;
            for (int64_t c = c0; c < c1; ++c) k += r.mapped(c) ? 1 : 0;
            return k;
        };
        while ((int64_t) kvg.spare.size() + freeable() < need && kvg.lo > kvg.floor) {
            if (owner.empty()) {
                owner.assign((size_t) xcache.slots(), -1);
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] >= 0 && host_res[i] < xcache.slots()) owner[(size_t) host_res[i]] = (int32_t) i;
                rank.assign(host_res.size(), (int32_t) profile.size());
                for (size_t r = 0; r < profile.size(); ++r)
                    rank[(size_t) profile[r].first * (size_t) g.n_expert + (size_t) profile[r].second] = (int32_t) r;
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] >= kvg.top) cold.emplace_back(heat((int32_t) i), (int32_t) i);
                std::sort(cold.begin(), cold.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            }
            --kvg.lo;
            ++gave;
            if (const int32_t oi = owner[(size_t) kvg.lo]; oi >= 0) {
                const int64_t layer = oi / g.n_expert;
                int32_t vi = -1;
                if (!cold.empty() && cold.back().first < heat(oi)) {
                    vi = cold.back().second;
                    const int32_t vs = host_res[(size_t) vi];
                    if (xcache.slot_offset(vs + 1) - xcache.slot_offset(vs) < lay.blob_bytes(layer) ||
                        cudaMemcpy(xcache.device_slot(vs), xcache.device_slot((int32_t) kvg.lo),
                                   (size_t) lay.blob_bytes(layer), cudaMemcpyDeviceToDevice) != cudaSuccess)
                        vi = -1;
                    else {
                        cold.pop_back();
                        host_res[(size_t) oi] = vs;
                        host_res[(size_t) vi] = strata::core::kNotResident;
                        ++moved;
                    }
                }
                if (vi < 0) host_res[(size_t) oi] = strata::core::kNotResident;
                ++kvg.evicted;
            }
        }
        // a device-to-device cudaMemcpy does not wait for the copy: the moves still read the slots given up, and a
        // chunk unmapped under a copy faults it (the error then sticks to whatever runs next)
        if (gave > 0 && cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "strata: moving experts for the K/V failed: %s\n", cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        {
            const int64_t c0 = (int64_t) ((xcache.slot_offset(kvg.lo) + G - 1) / G);
            const int64_t c1 = (int64_t) (xcache.slot_offset(kvg.top) / G);
            for (int64_t c = c0; c < c1; ++c)
                if (r.mapped(c))
                    if (const strata::core::VmmChunk h = r.unmap(c)) kvg.spare.push_back(h);
        }
        if (gave > 0 &&
            cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::fprintf(stderr, "strata: the residency table upload failed: %s\n", cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        int64_t fresh = 0;
        const bool ok = strata::core::qsa_kv_elastic_grow(target, [&]() -> strata::core::VmmChunk {
            if (kvg.spare.empty()) { ++fresh; return 0; }   // the cache is at its floor: new memory
            const strata::core::VmmChunk h = kvg.spare.back();
            kvg.spare.pop_back();
            return h;
        });
        kvg.fresh += fresh;
        kvg.cells = strata::core::qsa_kv_elastic_cells();
        ++kvg.grows;
        std::fprintf(stderr, "strata: K/V grown to %lld cells (%.2f GiB); the expert cache gave %lld slots for it, "
                             "%lld hotter experts moved to colder ones' slots (%lld of %lld slots hold experts)%s\n",
                     (long long) kvg.cells, (double) strata::core::qsa_kv_elastic_mapped_bytes() / 1073741824.0,
                     (long long) gave, (long long) moved, (long long) (kvg.lo + (xcache.slots() - kvg.top)),
                     (long long) xcache.slots(),
                     fresh > 0 ? " - and new VRAM, the cache being at its floor" : "");
        if (!ok)
            std::fprintf(stderr, "strata: the K/V could not grow to %lld cells (%s)\n", (long long) target,
                         cudaGetErrorString(cudaGetLastError()));
        return ok;
    };
    // A request that needs far fewer cells than the K/V holds gives the rest back: the slots refill with the
    // profile's hottest experts the GPU does not hold.  Run on a quiet device (as kvg_ensure's `quiesce`).
    kvg_trim = [&](int64_t cells) -> bool {
        if (!kvg.on) return true;
        const int64_t want = cells;
        const int64_t target = std::max<int64_t>(kvg.step, (want + kvg.step - 1) / kvg.step * kvg.step);
        if (kvg.cells < target + 2 * kvg.step || kvg.lo >= kvg.top) return true;
        strata::core::qsa_kv_elastic_shrink(target, [&](strata::core::VmmChunk h) { kvg.spare.push_back(h); });
        kvg.cells = strata::core::qsa_kv_elastic_cells();
        strata::core::VmmRange& r = *xcache.vmm_range();
        const uint64_t G = strata::core::vmm_granularity();
        const int64_t c1 = (int64_t) (xcache.slot_offset(kvg.top) / G);
        {   // one run from the lowest unmapped chunk, as long as the chunks handed back last (one access call)
            const int64_t c0 = (int64_t) (xcache.slot_offset(kvg.lo) / G);
            int64_t cend = c0, need = (int64_t) kvg.spare.size();
            while (cend < c1 && need > 0) need -= r.mapped(cend++) ? 0 : 1;
            if (!r.map_range(c0, cend, [&]() -> strata::core::VmmChunk {
                    if (kvg.spare.empty()) return 0;
                    const strata::core::VmmChunk h = kvg.spare.back();
                    kvg.spare.pop_back();
                    return h;
                }))
                std::fprintf(stderr, "strata: the expert cache could not take its VRAM back from the K/V\n");
        }
        const int64_t lo0 = kvg.lo;
        while (kvg.lo < kvg.top) {   // a slot whose chunks are all mapped again holds an expert again
            const int64_t a = (int64_t) (xcache.slot_offset(kvg.lo) / G);
            const int64_t b = (int64_t) ((xcache.slot_offset(kvg.lo + 1) + G - 1) / G);
            bool all = true;
            for (int64_t c = a; c < b && all; ++c) all = r.mapped(c);
            if (!all) break;
            ++kvg.lo;
        }
        // the refills on one stream and one sync (a blocking copy each was a driver round trip per expert)
        const auto& lay = strata::kernels::cpu::expert_layout();
        std::vector<std::pair<size_t, int32_t>> filled;   // (residency index, slot), resident once the copies landed
        size_t pi = 0;
        for (int64_t s = lo0; s < kvg.lo; ++s) {
            const uint64_t room = xcache.slot_offset(s + 1) - xcache.slot_offset(s);
            for (; pi < profile.size(); ++pi) {
                const int32_t l = profile[pi].first, e = profile[pi].second;
                const size_t i = (size_t) l * (size_t) g.n_expert + (size_t) e;
                if (host_res[i] >= 0 || lay.blob_bytes(l) > room) continue;
                const uint8_t* b = srcp->blob_stable(l, e);
                if (b == nullptr) continue;
                cudaMemcpyAsync(xcache.device_slot((int32_t) s), b, (size_t) lay.blob_bytes(l), cudaMemcpyHostToDevice,
                                nullptr);
                filled.emplace_back(i, (int32_t) s);
                ++pi;
                break;
            }
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "strata: refilling the expert cache failed: %s\n", cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        for (const auto& [i, s] : filled) host_res[i] = s;
        kvg.refilled += (int64_t) filled.size();
        // res_put waits for its copy: the next reader can be a verify window (a non-blocking stream) right away
        if (res_put(d_res) != cudaSuccess) return false;
        for (const strata::core::VmmChunk h : kvg.spare) strata::core::vmm_chunk_free(h);   // new ones, if any
        kvg.spare.clear();
        ++kvg.trims;
        std::fprintf(stderr, "strata: K/V trimmed to %lld cells; %lld slots back to the expert cache, refilled from "
                             "the profile\n", (long long) kvg.cells, (long long) (kvg.lo - lo0));
        return true;
    };
    // ---- the resident RAM mode (--resident-experts / --resident-cpu-experts): the experts the GPU cache does not
    // hold are copied from experts.bin into RAM once, so no decode or prompt step reads the file (the plain mmap
    // mode reads them through the OS file cache, which a small-RAM PC keeps giving back to the SSD).  Built here,
    // after the prompt path's lend plan is known: the slots it may lend (the cache's last ones) have their experts
    // streamed during a prompt and copied back after it, so those are kept in RAM too as far as RAM allows.  The
    // bytes are the file's bytes and the placement is the same, so the answers are the plain mmap mode's; the
    // share-of-pinned figure above (which sizes the prompt path) is left as the mmap mode's for the same reason.
    // --adapt-async (opt-in): where the asynchronous adaptive tier does not apply, the blocking tier runs, said once.
    // It exchanges experts between the GPU cache and the resident RAM copy only (the copy is checked once it is
    // built, below).
    auto adapt_async_off = [&](const char* why) {
        std::fprintf(stderr, "strata generate: --adapt-async 1 is off (%s): the blocking adaptive tier\n", why);
        o.adapt_async = 0;
    };
    if (o.adapt_async) {
        const char* why = o.adapt_every <= 0 || o.adapt_swaps <= 0 ? "the adaptive tier is off"
                        : !o.resident_cpu_experts && srcp != &arena_src ? "it needs the resident RAM mode, --resident-experts, or the expert arena"
                        : nullptr;
        if (why != nullptr) adapt_async_off(why);
    }
    if (o.resident_cpu_experts) {
        int64_t lend_from = -1;
        if (o.prefill_chunk > 0 && !o.no_prefill_borrow && d_res != nullptr && xcache.slots() > 0) {
            int64_t chunk = o.prefill_chunk;
            const int64_t k = plan_lend(chunk);
            if (k > 0) lend_from = xcache.slots() - k;
        }
        bool resident_ok = src.pin_cache_complement(xcache, err, o.resident_pin, {}, lend_from,
                                                    o.resident_headroom, o.resident_budget, &profile);
        std::string whole_err;
        if (!resident_ok && o.resident_soft) {
            // #467: the whole complement does not fit - keep what does, the hottest by the profile, through the #403
            // budget path (sized by available memory) instead of none: the misses outside it read the same file bytes
            // the mmap fallback reads, so the answers are unchanged.  Nothing pinned: the old fallback below.
            whole_err = err;
            resident_ok = src.pin_cache_complement(xcache, err, o.resident_pin, {}, -1, o.resident_headroom,
                                                   strata::core::FileExpertSource::kResidentWhatFits, &profile);
            if (resident_ok)
                std::fprintf(stderr, "strata generate: WARNING: the whole resident RAM mode does not fit (%s); %.2f "
                                     "GiB of the experts the GPU does not hold, the hottest by the expert profile, are "
                                     "kept in RAM and the rest are read from the model folder through the OS file "
                                     "cache\n",
                             whole_err.c_str(), (double) src.resident_bytes() / 1073741824.0);
            else
                err = whole_err + "; " + err;
        }
        // the startup reads are done: the view of experts.bin back, the file tier as chosen above
        bool startup_kept = false;
        if (src.startup_unbuffered()) {
            std::string why;
            const auto end_t0 = std::chrono::steady_clock::now();
            startup_kept = !src.end_startup_unbuffered(why);
            if (startup_kept) std::fprintf(stderr, "strata generate: WARNING: %s\n", why.c_str());
            else
                std::fprintf(stderr, "strata generate: the startup reads are done: experts.bin mapped again (%.0f ms)\n",
                             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - end_t0).count());
        }
        // #669 / #765: a big chunk lends many cache slots to the prompt path, and each lent slot's expert is read back
        // from the SSD whenever the RAM could not keep its copy (31 GB: auto:32768 read prompts ~3x slower than auto).
        // Said, never capped: --prefill is the user's, and with the RAM for them it is the faster one (+21-35%).
        // #1683: with a RAM budget the lend region is kept after the ranked experts, as far as the budget reaches, so
        // the advice there is a larger budget, not a smaller chunk (the budget used to keep none of it whatever the chunk)
        if (resident_ok && lend_from >= 0 && (o.prefill_chunk > 8192 || o.resident_budget > 0) && xcache.slots() > lend_from &&
            src.resident_lent_slots() < xcache.slots() - lend_from)
            std::fprintf(stderr, "strata generate: WARNING: a prompt chunk of %lld tokens lends %lld cache slots to the "
                                 "prompt path, but only %lld of their experts fit in RAM: the others are read from the "
                                 "SSD on every chunk, and prompts can read ~3x slower than with --prefill auto (#669). "
                                 "%s keeps them all in RAM\n",
                         (long long) o.prefill_chunk, (long long) (xcache.slots() - lend_from),
                         (long long) src.resident_lent_slots(),
                         o.resident_budget > 0 ? "A larger --resident-budget-gib (the lend region comes after the ranked experts), or a smaller --prefill,"
                                               : "A smaller --prefill, or auto,");
        if (resident_ok) {
            // --adapt-async: twice the exchange buffers - the second half bounces the copies in whose source is not
            // page-locked (an asynchronous copy instead of the driver's staged, synchronous one)
            if (o.adapt_async && !src.reserve_exchanges(2 * std::min<int64_t>(o.adapt_swaps, 96), err,
                                                        std::min<int64_t>(o.adapt_swaps, 96)))
                adapt_async_off(err.c_str());
            // pageable buffers would make every copy of a round synchronous (the driver stages them): that is the
            // blocking tier's cost, so the asynchronous tier is not worth running - said, and the blocking tier stays
            if (o.adapt_async && !src.exchange_pinned())
                adapt_async_off("the exchange buffers could not be page-locked");
            if (o.adapt_every > 0 && o.adapt_swaps > 0 &&
                !src.reserve_exchanges(std::min<int64_t>(o.adapt_swaps, 96), err)) {
                return fail(1, "strata generate: CPU expert residency: %s\n", err.c_str());
            }
            std::fprintf(stderr, "strata generate: resident RAM mode: %.2f GiB of experts in RAM (%s), %lld in the GPU "
                                 "cache; adaptive swaps %s\n",
                         (double) src.resident_bytes() / 1073741824.0,
                         src.complement_pinned() ? "page-locked" : src.locked_bytes() > 0 ? "locked" : "pageable",
                         (long long) xcache.resident(),
                         o.adapt_every > 0 && o.adapt_swaps > 0 ? "exchange them with the GPU cache (no file reads)"
                                                                : "off");
        } else if (o.resident_soft) {
            std::fprintf(stderr, "strata generate: WARNING: the resident RAM mode does not fit (%s); the experts the "
                                 "GPU does not hold are read from the model folder through the OS file cache "
                                 "(--mmap-experts), which is slower when the RAM cannot keep them\n", err.c_str());
        } else if (o.resident_budget > 0) {
            // #403: a RAM budget that cannot be kept is not a reason to stop - the experts it would have held are
            // read from the files like the ones outside it (pin_cache_complement leaves nothing half-built)
            std::fprintf(stderr, "strata generate: WARNING: the RAM budget (--resident-budget-gib) cannot be kept (%s); "
                                 "every expert the GPU does not hold is read from the model files %s "
                                 "(--mmap-experts), which is slower\n", err.c_str(),
                         src.unbuffered() ? "with unbuffered I/O" : "through the OS file cache");
        } else {
            return fail(1, "strata generate: CPU expert residency: %s\n", err.c_str());
        }
        // #577: the unbuffered choice above was made before the RAM copy existed, from the budget asked for; now the
        // copy is built, decide again from the RAM it really holds and the expert bytes outside it (on a 96 GB PC
        // the file cache keeps those, and every refill after a prompt read the drive instead).  Windows and Linux: the
        // unbuffered reads exist there
        if (o.mmap_experts && o.resident_budget > 0 && !startup_kept) {
            std::string why;
            const bool was = src.unbuffered();
            const bool ub = src.recheck_unbuffered(why);
            std::fprintf(stderr, "strata generate: the file tier reads %s%s (%s)\n",
                         ub ? "unbuffered" : "through the file cache", ub == was ? "" : " (changed)", why.c_str());
            // MODEL-UPDATE round 2 (opt-in STRATA_PROMPT_DIRECT=1): the prompt path's batched reads unbuffered while
            // decode keeps the file cache (z440, UD-Q4_K_XL, 39 GiB budget: all-unbuffered doubled the 6K prompt but
            // cost decode 9-13%).  The same bytes either way.
            if (!ub && [] { const char* v = std::getenv("STRATA_PROMPT_DIRECT"); return v != nullptr && v[0] == '1'; }()) {
                std::string pw;
                const bool pd = src.open_prompt_direct(pw);
                std::fprintf(stderr, "strata generate: the prompt path's expert reads %s\n",
                             pd ? "go unbuffered (STRATA_PROMPT_DIRECT=1)" : ("stay buffered: " + pw).c_str());
            }
        }
    }
    // (the arena mode has no RAM copy to exchange with: its experts are always in RAM, the tier only copies them in)
    if (o.adapt_async && !src.complement_ready() && srcp != &arena_src) adapt_async_off("the resident RAM mode is not running");
    // pp-opt: with the file tier unbuffered, the mapped view of experts.bin is only a fallback from here on - and while
    // it exists NTFS serves the unbuffered reads one at a time (drop_mapping).  STRATA_KEEP_MAPPING=1 keeps it (A/B).
    if (srcp == &src && src.unbuffered() && std::getenv("STRATA_KEEP_MAPPING") == nullptr) {
        std::string why;
        const bool dropped = src.drop_mapping(why);
        std::fprintf(stderr, "strata generate: the mapped view of the experts %s (%s)\n",
                     dropped ? "is closed" : "stays open", why.c_str());
        // --ple-io mmap / ram maps the PLE shard for the whole run; when that shard also holds experts, NTFS still runs
        // the unbuffered reads of that one file one at a time (the default --ple-io direct keeps no mapping)
        if (dropped && src.gguf_mode() && !o.ple_gguf.empty() && o.ple_io != "direct")
            std::fprintf(stderr, "strata generate: note: --ple-io %s keeps %s mapped; if that shard holds experts, their "
                                 "reads stay serialized (--ple-io direct, the default, keeps no mapping)\n",
                         o.ple_io.c_str(), o.ple_gguf.c_str());
    }
    // 0.1.42 (H11): the default CPU share is armed only when the host has the memory for it.  One line says why not;
    // an explicit STRATA_PREFILL_CPU_SHARE (other than 0) is armed anyway, with a warning when memory is short.
    share_default_ok = [&]() -> bool {
        strata::prefill::share_rules::ShareMemory sm;
        strata::core::detail::HostMemory hm;
        if (strata::core::detail::host_available_memory(hm)) {
            sm.avail_ram = hm.available;
            sm.avail_commit = hm.commit == ~uint64_t{0} ? strata::prefill::share_rules::kUnknown : hm.commit;
        }
        sm.buffer_bytes = (uint64_t) (1024 + 8 * g.n_expert) * (uint64_t) g.n_embd * 4 + (256ull << 20);
        if (srcp == &src) {
            const auto& lay = strata::kernels::cpu::expert_layout();
            for (int64_t l = 0; l < g.n_layers; ++l)
                for (int64_t e = 0; e < g.n_expert; ++e)
                    if (xcache.slot_of(l, e) < 0 && src.resident_blob(l, e) == nullptr && !src.transient(l, e))
                        sm.inplace_bytes += (uint64_t) lay.blob_bytes(l);
        }
        const auto v = strata::prefill::share_rules::default_share_memory_verdict(sm);
        if (v.arm) return true;
        const char* ev = std::getenv("STRATA_PREFILL_CPU_SHARE");
        if (ev == nullptr)
            std::fprintf(stderr, "strata: the CPU prefill share is not armed by default: %s. STRATA_PREFILL_CPU_SHARE=auto "
                                 "turns it on anyway\n", v.why.c_str());
        else if (std::strcmp(ev, "0") != 0)
            std::fprintf(stderr, "strata: WARNING STRATA_PREFILL_CPU_SHARE=%s is set and kept, but %s: the share can slow "
                                 "prompts or fail on a machine this short of memory (STRATA_PREFILL_CPU_SHARE=0 turns it off)\n",
                         ev, v.why.c_str());
        return false;
    };
    if (o.spec < 2 || o.prefill_chunk <= 0 ||
        (graph_hits && (thits.d_res == nullptr || host_res.empty()))) {
        return fail(2, "strata serve: needs --spec T and --prefill CHUNK (and a fillable "
                       "--expert-cache; the graphed hit path additionally needs --expert-profile P)\n");
    }
    return 0;
}

int EngineState::serve_setup() {
    // STRATA_VOLTA_BUILD backport (see docs/ and ninfer-flash-next-v100-backport): --mtp is optional here.
    // Without it, drafting falls back
    // to the suffix/prompt-lookup drafter (DraftPolicy still picks Lookup vs plain per round) or, when
    // neither fires, plain one-token-per-round decoding - slower, never wrong: the verify window still
    // confirms every emitted token against the real model regardless of where the draft came from.
    use_mtp = !o.mtp.empty();
    sp.emplace();
    // the pool is idle while a prompt is read; with STRATA_PREFILL_CPU_SHARE the staged-chunk limit before the
    // chunk below sizes the loans (bytes_needed reads it)
    share_pool = !o.no_pool;
    bool share_has_work = strata::prefill::share_rules::default_share_has_work(
        (int64_t) xcache.slots(), (int64_t) (g.n_layers * g.n_expert));   // #1595
    if (!share_has_work)
        std::fprintf(stderr, "strata serve: every expert is in the GPU cache (%lld of %lld): the CPU prefill share has "
                             "nothing to take, so it is not armed by default (STRATA_PREFILL_CPU_SHARE=auto forces it)\n",
                     (long long) xcache.slots(), (long long) (g.n_layers * g.n_expert));
    if (share_pool && share_has_work && !share_default_ok()) share_has_work = false;   // H11
    strata::prefill::Prefill::arm_cpu_share(share_pool, share_pool && share_has_work);
    // bytes -> slots for one cache: exact when it knows its per-slot offsets (a native pack's blobs differ
    // per layer), otherwise max_blob each.
    cache_slots_for = [this](const strata::core::ExpertCache& xc, uint64_t need) -> int64_t {
        return strata::program::split_lend_slots(xc.slots(), xc.bytes(), xc.slot_offsets(),
                                                 (uint64_t) strata::kernels::cpu::expert_layout().max_blob, need);
    };
    part_slots = [this](const PfPart& p, int64_t c) -> int64_t {
        return cache_slots_for(*p.cache, strata::prefill::Prefill::bytes_needed(g, *p.ses, c, srcp != nullptr));
    };
    part_bytes = [this](const PfPart& p, int32_t first) -> uint64_t {
        strata::core::ExpertCache& xc = *p.cache;
        return xc.slot_offsets() ? (uint64_t) (xc.bytes() - (int64_t) xc.slot_offsets()[first])
                                 : (uint64_t) (xc.slots() - first) *
                                       (uint64_t) strata::kernels::cpu::expert_layout().max_blob;
    };
    // a cache too small to lend the prompt path its buffers would make it allocate them on top - on a card
    // whose cache already filled its reserve, that is the over-subscription the auto sizing avoids - so the
    // chunk is the largest one EVERY participant can lend (a smaller chunk only reads slower)
    if (pf_borrow && d_res != nullptr) {
        pf_parts.push_back({&xcache, &ss, &*sp, -1, 0, g.n_layers, -1, -1, 0, {}});
        // The two tests plan_lend makes for CUDA0 alone, one participant at a time: a loan must leave the
        // 128-slot floor.  The percentage cap is an AUTO-chunk rule and only the auto scan applies it - an
        // explicit --prefill is the operator's number, and a loan of it only has to fit.  With one participant
        // (no split) this reduces to plan_lend exactly, so the single-GPU loan is unchanged from main.
        auto fits_one = [&](const PfPart& p, int64_t c, bool cap) -> bool {
            const int64_t k = part_slots(p, c);
            if (k <= 0 || k + 128 > p.cache->slots()) return false;
            return !(cap && k * 100 > kAutoLendPct * p.cache->slots());
        };
        // `only`: CUDA0's cache alone (#448: what one GPU would choose, for the log below); null: every one
        auto fits = [&](int64_t c, bool cap, const PfPart* only = nullptr) -> bool {
            if (only != nullptr) return fits_one(*only, c, cap);
            for (const PfPart& p : pf_parts)
                if (!fits_one(p, c, cap)) return false;
            return true;
        };
        // The chunk and the ring are one budget.  The ring used to be a fixed 384 slots taken off the top
        // before the chunk was considered: 506 MiB on a Q2_0 pack, where it was tuned, but 1,912 MiB on
        // Q8_0, more than half an 8 GB card's cache - which is what kept that card at a 1024-token chunk.
        // Size the chunk first and hand the ring what the smallest participant leaves over, up to the ring's
        // byte budget - but hand it ALL of it, as in plan_lend: the ring is the better buy per byte (the
        // measurements are there), so the chunk to take is the largest one that still leaves the ring full,
        // not the largest one that leaves it anything at all.  kRingMin is the fallback floor for a rig where
        // no chunk can afford a full ring; ring_slots() returns STAGE at or below it, which turns streaming
        // off.
        constexpr int64_t kRingMin = 16;
        // The room is a BYTE budget, as in plan_lend: a ring slot is max_blob bytes, but a cache slot holds
        // its own layer's blob, which on this rig averages 0.85 of it.  In slots the ring looked 18% cheaper
        // than it is, and a chunk whose ring only fitted after that discount was refused outright - the scan
        // stopped one step short while the chunk it did take had room to spare for the cap-sized ring.
        const int64_t kBlob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
        auto ring_room = [&](int64_t c, const PfPart* only, int64_t cap) -> int64_t {
            int64_t room = cap;
            for (const PfPart& p : pf_parts) {
                if (only != nullptr && &p != only) continue;
                // clamped at 0: a cache under 128 slots has a negative lend budget, and `slots - budget` is
                // then `slots`, which is off_[slots] - the end of the prefix sum, so that cache lends 0 bytes
                // rather than reading one entry past its offsets
                const int64_t budget = std::max<int64_t>(0, std::min(p.cache->slots() - 128,
                                                                    kAutoLendPct * p.cache->slots() / 100));
                const uint64_t avail = part_bytes(p, (int32_t) (p.cache->slots() - budget));
                const uint64_t nonring = strata::prefill::Prefill::bytes_needed_no_ring(g, *p.ses, c, srcp != nullptr);
                room = std::min(room, (int64_t) ((avail - std::min(avail, nonring)) / (uint64_t) kBlob));
            }
            return room;
        };
        // The largest chunk on the 256-token grid that `only` - or, with null, every stage - can lend with the
        // ring still at its full byte budget (ring_room is capped at ring_max, so that is room == ring_max),
        // under the operator's ceiling (auto_ceiling keeps a bare `auto` at 8192, #282).  Failing that, the
        // largest chunk whose ring clears kRingMin.
        // 0.1.39's list with its ring: the chunk it would pick, and the size up to which a prompt keeps that ring
        auto old_pick = [&](const PfPart* only) -> int64_t {
            strata::prefill::Prefill::set_ring_budget(0, 0);
            static constexpr int64_t kAutoChunks[] = {32768, 16384, 8192, 6144, 4096, 3072, 2048, 1024, 512, 256};
            for (const int64_t c : kAutoChunks) {
                if (c > 8192 && (c > o.prefill_auto_max || c > o.max_context)) continue;   // #282, as plan_lend
                    if (fits(c, true, only)) return c;
            }
            return 0;
        };
        auto scan = [&](const PfPart* only) -> int64_t {
            if (!o.prefill_auto) {   // an explicit --prefill is the operator's number, and a loan of it only has to fit
                for (int64_t c = o.prefill_chunk; c >= 256; c /= 2)
                    if (fits(c, false, only)) return c;
                return 0;
            }
            if (!strata::prefill::Prefill::ring_bytes_enabled()) return old_pick(only);   // STRATA_RING_BYTES=0
            const int64_t small = old_pick(only);
            const int64_t cap = strata::prefill::Prefill::ring_cap_for(small);
            auto probe = [&](int64_t floor) -> int64_t {
                return biggest_chunk(auto_ceiling, [&](int64_t t) -> bool {
                    const int64_t room = ring_room(t, only, cap);
                    if (room < floor) return false;
                    strata::prefill::Prefill::set_ring_budget((int) room, 0);
                    // ring_slots() now returns `room` - unless STRATA_PREFILL_RING overrides it, in which
                    // case the chunk has to fit THAT ring, which is the pre-fix rule and the A/B arm
                    return fits(t, true, only);
                });
            };
            int64_t c = probe(cap);
            if (c == 0 && cap < strata::prefill::Prefill::ring_default_slots()) c = probe(kRingMin);
            return std::max(c, small);   // never a smaller chunk than 0.1.39's
        };
        const int64_t chunk = scan(nullptr);
        // the scan left the ring override on its own last trial: put it back on the chunk that won, which
        // is what `init` lays out and what the INFO line reports
        if (chunk > 0 && o.prefill_auto && strata::prefill::Prefill::ring_bytes_enabled()) {
            // a prompt that fits 0.1.39's chunk keeps 0.1.39's ring (one chunk: the smaller ring only slowed
            // it); the byte-budget ring is for the chunks past it
            const int64_t small = old_pick(nullptr);
            strata::prefill::Prefill::set_ring_budget(
                small >= chunk ? 0 : (int) ring_room(chunk, nullptr, strata::prefill::Prefill::ring_cap_for(small)),
                small);
        }
        if (chunk > 0) {
            if (o.prefill_auto)
                // the RESOLVED ring, not the room offered: ring_slots() clamps to [16, RING_MAX] and a chunk
                // under stream_all_min() gets STAGE, and STRATA_PREFILL_RING overrides both.  This is what
                // `init` lays out, and what the INFO line reports to the Monitor tab.
                std::fprintf(stderr, "strata serve: prompt chunk auto: %lld tokens, a %lld-slot ring\n",
                             (long long) chunk,
                             (long long) strata::prefill::Prefill::ring_slots_for(chunk));
            else if (chunk != o.prefill_chunk)
                std::fprintf(stderr, "strata serve: prompt chunk %lld -> %lld tokens so its buffers fit in "
                                     "every expert cache\n", (long long) o.prefill_chunk, (long long) chunk);
            o.prefill_chunk = chunk;
            for (PfPart& p : pf_parts) {
                p.first = (int32_t) (p.cache->slots() - part_slots(p, chunk));
                p.first_now = p.first;
            }
            lend_first = pf_parts[0].first;
            borrow = lend_first >= 0 ? xcache.device_slot(lend_first) : nullptr;
            borrow_bytes = lend_first >= 0 ? part_bytes(pf_parts[0], lend_first) : 0;
        } else if (o.prefill_auto) {
            o.prefill_chunk = 1024;   // nothing lendable: small buffers of its own
        }
    } else if (o.prefill_auto && d_res == nullptr) {
        o.prefill_chunk = 1024;       // #85: no expert cache at all (a full 8 GB card): small buffers of its own
    }
    if (o.prefill_head_alias) {
        for (PfPart& p : pf_parts) {
            p.first = p.cache->slots() > 1 ? 1 : -1;
            p.first_now = p.first;
        }
        lend_first = pf_parts[0].first;
        borrow = lend_first >= 0 ? xcache.device_slot(lend_first) : nullptr;
        borrow_bytes = lend_first >= 0 ? part_bytes(pf_parts[0], lend_first) : 0;
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        static constexpr int64_t candidates[] = {2048, 1536, 1024, 512};
        for (const int64_t candidate : candidates)
            if (strata::prefill::Prefill::bytes_needed(g, ss, candidate) + (512ull << 20) <=
                free_b + borrow_bytes) {
                o.prefill_chunk = candidate;
                break;
            }
        std::fprintf(stderr, "strata serve: prompt chunk head alias: %lld tokens with %lld MiB borrowed and "
                             "%lld MiB free\n",
                     (long long) o.prefill_chunk, (long long) (borrow_bytes >> 20), (long long) (free_b >> 20));
    }
    bool any_loan = borrow != nullptr;
    for (size_t i = 1; i < pf_parts.size(); ++i) any_loan = any_loan || pf_parts[i].first >= 0;
    if (any_loan) {
        if (borrow != nullptr)
            std::fprintf(stderr, "strata serve: the prompt path borrows %lld CUDA0 cache slots (%.2f GiB)\n",
                         (long long) (xcache.slots() - lend_first), (double) borrow_bytes / 1073741824.0);
    } else {
        std::fprintf(stderr, "strata serve: the prompt path allocates its own buffers (too few cache slots to borrow)\n");
    }
    rss_probe("the prompt path set up");
    // THE CHUNK STEPS DOWN INSTEAD OF EXITING: a chunk whose prompt buffers do not fit is tried again one size
    // smaller, down to 512 tokens (a smaller chunk only reads slower).
    init_prompt_paths = [this]() -> int {   // 0: ready; 1: failed (err set); 2: failed with "do not fit"
        if (share_pool) sp->set_cpu_pool(&*pool);
        if (!sp->init(wt, g, ss, srcp, &xcache, host_res.data(), o.prefill_chunk, main_cs, err, borrow, borrow_bytes,
                     o.prefill_head_alias))
            return err.find("do not fit") != std::string::npos ? 2 : 1;
        return 0;
    };
    static constexpr int64_t kStepChunks[] = {6144, 4096, 3072, 2048, 1536, 1024, 512};
    {
        // First by arithmetic: a prompt path without a loan allocates its buffers, so the chunk must leave
        // headroom on that device (a chunk that fits to the last MiB left hipBLAS nothing: its GEMMs then
        // failed to launch on gfx1201 and the prompt hung).  `bytes_needed` is the same count `init` makes.
        const int64_t kHeadroom = 512ll << 20;
        auto own_fits = [&](int64_t c, int& dev_out, int64_t& need_out, int64_t& free_out) -> bool {
            if (borrow != nullptr) return true;
            size_t fb = 0, tb = 0;
            cudaMemGetInfo(&fb, &tb);
            const int64_t need = (int64_t) strata::prefill::Prefill::bytes_needed(g, ss, c, srcp != nullptr);
            if (need + kHeadroom > (int64_t) fb) {
                dev_out = 0; need_out = need; free_out = (int64_t) fb;
                return false;
            }
            return true;
        };
        int dev = 0;
        int64_t need = 0, fb = 0;
        if (!own_fits(o.prefill_chunk, dev, need, fb)) {
            int64_t c = 0;
            for (const int64_t s : kStepChunks) {
                int d2 = 0;
                int64_t n2 = 0, f2 = 0;
                if (s < o.prefill_chunk && own_fits(s, d2, n2, f2)) { c = s; break; }
            }
            if (c > 0) {
                std::fprintf(stderr, "strata serve: a %lld-token chunk's prompt buffers need %lld MiB on CUDA%d, "
                                     "%lld MiB free: %lld-token chunks\n", (long long) o.prefill_chunk,
                             (long long) (need >> 20), dev, (long long) (fb >> 20), (long long) c);
                o.prefill_chunk = c;
            }
        }
    }
    mem_mark("before the prompt path init");
    for (;;) {
        const int r = init_prompt_paths();
        if (r == 0) break;
        int64_t next = 0;
        for (const int64_t c : kStepChunks)
            if (c < o.prefill_chunk) { next = c; break; }
        if (r == 2 && next > 0) {
            std::fprintf(stderr, "strata serve: %s: trying a %lld-token chunk\n", err.c_str(), (long long) next);
            // start the prompt paths over (a failed init may hold buffers), and consume the failed allocation's
            // error: it is sticky, and the next launch check would report "out of memory" for a kernel
            sp->reset();
            (void) cudaGetLastError();
            o.prefill_chunk = next;
            if (any_loan) {                  // smaller loans for the smaller chunk
                for (PfPart& p : pf_parts)
                    if (p.first >= 0) {
                        p.first = (int32_t) (p.cache->slots() - part_slots(p, next));
                        p.first_now = p.first;
                    }
                lend_first = pf_parts[0].first;
                borrow = lend_first >= 0 ? xcache.device_slot(lend_first) : nullptr;
                borrow_bytes = lend_first >= 0 ? part_bytes(pf_parts[0], lend_first) : 0;
            }
            err.clear();
            continue;
        }
        const int status = fail(1, "strata serve: %s\n", err.c_str());
        if (err.find("fit") != std::string::npos)   // #85: say what frees VRAM
            std::fprintf(stderr, "strata serve: the GPU has too little free VRAM for the prompt path: turn images "
                                 "off (setup: --vision no), close other programs using the GPU, use a shorter "
                                 "context, or read prompts in smaller chunks (--prefill 512)\n");
        return status;
    }
    mem_mark("the head and the prompt path");
    kvg_start(borrow != nullptr ? (int64_t) lend_first : xcache.slots());
    hist_stage = std::vector<int32_t>(kHistSlots, -1);
    if (cudaMalloc(&d_hist, kHistSlots * sizeof(int32_t)) != cudaSuccess)
        return fail(1, "strata serve: the penalty-history allocation failed\n");
    vh.d_res = thits.d_res;
    // STRATA_ROUTE_RESIDENT_MTP=1 (EXPERIMENTAL, with STRATA_ROUTE_RESIDENT=<margin>; changes the output): the MTP
    // draft's router gets the same residency bias, with the last layer's residency row as its table (single GPU)
    if (use_mtp && thits.d_res != nullptr && std::getenv("STRATA_ROUTE_RESIDENT") != nullptr &&
        std::getenv("STRATA_ROUTE_RESIDENT_MTP") != nullptr && std::atoi(std::getenv("STRATA_ROUTE_RESIDENT_MTP")) != 0) {
        const float margin = (float) std::atof(std::getenv("STRATA_ROUTE_RESIDENT"));
        int lo = 6, hi = 9;
        if (const char* v = std::getenv("STRATA_ROUTE_RESIDENT_RANKS")) std::sscanf(v, "%d-%d", &lo, &hi);
        if (margin > 0.0f) {
            mtp.set_route_resident(thits.d_res + (size_t) (g.n_layers - 1) * (size_t) g.n_expert, margin, lo, hi);
            std::fprintf(stderr, "strata generate: STRATA_ROUTE_RESIDENT_MTP: the MTP draft's router is biased toward layer %lld's resident experts (margin %g, ranks %d-%d; experimental)\n",
                         (long long) (g.n_layers - 1), margin, lo, hi);
        }
    }
    vh.h_res = host_res.empty() ? nullptr : host_res.data();
    vh.cache_base = thits.cache_base;
    vh.blob = thits.blob;
    vh.slot_off = xcache.slot_offsets();   // E-6: the device plan's pointers
    vh.n_slots = xcache.slots();
    if (pcie_balance()) ver.set_pcie_balance(&drive.d.pcie_model);
    if (!ver.init(wt, g, ss, vh, native_head.loaded() || prefill_head_alias_bound ? &native_head : nullptr, o.spec, err) ||
        (use_mtp && !prefill_head_alias_bound && !mtp.bind(wt, &native_head, ver.final_R_all(), err)))
        return fail(1, "strata serve: %s\n", err.c_str());
    if (prefill_head_alias_bound) mtp.set_window_R(ver.final_R_all());
    win_pool_fn = &drive_pool_multi;
    win_pool_user = (void*) &drive;
    mem_mark("the verifier and the drafter's binding");
    ver.set_split(o.spec_split);
    // auto: the copy kernel for every pack.  DMA (the native packs' default until 0.1.13) has the host call
    // cudaMemcpyAsync + cudaLaunchHostFunc inside a verify window while the GPU spins on the flag they raise;
    // issue #31's thread dumps show the host stuck in that cudaMemcpyAsync on a driver lock for good.  The copy
    // kernel needs no host CUDA call there, and costs ~1-3% decode on IQ3_S (45.3 -> 44.8 tok/s, 8 requests).
    ver.set_pcie_mode(o.pcie_mode == "dma" ? 0 : o.pcie_mode == "direct" ? 1 : 2);
    ckpt_pool.emplace(ckpt_pool_env());
    pp_t0 = Clock::now();
    nonfinite_hit = [this](const strata::core::Verifier& v, const char* what) -> bool {
        const int row = v.nonfinite_row();
        if (row < 0) return false;
        std::fprintf(stderr, "strata serve: non-finite logits (#879) in a %s window at position %lld (row %d): its "
                             "tokens are not emitted, the request ends (DONE nonfinite) and the next request reads "
                             "its prompt from token 0\n", what, (long long) v.nonfinite_pos(), row);
        std::fflush(stderr);
        nf_hit = true;
        nan_flush = true;
        return true;
    };
    imgs_below = [](const std::vector<ImgKey>& all, int64_t L) {
        std::vector<ImgKey> v;
        for (const ImgKey& k : all) if (k.start < L) v.push_back(k);
        return v;
    };
    ckpt_trace = std::getenv("STRATA_TRACE") != nullptr;
    gpu_sync_ok = [this]() -> bool {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e == cudaSuccess) return true;
        ckpt_why = std::string(": the GPU stopped responding before it (") + cudaGetErrorString(e) +
                   ") - a GPU hang; on Windows the driver is then reset";
        return false;
    };
    // a checkpoint of the state after `cur[0, L)`; false only when the copy itself failed
    checkpoint_at = [this](int64_t L, bool as_tail = false) -> bool {
        ckpt_why.clear();
        if (o.prompt_cache <= 0 || L < 1) return true;
        for (ConvCheckpoint& c : checks)
            if ((int64_t) c.ids.size() == L) {
                c.used = ++check_clock;
                if (!as_tail && L == tail_ckpt_len) tail_ckpt_len = -1;   // a periodic one lands here: a normal item now
                return true;
            }
        ConvCheckpoint c = ckpt_pool->take();   // (STRATA_CKPT_POOL)
        c.ids.assign(cur.begin(), cur.begin() + L);
        c.imgs = imgs_below(req_imgs, L);
        if (ckpt_pool->on()) {
            // STRATA_CKPT_POOL: the state goes into the vectors a pooled checkpoint already has (none resized),
            // synchronized first
            const size_t n = 1;
            const bool reused = !c.gdn.empty();   // (a pooled one)
            std::vector<char> synced(n, 0), saved(n, 0);
            std::vector<std::string> why(n);
            std::vector<double> sync_part(n, 0.0), part_ms(n, 0.0);
            std::vector<std::exception_ptr> thrown(n);   // (a failed allocation: thrown here, as without the threads)
            const std::function<void(size_t)> save_part = [&](size_t k) {
                try {
                    const Clock::time_point p0 = Clock::now();
                    const cudaError_t e = cudaDeviceSynchronize();
                    if (e != cudaSuccess) {
                        why[k] = std::string(": the GPU stopped responding before it (") + cudaGetErrorString(e) +
                                 ") - a GPU hang; on Windows the driver is then reset";
                        return;
                    }
                    synced[k] = 1;
                    const Clock::time_point p1 = Clock::now();
                    saved[k] = checkpoint_save(c, ss, g);
                    sync_part[k] = std::chrono::duration<double, std::milli>(p1 - p0).count();
                    part_ms[k] = std::chrono::duration<double, std::milli>(Clock::now() - p0).count();
                } catch (...) {
                    thrown[k] = std::current_exception();
                }
            };
            const Clock::time_point t0 = Clock::now();
            for (size_t k = 0; k < n; ++k) {
                save_part(k);
                if (!synced[k] || !saved[k]) break;
            }
            for (size_t k = 0; k < n; ++k)
                if (thrown[k]) std::rethrow_exception(thrown[k]);
            for (size_t k = 0; k < n; ++k) {
                if (!synced[k]) { ckpt_why = why[k]; return false; }
                if (!saved[k]) return false;
            }
            if (ckpt_trace) {   // sync: the first stage's (the others' are in their parts' times)
                const double wall = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                std::string each;
                for (size_t k = 0; k < n; ++k) {
                    char b[32];
                    std::snprintf(b, sizeof b, " %.1f", part_ms[k]);
                    each += b;
                }
                ++req_times.ckpts;
                req_times.sync_ms += sync_part[0];
                req_times.save_ms += wall - sync_part[0];
                std::fprintf(stderr, "strata trace: checkpoint at %lld: sync %.1f ms, save %.1f ms (%zu stage(s), "
                                     "each%s ms, %s)\n", (long long) L, sync_part[0], wall - sync_part[0], n,
                             each.c_str(),
                             reused ? "into a dropped checkpoint's vectors" : "new vectors");
            }
        } else {
            Clock::time_point t0 = Clock::now();
            auto lap = [&](double& acc) {   // (STRATA_TRACE only)
                if (!ckpt_trace) return;
                const Clock::time_point t1 = Clock::now();
                acc += std::chrono::duration<double, std::milli>(t1 - t0).count();
                t0 = t1;
            };
            double sync_ms = 0.0, save_ms = 0.0;
            if (!gpu_sync_ok()) return false;
            lap(sync_ms);
            if (!checkpoint_save(c, ss, g)) return false;
            lap(save_ms);
            if (ckpt_trace) {
                ++req_times.ckpts;
                req_times.sync_ms += sync_ms;
                req_times.save_ms += save_ms;
                std::fprintf(stderr, "strata trace: checkpoint at %lld: sync %.1f ms, save %.1f ms (1 stage(s))\n",
                             (long long) L, sync_ms, save_ms);
            }
        }
        c.used = ++check_clock;
        if (seg_plan.chunk > 0 && L > seg_plan.from && L < seg_plan.to) c.plan = seg_plan;   // inside a planned segment
        if (as_tail) {   // only one tail checkpoint stays alive: the previous request's goes
            const int64_t old_tail = tail_ckpt_len;
            auto old = [&](const ConvCheckpoint& k) { return old_tail >= 0 && (int64_t) k.ids.size() == old_tail; };
            if (ckpt_pool->on())
                ckpt_pool->erase_if(checks, old);
            else
                checks.erase(std::remove_if(checks.begin(), checks.end(), old), checks.end());
            tail_ckpt_len = L;
        } else if (L == tail_ckpt_len) {
            tail_ckpt_len = -1;   // the flagged tail was dropped and a normal checkpoint takes its length
        }
        checks.push_back(std::move(c));
        const Clock::time_point te0 = Clock::now();
        const size_t before_evict = checks.size();
        while ((int) checks.size() > o.prompt_cache) {
            std::vector<uint64_t> stamps;
            std::vector<char> is_tail;
            stamps.reserve(checks.size());
            for (const ConvCheckpoint& k : checks) {
                stamps.push_back(k.used);
                is_tail.push_back(tail_ckpt_len >= 0 && (int64_t) k.ids.size() == tail_ckpt_len);
            }
            // the tail checkpoint serves only a branch of the last request: it leaves before a periodic one
            std::unique_ptr<bool[]> tail_flags(new bool[is_tail.size()]);
            for (size_t i = 0; i < is_tail.size(); ++i) tail_flags[i] = is_tail[i] != 0;
            std::vector<int64_t> lens;
            for (const ConvCheckpoint& k : checks) lens.push_back((int64_t) k.ids.size());
            const size_t victim = o.prompt_cache_spread
                ? strata::program::conv_cache::spread_victim(lens.data(), stamps.data(), stamps.size(),
                                                             o.prompt_cache, tail_flags.get(), nullptr)
                : strata::program::conv_cache::eviction_victim(stamps.data(), stamps.size(), o.prompt_cache,
                                                               tail_flags.get(), nullptr);
            ckpt_pool->give(std::move(checks[victim]));   // (STRATA_CKPT_POOL; else nothing)
            checks.erase(checks.begin() + (std::ptrdiff_t) victim);
        }
        if (ckpt_trace && checks.size() < before_evict) {
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - te0).count();
            req_times.evict_ms += ms;
            std::fprintf(stderr, "strata trace: checkpoint at %lld: %zu evicted in %.1f ms\n", (long long) L,
                         before_evict - checks.size(), ms);
        }
        return true;
    };
    sp->on_chunk = [this](const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {
        std::vector<int32_t> nxt((size_t) T);
        for (int64_t t = 0; t < T; ++t) nxt[(size_t) t] = (int32_t) cur[(size_t) (p0 + t + 1)];
        // E-9: batched through the prompt path when it can
        bool batched = false;
        const strata::program::draft_kv::Plan dkv = strata::program::draft_kv::plan(
            use_mtp, false, false, strata::core::native_embed() != nullptr);
        if (dkv.try_batched || dkv.why != nullptr) {
            const char* why = dkv.why;
            if (dkv.try_batched) batched = sp->draft_kv(mtp, R_rows, nxt.data(), T, p0, e, &why);
        }
        if (!e.empty() || (use_mtp && !batched && !mtp.prefill(R_rows, nxt.data(), T, p0, e))) return false;
        if (use_mtp && std::getenv("STRATA_SNAPSHOT_VERIFY") != nullptr)
            std::fprintf(stderr, "strata serve: DRAFT_PREFILL path=%s mode=%d cells=%lld\n",
                         batched ? "batched" : "token", mtp.kv_state().kv_mode, (long long) T);
        // progress for the server window: PP <position reached> <prompt tokens> <ms> <fresh tokens/s>
        const int64_t done = p0 + T;
        pp_reached = done;
        report_progress(done);
        strata::core::progress_at("reading the prompt (batched), done up to token", done);
        strata::core::progress_beat();
        const bool periodic_checkpoint = o.prompt_cache_every > 0 && done >= pp_next_check;
        // Keep the prefill's existing chunk geometry. One extra near-tail checkpoint
        // can serve a branch whose shared prefix ends before the final cached state.
        const bool tail_checkpoint = o.prompt_cache_tail && o.prompt_cache > 0 && o.prompt_cache_every > 0 &&
            !pp_tail_saved && pp_total - done > std::max<int64_t>(1, o.short_read) &&
            pp_total - done <= T;   // this chunk's real size (--prefill auto chunks differ from o.prefill_chunk)
        if (periodic_checkpoint || tail_checkpoint) {
            const bool saved = checkpoint_at(done, tail_checkpoint && !periodic_checkpoint);
            if (!saved) { e = "saving a conversation checkpoint failed" + ckpt_why; return false; }
            if (periodic_checkpoint) pp_next_check = done + o.prompt_cache_every;
            if (tail_checkpoint) pp_tail_saved = true;
        }
        return true;
    };
    drive.d.plan = ver.plan_sink();
    {
        int main_dev = 0;
        cudaGetDevice(&main_dev);
        std::vector<int> devs{main_dev};
        std::vector<int> card((size_t) g.n_layers, 0);
        fs_swap.reset(strata::core::foresight_swap_from_env(g.n_layers, g.n_expert, card, devs));
        drive.d.fs = fs_swap.get();
        if (fs_swap && drive.d.lookahead != nullptr) lookahead.set_foresight(fs_swap.get());
        src.set_foresight_swap(fs_swap.get());   // #2134: the exchanges hold the experts whose bytes they move
    }
    fs_detach.la = &lookahead;
    fs_detach.src = &src;
    drive.d.pcie_num = std::max(0, std::min(256, (int) (o.pcie_frac * 256.0 + 0.5)));
    const bool all_experts_resident = !host_res.empty() &&
        std::all_of(host_res.begin(), host_res.end(), [](int32_t r) { return r >= 0; });
    // STRATA_ROUTE_TAIL_SKIP: ON BY DEFAULT on CUDA builds (owner decision, 0.1.42) and on HIP gfx1200/gfx1201 (0.1.43): a missed expert that every token of
    // a verify window routes at rank 7 or lower is neither copied nor computed (answers differ slightly from 0.1.41,
    // KL ~0.003-0.006; docs/DETAILS.md).  =0 restores 0.1.41's bytes.  Nothing to do, so no default and no line, when every
    // expert is in VRAM.
    {
        const char* tse = std::getenv("STRATA_ROUTE_TAIL_SKIP");
        // #1884: with nearly every expert in the GPU cache (V100 32 GB, Coder IQ1_M: 94.8% of the experts, 99.9% decode hits) the few
        // misses are cheap and the skip costs more draft acceptance (70.1% -> 68.2%) than it saves: -1..2% decode.  No default then.
        size_t cached_experts = 0;
        for (int32_t r : host_res) cached_experts += r >= 0 ? 1 : 0;
        // (STRATA_TAIL_SKIP_MAX_CACHED=<percent>, default 90, moves the line; 101 = never).  0.1.43: 60 on a HIP layer split, where
        // 2x R9700 (5 pairs, skip on vs off): Q2_0 story -6.2% at 70% cached, -1.8% at 60%; IQ3_S +6..15% at 44-52%, about 0 at 60%.
        const double max_cached_default = 0.90;
        const char* mcs = std::getenv("STRATA_TAIL_SKIP_MAX_CACHED");
        const double max_cached = mcs != nullptr && mcs[0] != 0 ? std::atof(mcs) / 100.0 : max_cached_default;
        const bool nearly_all_cached = !host_res.empty() && !all_experts_resident &&
                                       (double) cached_experts >= max_cached * (double) host_res.size();
        const bool tail_default_path = tse == nullptr && !all_experts_resident && !nearly_all_cached;
        const bool tail_default_ok = tail_default_path;
        if (tse == nullptr && nearly_all_cached)
            std::fprintf(stderr, "strata serve: STRATA_ROUTE_TAIL_SKIP is off by default: %.1f%% of the experts are in the GPU cache, so "
                                 "few are missed and skipping them costs more draft acceptance than it saves; "
                                 "STRATA_ROUTE_TAIL_SKIP=7 turns it on\n", 100.0 * (double) cached_experts / (double) host_res.size());
        if (tail_default_ok) strata::core::set_tail_skip_rank(7);
        if (strata::core::tail_skip_rank() > 0 && !all_experts_resident)
            std::fprintf(stderr, "strata serve: STRATA_ROUTE_TAIL_SKIP=%d is ON%s: a missed expert that every token of a verify window routes at rank %d or "
                                 "lower is skipped (+10..20%% decode on 12-16 GB cards that miss experts, answers differ slightly from 0.1.41); "
                                 "STRATA_ROUTE_TAIL_SKIP=0 turns it off\n", strata::core::tail_skip_rank(),
                         tse == nullptr ? " (default)" : "", strata::core::tail_skip_rank());
    }
    if (o.adapt_every > 0 && o.adapt_swaps > 0 && !all_experts_resident)
        drive.d.usage.assign((size_t) (g.n_layers * g.n_expert), 0.0f);
    if (o.pcie_frac_auto && !all_experts_resident) {
        const auto& pl = strata::kernels::cpu::expert_layout();
        double sum = 0.0;
        for (int64_t l = 0; l < g.n_layers; ++l) sum += (double) pl.blob_bytes(l);
        pcie_auto.on = true;
        pcie_auto.link_gbps = o.pcie_probe_gbps;
        pcie_auto.blob_bytes = g.n_layers > 0 ? sum / (double) g.n_layers : 0.0;
    }
    // PcieAuto after a decode window: the share, refined from the CPU pool's speed.
    pcie_auto_step = [this](double& req_pcie_frac) {
        if (!pcie_auto.on) return;
        double tc_us = 0.0, rate = 0.0;
        const double nf = req_pcie_frac == o.pcie_frac
                              ? pcie_auto.window(drive.d.ms_run, drive.d.multi_misses, o.pcie_frac, &tc_us, &rate)
                              : (pcie_auto.mark(drive.d.ms_run, drive.d.multi_misses), -1.0);
        if (nf < 0.0) return;
        std::fprintf(stderr, "strata serve: PCIe share: the CPU pool takes %.0f us per missed expert (%.1f GB/s of expert bytes), "
                             "the link %.1f GB/s -> pcie_frac %.2f (was %.2f; --pcie-frac N fixes it, "
                             "STRATA_PCIE_FRAC_DEFAULT=old keeps the start-up rule)\n",
                     tc_us, rate, pcie_auto.link_gbps, nf, o.pcie_frac);
        o.pcie_frac = nf;
        req_pcie_frac = nf;
        const auto num_of = [](double f) { return std::max(0, std::min(256, (int) (f * 256.0 + 0.5))); };
        drive.d.pcie_num = num_of(nf);
    };
    if (!o.expert_profile_save.empty() && !all_experts_resident) {
        if (drive.d.usage.empty() || host_res.empty())
            std::fprintf(stderr, "strata serve: --expert-profile-save needs the adaptive tier (--adapt-every and "
                                 "--adapt-swaps above 0) and --expert-profile: nothing will be saved\n");
        else
            heat.assign(drive.d.usage.size(), 0.0);
    }
    profile_saved_at = Clock::now();
    if (cudaStreamCreateWithFlags(&adapt_stream, cudaStreamNonBlocking) != cudaSuccess)
        return fail(1, "strata serve: cannot create the refill stream\n");
    cudaEventCreateWithFlags(&adapt_ev, cudaEventDisableTiming);
    // #2010: the overlap is opt-in on every backend now, so the off state is the default and has to be visible too.
    // The predicates are side-effect free, so they are read in both states: they name why it could not run.
    const bool aov_want = adapt_overlap();
    const char* aov_why = o.adapt_async ? "not with --adapt-async"
                          : strata::core::dma_batch_mode() != 0 ? "not with STRATA_DMA_BATCH"
                          : adapt_lag() != 1 || adapt_nowait() ? "not with STRATA_ADAPT_LAG / STRATA_ADAPT_NOWAIT"
                          // the resident RAM mode is exact too (soft_commit_exchanges) but not a win everywhere: RTX A4000 +7.0%
                          // (5/5 pairs), RTX 3060 on a 46 GB box -2.6% (0/5; the CPU experts and the copies share the RAM bandwidth)
                          : src.complement_ready() && !(std::getenv("STRATA_ADAPT_OVERLAP_RESIDENT") &&
                                                        std::getenv("STRATA_ADAPT_OVERLAP_RESIDENT")[0] == '1')
                                ? "not in the resident RAM mode unless STRATA_ADAPT_OVERLAP_RESIDENT=1"
                          : drive.d.usage.empty() ? "the adaptive tier is off"
                          : nullptr;
    if (aov_want) {
        if (aov_why == nullptr) {
            agate.ticks.resize((size_t) g.n_layers + 1);
            for (int64_t l = 0; l <= g.n_layers; ++l) agate.ticks[(size_t) l] = {&agate, (int32_t) l};
            agate.has.assign((size_t) g.n_layers, 0);
            if (!ver.set_adapt_gate(&agate)) aov_why = "the device-planned layers (STRATA_VERIFY_DEVICE_PLAN) are on";
            // #2188: on Windows the default is the host function.  Under WDDM a stream's work sits in the driver's
            // command buffer until something flushes it, and nothing does while the host only spins on the pinned
            // int: the first request stopped with "copies for layer 45 never landed".  STRATA_ADAPT_SIGNAL=copy asks
            // for the stream-ordered copy there too (the enqueue then flushes the stream), =host everywhere.
            const char* asig = std::getenv("STRATA_ADAPT_SIGNAL");
            const bool sig_copy = !(asig != nullptr && asig[0] == 'h');
            if (aov_why == nullptr && !sig_copy)
                std::fprintf(stderr, "strata serve: adaptive overlap signal: host functions%s\n",
                             asig == nullptr ? " (the Windows default, #2188; STRATA_ADAPT_SIGNAL=copy: a stream-ordered copy)" : "");
            if (aov_why == nullptr && sig_copy) {
                // the landing of each layer's copies is told by a stream-ordered D2H copy of the layer number into pinned memory
                // (STRATA_ADAPT_SIGNAL=host: a host function instead). No callback thread has to be scheduled while the pool's workers spin on every other CPU
                int32_t* vals = nullptr; void* hp = nullptr;
                std::vector<int32_t> iota((size_t) g.n_layers + 1);
                for (size_t i = 0; i < iota.size(); ++i) iota[i] = (int32_t) i;
                if (cudaMalloc((void**) &vals, iota.size() * 4) == cudaSuccess &&
                    cudaMemcpy(vals, iota.data(), iota.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess &&
                    cudaHostAlloc(&hp, 64, cudaHostAllocPortable) == cudaSuccess) {
                    *(volatile int32_t*) hp = -1;
                    agate.hland = (volatile int32_t*) hp;
                    agate.dvals = vals;
                    std::fprintf(stderr, "strata serve: adaptive overlap signal: stream-ordered copies (STRATA_ADAPT_SIGNAL=host: a host function)\n");
                } else { (void) cudaGetLastError(); }
            }
        }
        aov = aov_why == nullptr;
        if (aov)
            std::fprintf(stderr, "strata serve: adaptive overlap (STRATA_ADAPT_OVERLAP=0 turns it off): the adaptive tier's copies land while the next window runs "
                                 "(same tokens, a layer waits for its own copies)\n");
        else if (std::getenv("STRATA_ADAPT_OVERLAP") != nullptr)   // the env var was set: say what stopped it
            std::fprintf(stderr, "strata serve: STRATA_ADAPT_OVERLAP=1 is off: %s\n", aov_why);
    } else
        std::fprintf(stderr, "strata serve: adaptive overlap: off%s%s\n",
                     aov_why ? ": " : " (STRATA_ADAPT_OVERLAP=1 turns it on)", aov_why ? aov_why : "");
    res_upload = [this]() -> cudaError_t {
        return d_res != nullptr ? res_put(d_res) : cudaSuccess;
    };
    apply_pending = [this](bool wait) {
        if (pending.empty()) return;
        if (wait) cudaEventSynchronize(adapt_ev);
        else if (cudaEventQuery(adapt_ev) != cudaSuccess) return;
        unpin_blobs(pin_live);
        src.commit_exchanges();   // the resident RAM mode: the evicted experts take their places in RAM
        for (const auto& [i, slot] : pending) {
            host_res[(size_t) i] = slot;
            srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);   // in VRAM now: RAM not needed
        }
        if (drive.d.evict_sync) {   // a failed upload leaves them to the pre-launch hook
            for (const auto& pr : pending) drive.d.res_dirty_idx.push_back(pr.first);
            drive.d.res_dirty = true;
        }
        pending.clear();
        pending_age = 0;
        agate.active.store(false, std::memory_order_release);   // 0.1.43: the copies have landed (STRATA_ADAPT_OVERLAP)
        aflipped = false;
        if (res_upload() == cudaSuccess) {   // the whole table: the hook has nothing left
            drive.d.res_dirty = false;
            drive.d.res_dirty_idx.clear();
        }
    };
    // Where the tier does not wait for its copies (STRATA_ADAPT_NOWAIT=1, STRATA_ADAPT_LAG > 1), its evictions reach
    // the device's tables before the next window: adapt() marks them non-resident at once, apply_pending uploads
    // only when the round's copies land - a window or more later, and until then doorbell_publish_res skips the
    // activation of a layer whose routed experts all look resident, while the pool computes an evicted one from the
    // activation published before.  The hook runs at each verifier's launch.  --adapt-async uploads at each of its
    // steps.
    // STRATA_ADAPT_CHECKRES=1 reads the tables back at each launch and compares them with the host's.
    const bool tier_lags = !drive.d.usage.empty() && (adapt_nowait() || adapt_lag() > 1);
    drive.d.evict_sync = tier_lags && adapt_evict_sync();
    res_checking = std::getenv("STRATA_ADAPT_CHECKRES") != nullptr;
    if (res_checking) {
        res_back.assign(host_res.size(), 0);
        drive.d.dres_check = res_back.data();
    }
    // The changed entries go up by a kernel on each card's verifier stream, reading them from mapped memory: a
    // synchronous upload waits for the copy engine, which the tier's swaps of the round before still hold (~4 ms
    // on 33 swaps; measured +0.9 ms a chat round).  Too many changes, or no mapped memory: the whole table.
    std::vector<int> res_devs{-1};
    res_pairs = drive.d.evict_sync ? res_pairs_alloc(kResPatchCap, res_devs) : nullptr;
    res_sync = [this](bool check, std::string& e) -> bool {
        if (drive.d.res_dirty) {
            const int64_t n = res_patch_pairs(drive.d.res_dirty_idx, host_res, res_pairs, kResPatchCap);
            cudaError_t r = cudaSuccess;
            if (n >= 0) {
                if (d_res != nullptr) {
                    strata::kernels::res_patch(d_res, res_pairs, (int) n, ver.stream());
                    r = cudaGetLastError();
                }
            } else {
                r = res_upload();
            }
            if (r != cudaSuccess) {
                e = std::string("the residency table update failed: ") + cudaGetErrorString(r);
                return false;
            }
            drive.d.res_dirty = false;
            drive.d.res_dirty_idx.clear();
        }
        if (!check) return true;
        cudaStreamSynchronize(ver.stream());   // the patches landed (the verifier streams do not block the read)
        int64_t bad = 0;
        bad += res_check(d_res, host_res, res_back, g.n_expert, res_windows);
        ++res_windows;
        res_bad += bad;
        res_bad_windows += bad > 0;
        if ((res_windows & 255) == 0 || (bad > 0 && res_bad_windows <= 4))
            std::fprintf(stderr, "strata serve: res check: %lld windows, %lld with mismatches (%lld entries), "
                                 "%lld layer-windows with CPU experts and no activation published\n",
                         (long long) res_windows, (long long) res_bad_windows, (long long) res_bad,
                         (long long) drive.d.stale_x);
        return true;
    };
    pre_launch_off.v = &ver;
    pre_launch_off.d = &drive.d;
    pre_launch_off.pairs = res_pairs;
    if (drive.d.evict_sync || res_checking) {
        ver.set_pre_launch([this](cudaStream_t, std::string& e) { return res_sync(res_checking, e); });
        if (drive.d.evict_sync)
            std::fprintf(stderr, "strata serve: the adaptive tier's evictions reach the device's table before the next "
                                 "window (STRATA_ADAPT_EVICT_SYNC=0: when its copies land)\n");
    }
    // the VRAM tier follows the conversation (the same rule as the speculative loop below)
    adapt = [this]() -> bool {
        if (!pending.empty()) return true;   // the previous swaps are still in flight
        struct Swap { float gain; int32_t layer, in, out; };
        std::vector<Swap> swaps;
        std::vector<std::pair<float, int32_t>> cand, vict;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            cand.clear();
            vict.clear();
            const float* u = drive.d.usage.data() + l * g.n_expert;
            const int32_t* r = host_res.data() + l * g.n_expert;
            for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                if (r[e] < 0) { if (u[e] >= 2.0f) cand.emplace_back(u[e], e); }
                else vict.emplace_back(u[e], e);
            }
            if (cand.empty() || vict.empty()) continue;
            std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
            const size_t nc = std::min(cand.size(), vict.size());
            std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                              [](auto& a, auto& b) { return a.first < b.first; });
            for (size_t i = 0; i < nc; ++i) {
                if (cand[i].first < vict[i].first + o.adapt_min_gain) break;
                swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
            }
        }
        std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
        if ((int) swaps.size() > o.adapt_swaps) swaps.resize((size_t) o.adapt_swaps);
        if (aov) {   // 0.1.43: the copies go in layer order and each layer's landing is told to the verify window
            std::stable_sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.layer < b.layer; });
            agate.landed.store(-1, std::memory_order_release);
            if (agate.hland != nullptr) *agate.hland = -1;
            agate.has.assign((size_t) g.n_layers, 0);
            for (const Swap& s : swaps) agate.has[(size_t) s.layer] = 1;
        }
        // the copies back read the cache of the card that owns each layer (see resident_stage_swaps)
        auto home_of = [&](int64_t) -> SwapHome { return {&xcache, adapt_stream, -1}; };
        if (!resident_stage_swaps(src, host_res, g.n_expert, swaps, home_of)) return false;
        adapt_nswaps += (int64_t) swaps.size();   // (read after the thread's join)
        bool main_live = false;
        // STRATA_DMA_BATCH (F12): the main device's copies as one cudaMemcpyBatchAsync (mode 0: one each)
        const int dma_mode = strata::core::dma_batch_mode();
        std::vector<void*> cp_dst;
        std::vector<const void*> cp_src;
        std::vector<size_t> cp_bytes;
        for (size_t si = 0; si < swaps.size(); ++si) {
            const Swap& s = swaps[si];
            if (aov && (si == 0 || swaps[si - 1].layer != s.layer)) {
                // test hook (STRATA_ADAPT_OVERLAP_SLOW_US=N): a slow link, to make the windows wait for the copies
                static const int slow_us = [] { const char* e = std::getenv("STRATA_ADAPT_OVERLAP_SLOW_US"); return e ? std::atoi(e) : 0; }();
                static int slow_arg = slow_us;
                if (slow_us > 0)
                    cudaLaunchHostFunc(adapt_stream, [](void* p) { std::this_thread::sleep_for(std::chrono::microseconds(*(int*) p)); }, &slow_arg);
            }
            const size_t in = (size_t) s.layer * g.n_expert + s.in, out = (size_t) s.layer * g.n_expert + s.out;
            const int32_t slot = host_res[out];
            const uint8_t* b = srcp->blob_stable(s.layer, s.in);
            if (dma_mode != 0 && slot >= 0 && b != nullptr) {
                cp_dst.push_back(xcache.device_slot(slot));
                cp_src.push_back(b);
                cp_bytes.push_back((size_t) strata::kernels::cpu::expert_layout().blob_bytes(s.layer));
                main_live = true;
                host_res[out] = strata::core::kNotResident;
                srcp->prefetch(s.layer, s.out);
                pending.emplace_back((int32_t) in, slot);
                continue;
            }
            if (slot < 0 || b == nullptr ||
                adapt_copy_h2d(xcache.device_slot(slot), b,
                               (size_t) strata::kernels::cpu::expert_layout().blob_bytes(s.layer),
                               adapt_stream) != cudaSuccess) {
                std::fprintf(stderr, "strata serve: adaptive swap copy failed (layer %d, slot %d, pinned %d): %s\n",
                             (int) s.layer, (int) slot, pin_live.empty() ? 0 : 1, cudaGetErrorString(cudaGetLastError()));
                return false;
            }
            main_live = true;
            host_res[out] = strata::core::kNotResident;   // evicted now: the CPU computes it meanwhile
            srcp->prefetch(s.layer, s.out);   // a file-backed arena released its pages: read them back ahead
            pending.emplace_back((int32_t) in, slot);      // resident once the copy has landed
            if (aov && (si + 1 == swaps.size() || swaps[si + 1].layer != s.layer) &&
                (agate.hland != nullptr
                     ? cudaMemcpyAsync((void*) agate.hland, agate.dvals + s.layer, 4, cudaMemcpyDeviceToHost, adapt_stream)
                     : cudaLaunchHostFunc(adapt_stream, [](void* p) {
                    auto* t = (strata::core::AdaptGate::Tick*) p;
                    int32_t cur = t->gate->landed.load(std::memory_order_relaxed);
                    while (cur < t->layer && !t->gate->landed.compare_exchange_weak(cur, t->layer, std::memory_order_release)) {}
                }, &agate.ticks[(size_t) s.layer])) != cudaSuccess) {
                (void) cudaGetLastError();
                cudaStreamSynchronize(adapt_stream);   // no callback: wait here, the gate opens at once
                agate.landed.store((int32_t) g.n_layers, std::memory_order_release);
            }
        }
        if (drive.d.evict_sync && !swaps.empty()) {   // the evicted ones are misses now (see the pre-launch hook)
            for (const Swap& s : swaps) drive.d.res_dirty_idx.push_back((int32_t) (s.layer * g.n_expert + s.out));
            drive.d.res_dirty = true;
        }
        if (!cp_dst.empty() && strata::core::copy_blobs(cp_dst.data(), cp_src.data(), cp_bytes.data(), cp_dst.size(),
                                                        adapt_stream, dma_mode) != cudaSuccess) {
            std::fprintf(stderr, "strata serve: adaptive swap copy failed (batched): %s\n",
                         cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        if (!swaps.empty()) cudaEventRecord(adapt_ev, adapt_stream);
        // #2188: hand the copies and their landing signals to the GPU now.  WDDM batches a stream's work until a
        // driver call flushes it, and the verify window only spins on the pinned int (Linux submits at once: no-op)
        if (aov && agate.hland != nullptr && !swaps.empty()) (void) cudaStreamQuery(adapt_stream);
        (void) main_live;
        // #477: the routing counted since the start (each count adds up to 1 / (1 - --adapt-decay) over its
        // decays: the sum is proportional to the routing itself) - only with --expert-profile-save, else `heat`
        // is empty
        for (size_t i = 0; i < heat.size(); ++i) heat[i] += (double) drive.d.usage[i];
        for (float& v : drive.d.usage) v *= o.adapt_decay;
        return true;
    };
    // ---- --adapt-async 1: the asynchronous adaptive tier (opt-in; the resident RAM mode, see the checks where
    // the RAM copy is built).  adapt() runs a whole round - choose, copy the evicted slots back with a stream sync,
    // copy the new experts in from the RAM copy - on a thread the decode loop joins after the draft: ~24 ms every
    // 4th window with ~90 swaps.  Here a round moves through four steps, each started by `adapt_tick` between
    // windows once the previous step's copies (events) and host work (the job thread) are done:
    //   1. job: choose (adapt()'s rule, over snapshots of the routing counts and the residency) and copy the
    //      evicted slots back into exchange buffers (D2H), each from the cache of the card that owns its layer;
    //   2. stage the exchanges and mark the evicted experts non-resident (the CPU reads them from the exchange
    //      buffers from the next window on); job: copy the new experts in (straight from a page-locked RAM copy,
    //      else through a pinned bounce buffer), on the owning card's refill stream;
    //   3. mark the new experts resident; job: move the evicted blobs into the RAM places the new ones left;
    //   4. flip the copy's offsets.
    // A slot is overwritten only after its old expert stopped being planned for the GPU (step 2, between windows,
    // with the device copies of the residency table uploaded as apply_pending does), and a RAM place only once its
    // expert is resident (step 3), so no window computes from memory that is changing.  A request starts with the
    // round drained (adapt_tick(true)), before its prompt lends any slot.  Not bit-exact run to run: which window
    // first computes a swapped-in expert on the GPU (which rounds differently from the CPU) depends on when its
    // copy lands.  `j`: the swap's exchange buffer (its copy back) and bounce buffer (a_cap + j).
    // The arena mode (srcp == &arena_src, a_arena below): every expert is in the arena already (a pageable or
    // partly pinned copy that never moves), so a round has no exchange and no RAM move - choose, mark the evicted
    // experts non-resident, copy the new ones in, mark them resident (steps 1-3 reduced to the copy in; RamFence and
    // step 4 pass through).  By default (STRATA_ADAPT_PICK_INLINE) the choice runs on this thread in the tick that
    // starts the round, and the copies in are posted in the same tick; the next tick lands them
    // (STRATA_ADAPT_ASYNC_WAIT: it waits for the copies rather than polling, so the window at which an expert first
    // runs on the GPU does not depend on timing) and a new round may start at once.  The job thread's copies
    // read the arena (blob_at: no counter) and block it on pageable memory, never this thread;
    // STRATA_ADAPT_ASYNC_BATCH caps the copies in flight per tick (the copy engine is shared with the windows).
    // Invariant (the serial loop, and the batch loop between windows): at a tick nothing in flight reads a slot -
    // a window's run ends with cudaStreamSynchronize (verify.cpp), the draft is synchronous and the commit does not
    // read the expert cache - so a slot whose expert was marked non-resident here may be overwritten right away;
    // and the job thread never touches host_res or drive.d.usage (it works on the snapshots a_usage / a_res).
    // Anything that gives a slot to something else (kvg_ensure's quiesce, a prompt's lend) calls adapt_quiesce first.
    // the arena mode (srcp == &arena_src): every expert is in the arena already, so a round is a copy in only - no
    // exchange buffers, no RAM moves (see a_arena below)
    a_arena = srcp == &arena_src;
    // as the blocking tier's exchange buffers; the arena has none, so its rounds are as large as the blocking tier's
    a_cap = a_arena ? o.adapt_swaps : std::min<int64_t>(o.adapt_swaps, 96);
    const auto a_env = [](const char* name, int def) {
        const char* v = std::getenv(name);
        return v != nullptr && *v != 0 ? std::atoi(v) : def;
    };
    // arena only (the other modes keep their steps as they were): the choice on this thread (STRATA_ADAPT_PICK_INLINE),
    // the wait for the copies (STRATA_ADAPT_ASYNC_WAIT), the copies in flight per tick (STRATA_ADAPT_ASYNC_BATCH, 0 = all)
    a_inline = a_arena && a_env("STRATA_ADAPT_PICK_INLINE", 1) != 0;
    a_wait = a_arena && a_env("STRATA_ADAPT_ASYNC_WAIT", 1) != 0;
    a_batch = a_arena ? (size_t) std::max(0, a_env("STRATA_ADAPT_ASYNC_BATCH", 0)) : 0;
    if (o.adapt_async && !drive.d.usage.empty() &&
        ((src.complement_ready() && src.exchange_capacity() >= 2 * a_cap) || a_arena)) {
        ahomes.push_back({&xcache, adapt_stream, adapt_ev, 0, false});
        ajob = std::make_unique<JobThread>();
        std::fprintf(stderr, "strata serve: adaptive tier asynchronous (--adapt-async 1): up to %lld swaps per round, "
                             "a round every %d windows%s\n", (long long) a_cap, o.adapt_every,
                     a_arena ? " (arena: copy-in only)" : "");
    }
    a_mark = [this](AHome& h) {   // an event after the copies just queued on h's stream, submitted now (WDDM)
        const strata::core::OnDevice on(h.dev);
        if (cudaEventRecord(h.ev, h.stream) != cudaSuccess) a_err = true;
        (void) cudaStreamQuery(h.stream);
        h.used = true;
    };
    a_pick = [this]() {   // step 1's choice (host work)
        struct C { float gain; int32_t layer, in, out; };
        std::vector<C> all;
        std::vector<std::pair<float, int32_t>> cand, vict;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            cand.clear();
            vict.clear();
            const float* u = a_usage.data() + l * g.n_expert;
            const int32_t* r = a_res.data() + l * g.n_expert;
            for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                if (r[e] < 0) { if (u[e] >= 2.0f) cand.emplace_back(u[e], e); }
                else vict.emplace_back(u[e], e);
            }
            if (cand.empty() || vict.empty()) continue;
            std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
            const size_t nc = std::min(cand.size(), vict.size());
            std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                              [](auto& a, auto& b) { return a.first < b.first; });
            for (size_t i = 0; i < nc; ++i) {
                if (cand[i].first < vict[i].first + o.adapt_min_gain) break;
                all.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
            }
        }
        std::sort(all.begin(), all.end(), [](const C& a, const C& b) { return a.gain > b.gain; });
        aswaps.clear();
        for (const C& c : all) {
            if ((int64_t) aswaps.size() >= a_cap) break;
            const int32_t slot = a_res[(size_t) c.layer * g.n_expert + c.out];
            if (slot < 0) continue;
            // an exchange (as resident_stage_swaps): `in` from the RAM copy, `out` held by its slot alone takes its
            // place there; otherwise `out` is in the copy already (the lend region) or `in` is not (a RAM budget)
            const bool x = !a_arena && src.has_resident(c.layer, c.in) && !src.has_resident(c.layer, c.out);
            aswaps.push_back({c.layer, c.in, c.out, slot, 0, (int64_t) aswaps.size(), x});
        }
    };
    a_back = [this](int hi) {   // step 1's copies back (D2H) from home hi's cache, then its event
        AHome& h = ahomes[(size_t) hi];
        bool any = false;
        {
            const strata::core::OnDevice on(h.dev);
            for (const ASwap& w : aswaps) {
                if (w.home != hi || !w.exchange) continue;
                if (const cudaError_t e = cudaMemcpyAsync(src.exchange_buffer(w.j, w.layer), h.cache->device_slot(w.slot),
                                                          (size_t) strata::kernels::cpu::expert_layout().blob_bytes(w.layer),
                                                          cudaMemcpyDeviceToHost, h.stream); e != cudaSuccess) {
                    std::fprintf(stderr, "strata serve: adaptive tier: copy back of layer %d slot %d (CUDA%d) failed: %s\n",
                                 (int) w.layer, (int) w.slot, h.dev, cudaGetErrorString(e));
                    a_err = true;
                }
                any = true;
            }
        }
        if (any) a_mark(h);
    };
    a_from = [this](const ASwap& w) -> const uint8_t* {   // step 2: where `in` is copied from (host work)
        if (a_arena) {   // the arena's blob, pinned or not (a pageable one blocks the job thread, not the host)
            const uint8_t* b = arena_src.blob_at(w.layer, w.in);   // (blob() counts reads_: not from this thread)
            if (b == nullptr) {
                std::fprintf(stderr, "strata serve: adaptive tier: layer %d expert %d is not in the arena\n",
                             (int) w.layer, (int) w.in);
                a_err = true;
            }
            return b;
        }
        if (src.pinned(w.layer, w.in))
            if (const uint8_t* b = src.resident_blob(w.layer, w.in)) return b;
        // pageable, or not in the RAM copy: through a bounce buffer (an async copy)
        uint8_t* bb = src.exchange_buffer(a_cap + w.j);
        if (bb == nullptr || !src.copy_blob(w.layer, w.in, bb)) {
            std::fprintf(stderr, "strata serve: adaptive tier: layer %d expert %d could not be read\n", (int) w.layer,
                         (int) w.in);
            a_err = true;
            return nullptr;
        }
        return bb;
    };
    a_in_one = [this](const ASwap& w, const uint8_t* b) {   // step 2: one copy in (H2D), on its home's stream
        AHome& h = ahomes[(size_t) w.home];
        const strata::core::OnDevice on(h.dev);
        if (const cudaError_t e = cudaMemcpyAsync(h.cache->device_slot(w.slot), b,
                                                  (size_t) strata::kernels::cpu::expert_layout().blob_bytes(w.layer),
                                                  cudaMemcpyHostToDevice, h.stream); e != cudaSuccess) {
            std::fprintf(stderr, "strata serve: adaptive tier: copy of layer %d into slot %d (CUDA%d) failed: %s\n",
                         (int) w.layer, (int) w.slot, h.dev, cudaGetErrorString(e));
            a_err = true;
        }
    };
    a_choose = [this]() {   // step 1 on the job thread (the serial loop)
        a_pick();
        for (AHome& h : ahomes) h.used = false;
        for (size_t hi = 0; hi < ahomes.size(); ++hi) a_back((int) hi);
    };
    a_copy_in = [this]() {   // step 2 on the job thread (the serial loop): each source read, then copied in
        std::vector<uint8_t> used(ahomes.size(), 0);
        for (size_t i = a_lo; i < a_hi; ++i) {   // (the batch; all the round's swaps unless the arena caps it)
            const ASwap& w = aswaps[i];
            if (const uint8_t* b = a_from(w)) {
                a_in_one(w, b);
                used[(size_t) w.home] = 1;
            }
        }
        for (size_t hi = 0; hi < ahomes.size(); ++hi)
            if (used[hi]) a_mark(ahomes[hi]);
    };
    a_post_batch = [this]() {   // step 2: the next batch of copies in goes to the job thread (all of them but for the arena's cap)
        a_lo = a_bi;
        a_hi = a_batch > 0 ? std::min(aswaps.size(), a_bi + a_batch) : aswaps.size();
        a_bi = a_hi;
        ajob->post(a_copy_in);
    };
    a_ready = [this](bool wait) -> bool {   // the job and its copies are done
        if (wait) ajob->wait();
        else if (!ajob->idle()) return false;
        for (AHome& h : ahomes)
            if (h.used) {
                const strata::core::OnDevice on(h.dev);
                if (wait) { if (cudaEventSynchronize(h.ev) != cudaSuccess) a_err = true; }
                else if (const cudaError_t q = cudaEventQuery(h.ev); q == cudaErrorNotReady) return false;
                else if (q != cudaSuccess) a_err = true;
            }
        return true;
    };
    // kvg_ensure's `quiesce` (and any caller about to give cache slots away): nothing runs on the device, the blocking
    // tier's swaps have landed AND the asynchronous round in flight is finished - a swap still copying could write a
    // slot the K/V is lent.  A failed refill is surfaced as ERR at the next tick (a_err).
    adapt_quiesce = [this] {
        cudaDeviceSynchronize();
        apply_pending(true);
        if (!adapt_tick(true)) {   // a failed round still has copies in flight: they land before any slot moves
            a_err = true;
            if (ajob) ajob->wait();
            cudaDeviceSynchronize();
        }
    };
    // #477: write the learned profile (between requests and at QUIT: a prompt's lent slots are back by then).
    // A swap still in flight counts as done - its expert is resident once the copy lands.  `why`: for the log.
    save_profile = [this](const char* why) {
        if (heat.empty()) return;
        std::vector<uint8_t> resident(host_res.size(), 0);
        for (size_t i = 0; i < host_res.size(); ++i) resident[i] = host_res[i] >= 0;
        for (const auto& p : pending) resident[(size_t) p.first] = 1;
        if (astate == AState::SwapIn)   // --adapt-async: the same for a round past its step 2
            for (const ASwap& w : aswaps) resident[(size_t) w.layer * g.n_expert + w.in] = 1;
        std::string e;
        const auto ranked = strata::core::rank_learned_profile(g.n_layers, g.n_expert, resident, heat,
                                                               profile_loaded);
        if (strata::core::write_expert_profile(o.expert_profile_save, g.n_layers, g.n_expert, ranked, e))
            std::fprintf(stderr, "strata serve: expert profile saved to %s (%s)\n", o.expert_profile_save.c_str(),
                         why);
        else
            std::fprintf(stderr, "strata serve: the expert profile was not saved: %s\n", e.c_str());
        profile_saved_at = Clock::now();
    };
    sp->should_stop = [this] { return stop_requested(); };
    trace = std::getenv("STRATA_TRACE") != nullptr;
    {
        // what is left once everything is allocated: under WDDM a GPU filled to the brim does not fail, it pages -
        // and a page-in while the verify graph spins on a host flag stalls the request for good
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        // below ~256 MiB a later allocation (a first-used window's buffers, the desktop, another program) can make
        // the driver page GPU memory, and a verify graph spinning on a host flag then never finishes
        const int64_t free_mib = (int64_t) (free_b >> 20);
        if (free_mib >= 256) {
            std::fprintf(stderr, "strata serve: %lld MiB of VRAM free with everything loaded\n", (long long) free_mib);
            rss_probe("serving");
        } else if (reserve_adapted) {
            // #496: the reserve was already lowered to make the cache fit - a bigger one would leave it no room
            std::fprintf(stderr, "strata serve: WARNING: %lld MiB of VRAM free with everything loaded - this card "
                                 "only just fits the model (the VRAM reserve was lowered to %d MiB so the expert "
                                 "cache fits): requests may stall. Close other programs that use the GPU, or lower "
                                 "--max-context\n", (long long) free_mib, o.vram_reserve_mib);
        } else {
            std::fprintf(stderr, "strata serve: %lld MiB of VRAM free with everything loaded - LOW: requests may stall;"
                                 " add --vram-reserve-mib %lld to the config's args (or lower --max-context)\n",
                         (long long) free_mib, (long long) (o.vram_reserve_mib + 512 - free_mib));
            // #781 / #831: an explicit --expert-cache N is a byte budget that is not checked against the VRAM once
            // the slots are written (auto is), so on a card it fills a smaller N is the cure: a few hundred slots
            // can be the difference between 14 and 100 tok/s under WDDM.  Said, never changed.
            if (!auto_cache)
                std::fprintf(stderr, "strata serve: the expert cache was set with --expert-cache (%lld slots): a "
                                     "smaller one - or --expert-cache auto, which checks the free VRAM once the "
                                     "slots are written - leaves room; decode can be several times slower with the "
                                     "card this full\n", (long long) xcache.slots());
        }
    }
    // what the server's Monitor tab shows (servers before 0.1.8 skip unknown lines until READY)
    {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const int64_t slots_primary = (int64_t) xcache.slots();
        const int64_t mib_primary = (int64_t) (xcache.bytes() >> 20);
        info = string_printf("logit_bias=1 context=%lld kv=%s kv_resident=%lld expert_slots=%lld expert_cache_mib=%lld "
                             "expert_slots_primary=%lld expert_cache_primary_mib=%lld spec=%d "
                             "mtp_max=%d lookup=%d vram_free_mib=%lld cvec=0 arena_mib=%lld pool_workers=%d pcie_frac=%.2f "
                             "spec_min_p=%.2f conversation_cache_mib=0 conversation_cache_slots=4 "
                             "conversation_cache_min_free_mib=2560 conversation_cache_min_tokens=0 "
                             "tail_role_token=%lld vram_elastic=%d "
                             "prefill_cap=%lld prefill_stages=%d stream_all_min=%lld prefill_pipe=%g engine=" STRATA_VERSION,
                             (long long) o.max_context, o.kv.c_str(),
                             (long long) (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1
                                              ? ss.qsa_states[ss.qsa_primary()].n_slots * 4 : 0),
                             (long long) slots_primary, (long long) mib_primary,
                             (long long) slots_primary, (long long) mib_primary,
                             o.spec, o.mtp_max_t,
                             o.suffix_draft, (long long) (free_b >> 20),
                             (long long) ((o.mmap_experts ? src.resident_bytes() : strata::kernels::cpu::expert_layout().total) >> 20),
                             pool->workers(), o.pcie_frac,
                             o.spec_min_p,
                             (long long) o.tail_role_token, xcache.segmented() ? 1 : 0,
                             // the prompt path: its chunk cap, the stages a prompt flows through, the chunk size from which a
                             // chunk streams every expert, the prompt pipe setting
                             (long long) o.prefill_chunk, sp->stage_count(),
                             (long long) strata::prefill::Prefill::stream_all_min_tokens(),
                             strata::prefill::Prefill::pipe_setting());
    }
    S = o.spec;
    S_mtp = o.mtp_max_t > 0 ? std::min(o.mtp_max_t, S) : S;
    if (use_mtp && S_mtp < S) mtp.set_max_drafts(S_mtp - 1);
    else if (use_mtp && mtp.max_t() > S_mtp) mtp.set_max_drafts(S_mtp - 1);
    sfx.emplace(std::max(1, o.suffix_draft), 64, (size_t) o.max_context + 4096);
    policy.emplace(S);
    ready = true;
    return 0;
}

void EngineState::tr(const char* what, long long a, long long b) {
    if (!trace) return;
    std::fprintf(stderr, "strata trace: %s %lld %lld\n", what, a, b);
    std::fflush(stderr);
}

bool EngineState::adapt_tick(bool drain, bool count) {
    if (!ajob) return true;
    if (!drain && count) ++a_win;   // one tick per window
    for (;;) {
        if (a_err.load()) return false;
        switch (astate) {
        case AState::Idle:
            if (drain || a_win - a_last < (int64_t) o.adapt_every) return true;
            a_last = a_win;
            ++a_rounds;
            a_t0 = Clock::now();
            a_usage = drive.d.usage;
            a_res = host_res;
            // #477: the routing counted, as adapt() counts it (--expert-profile-save; else `heat` is empty)
            for (size_t i = 0; i < heat.size(); ++i) heat[i] += (double) drive.d.usage[i];
            for (float& v : drive.d.usage) v *= o.adapt_decay;
            if (a_inline) {   // the arena: choose here (no copy back to wait for), then straight on to step 2
                const Clock::time_point tp0 = Clock::now();
                a_pick();
                a_pick_ms += std::chrono::duration<double, std::milli>(Clock::now() - tp0).count();
                for (AHome& h : ahomes) h.used = false;
                astate = AState::CopyBack;
                continue;
            }
            ajob->post(a_choose);
            astate = AState::CopyBack;
            return true;
        case AState::CopyBack: {   // step 2: the evicted experts leave the GPU's plan
            if (!a_ready(drain)) return true;
            if (a_err.load()) return false;
            size_t k = 0;
            for (size_t i = 0; i < aswaps.size(); ++i) {
                const ASwap w = aswaps[i];
                const size_t out = (size_t) w.layer * g.n_expert + w.out;
                if (host_res[out] != w.slot) continue;   // the table moved since the snapshot: not this one
                if (w.exchange && !src.stage_exchange(w.layer, w.in, w.out, w.j)) {
                    std::fprintf(stderr, "strata serve: adaptive tier: stage_exchange refused layer %d in %d out %d\n",
                                 (int) w.layer, (int) w.in, (int) w.out);
                    return false;
                }
                host_res[out] = strata::core::kNotResident;   // the CPU computes it from the next window on
                srcp->prefetch(w.layer, w.out);
                aswaps[k++] = w;
            }
            aswaps.resize(k);
            if (aswaps.empty()) {   // nothing to move (converged, or the table moved since the snapshot): the round ends here
                a_ms += std::chrono::duration<double, std::milli>(Clock::now() - a_t0).count();
                astate = AState::Idle;
                continue;
            }
            for (AHome& h : ahomes) h.used = false;
            res_upload();
            a_lo = a_hi = a_bi = 0;
            a_post_batch();
            astate = AState::SwapIn;
            if (!drain) return true;
            continue;
        }
        case AState::SwapIn:   // step 3: the new experts are in their slots
            if (!a_ready(drain || a_wait)) return true;
            if (a_err.load()) return false;
            for (size_t i = a_lo; i < a_hi; ++i) {   // (the batch just landed)
                const ASwap& w = aswaps[i];
                host_res[(size_t) w.layer * g.n_expert + w.in] = w.slot;
                srcp->release(w.layer, w.in);
            }
            res_upload();
            a_swapped += (int64_t) (a_hi - a_lo);
            if (a_bi < aswaps.size()) {   // the arena's cap: the next batch goes now
                for (AHome& h : ahomes) h.used = false;
                a_post_batch();
                if (!drain) return true;
                continue;
            }
            astate = AState::RamFence;
            continue;
        case AState::RamFence:   // step 3's RAM moves, once no window may still read the places they fill
            if (a_arena) {   // no RAM copy to move
                astate = AState::Commit;
                continue;
            }
            astate = AState::Commit;
            if (src.exchange_rotation()) continue;   // STRATA_EXCHANGE_ROTATE: no bytes move, the flip is all there is
            ajob->post([&] { src.commit_copies(); });
            if (!drain) return true;
            continue;
        case AState::Commit:   // step 4
            if (!a_ready(drain)) return true;
            if (a_err.load()) return false;
            if (!a_arena) src.commit_flip();
            a_ms += std::chrono::duration<double, std::milli>(Clock::now() - a_t0).count();
            aswaps.clear();
            astate = AState::Idle;
            continue;   // a round due now starts at this same boundary
        }
    }
}

DecSnap EngineState::dec_snap() const {
    return DecSnap{ver.ms_wait, ver.ms_pool, ver.ms_host, drive.d.ms_plan, drive.d.ms_actq, drive.d.ms_jobs,
                   drive.d.ms_run, drive.d.multi_misses, drive.d.multi_entries, drive.d.cache_hits,
                   drive.d.pcie_experts};
}

// where PP was printed: the prompt's progress for the sync call's callback
void EngineState::report_progress(int64_t done) {
    if (!progress_fn) return;
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
    progress_fn(SyncProgress{done, pp_total, pp_from, ms});
}

bool EngineState::stop_requested() const { return cancel_fn && cancel_fn(); }

int EngineState::refuse(std::string message) {
    session_error = std::move(message);
    return kRefused;
}

int EngineState::die(std::string message) {
    dead = true;
    session_error = std::move(message);
    busy_scope.reset();
    return kDead;
}

int EngineState::sync_request(const int32_t* tokens, int64_t n, const ImageSpan* images, size_t image_count) {
    if (dead) return kDead;
    if (n < 1 || n >= session_ctx)
        return refuse("a prompt of " + std::to_string((long long) n) + " tokens does not fit a session of " +
                      std::to_string((long long) session_ctx));
    const std::vector<int64_t> ids(tokens, tokens + n);
    bool bad = false;
    for (int64_t t : ids) bad = bad || t < 0 || t >= n_vocab;
    if (bad) return refuse("a token id is outside the vocabulary");
    const bool geni = image_count > 0;
    if (geni && !o.vision) return refuse("this engine was started without --vision");
    std::vector<ImgKey> imgs_now;
    if (geni || !mrope_identity) {
        // positions for every cell this request can reach; the identity again for a text request
        std::string ve;
        int64_t row_floats = 0;
        row_ptr.assign((size_t) n, nullptr);
        const int64_t cells = (int64_t) mrope_host.size() / 3;
        auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
            mrope_host[(size_t) c * 3] = (int32_t) t;
            mrope_host[(size_t) c * 3 + 1] = (int32_t) h;
            mrope_host[(size_t) c * 3 + 2] = (int32_t) w;
        };
        if (!geni) {
            for (int64_t c = 0; c < cells; ++c) put(c, c, c, c);
        } else {
            for (size_t k = 0; k < image_count && ve.empty(); ++k) {
                const ImageSpan& im = images[k];
                if (im.rows == nullptr || im.token_count < 1 || im.grid_width < 1 || im.grid_height < 1 ||
                    im.grid_width * im.grid_height != im.token_count)
                    ve = "image " + std::to_string(k + 1) + ": its rows are not token_count = grid_width x grid_height";
                row_floats += im.token_count * g.n_embd;
            }
            int64_t p = 0, i = 0;
            size_t k = 0;
            while (ve.empty() && i < n) {
                if (ids[(size_t) i] != kImagePad) { put(i, p, p, p); ++p; ++i; continue; }
                if (k >= image_count) { ve = "the prompt has more images than the image list"; break; }
                const ImageSpan& im = images[k++];
                if (im.start != i) {
                    ve = "image " + std::to_string(k) + " does not start at its first <|image_pad|> token";
                    break;
                }
                {   // what the conversation cache compares: a picture is its grid and its key (#2228)
                    const int64_t grid[3] = {im.token_count, im.grid_width, im.grid_height};
                    uint64_t h = fnv1a(grid, sizeof grid);
                    h = fnv1a(&im.key, sizeof(uint64_t), h);
                    imgs_now.push_back({i, h});
                }
                for (int64_t j = 0; j < im.token_count && ve.empty(); ++j)
                    if (i + j >= n || ids[(size_t) (i + j)] != kImagePad)
                        ve = "image " + std::to_string(k) + " has " + std::to_string((long long) im.token_count) +
                             " rows but fewer <|image_pad|> tokens";
                for (int64_t j = 0; j < im.token_count && ve.empty(); ++j) {
                    const int64_t y = j / im.grid_width, x = j % im.grid_width;
                    put(i + j, p, p + y, p + x);
                    row_ptr[(size_t) (i + j)] = im.rows + (size_t) j * (size_t) g.n_embd;
                }
                i += im.token_count;
                p += std::max(im.grid_width, im.grid_height);
            }
            if (ve.empty() && k != image_count) ve = "the image list has more images than the prompt";
            if (ve.empty() && n > 0 && ids[(size_t) (n - 1)] == kImagePad) ve = "the prompt cannot end in an image";
            for (int64_t c = n; ve.empty() && c < cells; ++c) put(c, p + (c - n), p + (c - n), p + (c - n));
        }
        tr("positions built", (long long) row_floats);
        cudaDeviceSynchronize();
        tr("device idle");
        auto upload_mrope = [&]() -> bool {
            return cudaMemcpy(d_mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice) == cudaSuccess;
        };
        if (ve.empty() && !upload_mrope()) ve = "the image position upload failed";
        if (!ve.empty()) {
            // leave the table as the identity so the next text request is untouched
            for (int64_t c = 0; c < cells; ++c) put(c, c, c, c);
            upload_mrope();
            mrope_identity = true;
            return refuse(ve);
        }
        mrope_identity = !geni;
    }
    if (request_open && finish_request() != kOk) return kDead;
    req_imgs = std::move(imgs_now);
    sp->embd_rows = geni ? row_ptr.data() : nullptr;
    // a request may touch the expert cache (it lends slots, a K/V growth moves them): a round of the
    // asynchronous tier left in flight lands first
    if (!adapt_tick(true)) return die("an adaptive refill failed");
    // #477: every --expert-profile-save-every minutes, before the next request (at QUIT: at close)
    if (!heat.empty() && o.expert_profile_save_min > 0 &&
        Clock::now() - profile_saved_at >= std::chrono::duration<double>(o.expert_profile_save_min * 60.0))
        save_profile("periodic");
    busy_scope.emplace();
    {   // #2065: said, never changed - the free VRAM now (the verify windows' captures and the batch layouts
        // eat into what startup reported), once per 128 MiB lower it gets
        size_t vf_b = 0, vt_b = 0;
        if (cudaMemGetInfo(&vf_b, &vt_b) == cudaSuccess) {
            const int64_t vf = (int64_t) (vf_b >> 20);
            static int64_t low_said_at = INT64_MAX;
            if (vf < kVramSafeMib && vf + 128 <= low_said_at) {
                low_said_at = vf;
                std::fprintf(stderr, "strata serve: WARNING: only %lld MiB of VRAM free at the start of this request "
                                     "(below %lld MiB): when the card is this full the driver can page GPU memory "
                                     "to system RAM, decode falls to a few tok/s and a window may never finish "
                                     "(#2065). Add --vram-reserve-mib %lld to the config's args, or lower "
                                     "--max-context\n", (long long) vf, (long long) kVramSafeMib,
                             (long long) ((o.vram_reserve_mib + (kVramSafeMib - vf) + 63) / 64 * 64));
            }
        }
    }
    err.clear();
    // the request keys a session has no field for (ckpt, pcie_frac, spec_min_p, aux_cpus): the start-up values
    req_ckpt = 1;
    req_pcie_frac = o.pcie_frac;
    req_spec_min_p = o.spec_min_p;
    const int req_aux_cpus = -1;
    const bool req_pcie_set = false;
    request_open = true;
    cur = ids;
    const Clock::time_point r0 = Clock::now();
    streamed0 = sp->chain_experts_streamed();   // the DONE line's experts_streamed= is the difference
    // ---- where this request starts reading: the live session, or a checkpoint, whose tokens AND pictures are
    // exactly the start of this prompt - at most n - 1 of them, the last token is always the first window
    auto starts_with = [&](const std::vector<int32_t>& pre, const std::vector<ImgKey>& pre_imgs) -> bool {
        const int64_t L = (int64_t) pre.size();
        if (L < 1 || L > n - 1) return false;
        for (int64_t i = 0; i < L; ++i)
            if ((int32_t) ids[(size_t) i] != pre[(size_t) i]) return false;
        return imgs_below(req_imgs, L) == pre_imgs;
    };
    // the last request's final commit may still be running on the verifier's stream (set_commit_async):
    // everything below reads, restores or zeroes the session from other streams and the host (the end of the
    // last request waited already; this covers a request that ended on an error path)
    if (!ver.wait_commit(err)) return die(err);
    if (o.prefill_head_alias && native_head.loaded()) {
        const auto head_t0 = Clock::now();
        native_head.unload();
        sp->reset();
        if (init_prompt_paths() != 0) return die("rebuilding the head-alias prompt path failed: " + err);
        std::fprintf(stderr, "strata serve: --prefill-head-alias: prompt workspace rebuilt in %.1f ms\n",
                     std::chrono::duration<double, std::milli>(Clock::now() - head_t0).count());
    }
    auto kv_quiesce = adapt_quiesce;
    if (kvg.on) {   // the elastic K/V: this prompt's cells (it gives back what it does not need below)
        // a failed growth may leave the tier half-changed: the engine stops (as for any CUDA failure)
        if (!kvg_ensure(n + 256, kv_quiesce)) return die("the K/V cannot grow to this prompt: no VRAM is left");
    }
    nf_hit = false;
    // #879: an earlier window met non-finite logits.  The poison sits in a conversation's K/V and recurrent
    // state, so nothing that may hold it is continued: the live session and every checkpoint go, and this
    // prompt is read from token 0 (resume == 0 below)
    if (nan_flush) {
        nan_flush = false;
        const size_t n_checks = checks.size();
        live_ok = false;
        checks.clear();
        tail_ckpt_len = -1;
        std::fprintf(stderr, "strata serve: after non-finite logits (#879): dropped the live session, %zu "
                             "checkpoint%s, 0 parked conversations and 0 idle slots (0 running slots "
                             "not kept after their reply); reading this prompt from token 0\n", n_checks,
                     n_checks == 1 ? "" : "s");
    }
    resume = 0;
    bool from_live = false;
    if (o.prompt_cache > 0) {
        if (live_ok && starts_with(live, live_imgs)) { resume = (int64_t) live.size(); from_live = true; }
        for (const ConvCheckpoint& c : checks)
            if ((int64_t) c.ids.size() > resume && starts_with(c.ids, c.imgs)) {
                resume = (int64_t) c.ids.size();
                from_live = false;
            }
    }
    // #2228: a checkpoint or the live state whose TOKENS this prompt starts with, deeper than what is reused, but
    // whose pictures differ (a picture is its grid and the hash of its embeddings: an image encoded again with
    // other bits - another encoder start, a CPU encoder that is not bit-reproducible - no longer matches).  The
    // reuse then falls back to a checkpoint before that picture; said once per request, so the log tells this
    // apart from a client that rewrote its history.
    if (o.prompt_cache > 0 && !req_imgs.empty()) {
        int64_t deep = 0;
        const std::vector<ImgKey>* deep_imgs = nullptr;
        auto tokens_match = [&](const std::vector<int32_t>& pre) {
            const int64_t L = (int64_t) pre.size();
            if (L < 1 || L > n - 1) return false;
            for (int64_t i = 0; i < L; ++i)
                if ((int32_t) ids[(size_t) i] != pre[(size_t) i]) return false;
            return true;
        };
        auto consider = [&](const std::vector<int32_t>& pre, const std::vector<ImgKey>& pre_imgs) {
            if ((int64_t) pre.size() > std::max(deep, resume) && tokens_match(pre) &&
                imgs_below(req_imgs, (int64_t) pre.size()) != pre_imgs) {
                deep = (int64_t) pre.size();
                deep_imgs = &pre_imgs;
            }
        };
        if (live_ok) consider(live, live_imgs);
        for (const ConvCheckpoint& c : checks) consider(c.ids, c.imgs);
        if (deep_imgs != nullptr) {
            const std::vector<ImgKey> now = imgs_below(req_imgs, deep);
            size_t k = 0;
            while (k < now.size() && k < deep_imgs->size() && now[k] == (*deep_imgs)[k]) ++k;
            std::fprintf(stderr, "strata serve: conversation cache: the state at %lld tokens has this prompt's "
                                 "tokens but picture %zu (at token %lld) is not the same as when it was saved "
                                 "(its embeddings changed): it cannot be reused (#2228)\n", (long long) deep,
                         k + 1, (long long) (k < now.size() ? now[k].start : -1));
        }
    }
    // this request rewrites every cell from `resume` on, so a checkpoint past it (or not on this prompt's
    // path) no longer has its cells; the ones kept are prefixes of both the old tokens and the new
    auto off_path = [&](const ConvCheckpoint& c) {
        return (int64_t) c.ids.size() > resume || !starts_with(c.ids, c.imgs);
    };
    if (ckpt_pool->on())   // STRATA_CKPT_POOL: their storage is the next checkpoint's
        ckpt_pool->erase_if(checks, off_path);
    else
        checks.erase(std::remove_if(checks.begin(), checks.end(), off_path), checks.end());
    live_ok = false;   // until this request has finished, the session is in between
    // the elastic K/V: this request rewrites every cell from `resume` on, so cells past this prompt's are no
    // longer anyone's - far more than it needs go back to the cache
    if (kvg.on) {
        kv_quiesce();
        if (!kvg_trim(n + 256)) return die("the K/V could not give its VRAM back to the expert cache");
    }
    int64_t reread_to = -1;   // STRATA_CKPT_REREAD only: read [0, reread_to) again instead of restoring
    if (resume == 0) {
        strata::core::session_zero(ss, g, nullptr, main_cs);
        cudaStreamSynchronize(main_stream);
        checks.clear();
    } else if (!from_live) {
        ConvCheckpoint* c = nullptr;
        for (ConvCheckpoint& k : checks) if ((int64_t) k.ids.size() == resume) c = &k;
        if (c != nullptr) c->used = ++check_clock;   // mounting through it is the use LRU counts
        static const bool reread = std::getenv("STRATA_CKPT_REREAD") != nullptr;
        if (reread && c != nullptr) {
            // THE CHECK OF THE CHECKPOINT: instead of restoring it, read its tokens again from position 0 in
            // one run (below, with the prompt path's slots lent like any read) - the same chunks the request
            // that saved it read them in, when that request started at 0.  With the VRAM expert set fixed
            // (--adapt-swaps 0) the answer must match the restored one token for token; anything the
            // checkpoint missed shows up as a difference.
            strata::core::session_zero(ss, g, nullptr, main_cs);
            cudaStreamSynchronize(main_stream);
            reread_to = resume;
            std::fprintf(stderr, "strata serve: STRATA_CKPT_REREAD: reading %lld tokens again instead of "
                                 "restoring\n", (long long) resume);
        } else if (c == nullptr || !checkpoint_restore(*c, ss, g)) {
            return die("restoring a conversation checkpoint failed");
        }
    }
    // KV streaming: the drafter's ring may hold cells past `resume` from a longer turn; the main layers'
    // host copies and slots are always current (every writer writes both), so they need nothing
    if (use_mtp && resume > 0 && reread_to <= 0) mtp.kv_restore(resume);
    tr("request", n, geni ? 1 : 0);
    req_times = ReqTimes{};
    if (use_mtp) mtp.set_prompt_len(n);
    const int64_t read_from = reread_to > 0 ? 0 : resume;
    pp_total = n;
    pp_from = read_from;
    pp_reached = read_from;
    pp_exact = true;   // a restore, a checkpoint mount or a zeroed session: the state is exactly at read_from
    pp_t0 = r0;
    pp_next_check = reread_to > 0 || !req_ckpt ? INT64_MAX : resume + o.prompt_cache_every;   // ckpt=0: none
    pp_tail_saved = reread_to > 0 || !req_ckpt;   // ckpt=0 (#861): no tail checkpoint either
    report_progress(resume);   // before reading: this many prompt tokens are reused
    strata::core::progress_at("reading the prompt, from token", read_from);
    // A SHORT PART OF THE PROMPT - the new message of a chat that continues from a checkpoint, the assistant
    // header - goes through the verify windows, S tokens at a time, as decode reads them.  The batched path
    // costs ~300 ms per run however few tokens it has (it streams every expert the chunk routes to that is
    // not in VRAM over PCIe), and it borrows slots it must refill after (~180 ms); a window costs ~16 ms a
    // token, with the misses on the CPU.  Each part below is decided on its own, so a long first message is
    // read batched and its header still goes through the windows.  Picture rows need the batched path.
    // STRATA_CKPT_REREAD compares a restored checkpoint with a batched re-read, so it keeps every read batched.
    static const bool no_short = std::getenv("STRATA_CKPT_REREAD") != nullptr;
    auto windows_ok = [&](int64_t a, int64_t b) -> bool {
        if (no_short || b - a > o.short_read) return false;
        if (sp->embd_rows != nullptr)
            for (int64_t i = a; i < b; ++i)
                if (sp->embd_rows[i] != nullptr) return false;
        return true;
    };
    // tokens [a, b) through the windows: commit all of them, then give the draft layer their residuals
    auto read_windows = [&](int64_t a, int64_t b, std::string& e) -> bool {
        strata::core::progress_at("reading the prompt (verify windows), from token", a);   // #217: not "batched"
        // every token is committed and the picks are discarded: no head sampling (see set_head_sampling)
        struct NoHeadSampling {
            strata::core::Verifier& v;
            explicit NoHeadSampling(strata::core::Verifier& x) : v(x) { v.set_head_sampling(false); }
            ~NoHeadSampling() { v.set_head_sampling(true); }
        } no_head_sampling(ver);
        std::vector<int32_t> win((size_t) S), outw((size_t) S), nxt((size_t) S);
        pp_exact = true;   // between windows the session is exactly at `a`
        for (int64_t q = a; q < b;) {
            if (stop_requested()) { e = "cancelled"; return false; }
            pp_exact = false;   // until this window and the drafter's rows are in
            const int T = (int) std::min<int64_t>(S, b - q);
            for (int t = 0; t < T; ++t) {
                win[(size_t) t] = (int32_t) cur[(size_t) (q + t)];
                nxt[(size_t) t] = (int32_t) cur[(size_t) (q + t + 1)];
            }
            drive.d.layers = 0;
            drive.d.experts = 0;
            drive.d.failed = false;
            if (!ver.run(T, win.data(), q, win_pool_fn, win_pool_user, outw.data(), e) || drive.d.failed) {
                if (drive.d.failed && drive.d.fail) e = drive.d.fail;
                return false;
            }
            if (nonfinite_hit(ver, "prompt")) {   // (its commit is not needed: the next request reads from 0)
                e = "non-finite logits (#879)";
                return false;
            }
            // STRATA_LOGPOS=<path>: the teacher-forced log-probability of every token read here (every
            // token is committed and nxt[t] is the prompt's own next token), appended to <path>; the last
            // column is the log-probability of STRATA_LOGPOS_EXTRA (default 248046, <|im_end|>), so the
            // end-of-turn mass a chat model puts on raw text can be taken out of the measurement
            static std::FILE* logpos = [] {
                const char* p = std::getenv("STRATA_LOGPOS");
                return p != nullptr ? std::fopen(p, "ab") : nullptr;
            }();
            static const int32_t logpos_extra = [] {
                const char* p = std::getenv("STRATA_LOGPOS_EXTRA");
                return p != nullptr ? (int32_t) std::atoi(p) : (int32_t) 248046;
            }();
            if (logpos != nullptr && !ver.window_logprobs(nxt.data(), T, q, logpos_extra, logpos, e))
                return false;
            if (!ver.commit(T, e) || (use_mtp && !mtp.prefill(ver.final_R_all(), nxt.data(), T, q, e))) return false;
            q += T;
            pp_reached = q;   // #471
            pp_exact = true;
        }
        // the batched prompt path (other streams), checkpoints and snapshots may follow: the last commit first
        if (!ver.wait_commit(e)) return false;
        report_progress(b);
        strata::core::progress_beat();
        return true;
    };
    // the batched path's slots are lent just before its first run and given back (refilled) before a window
    // reads - so the windows always see the whole expert cache - or once the prompt is read
    // Every participant gives its loan back here: the rows it lent are refilled into the SAME slots from
    // the arena, the residency table is restored, and one upload puts it on every device.  A stage refills
    // through its own cache and its own device - a slot refilled into the wrong cache would leave that
    // stage's cache holding an expert it does not own, which is silent and produces plausible tokens.
    // STRATA_REFILL_PARTS=1 (gfx1151's table): a lent slot of a GGUF read in place is refilled from the pieces in the mapping.
    // The default assembles the blob in a staging buffer first (a 4 MB copy on this thread), then copies that.
    static const bool refill_parts = [] { const char* v = std::getenv("STRATA_REFILL_PARTS"); return v != nullptr && std::atoi(v) != 0; }();
    auto refill_issue = [&](PfPart& p, std::string& e) -> bool {
        tr("refill start", (long long) p.lent.size());
        const strata::core::OnDevice on(p.dev);
        for (const auto& [i, slot] : p.lent) {   // D-4: queued, one wait (STRATA_REFILL_BLOCKING=1: each)
            strata::core::BlobPart parts[3];
            const int np = refill_parts && !refill_blocking() ? srcp->blob_parts(i / g.n_expert, i % g.n_expert, parts) : 0;
            if (np > 0) {   // the GGUF read in place: the tensors' pieces go from the mapping to the slot, no assembled copy
                if (!p.cache->fill_slot_queued_parts(slot, parts, np, e)) return false;
                host_res[(size_t) i] = slot;
                continue;
            }
            const uint8_t* b = srcp->blob_stable(i / g.n_expert, i % g.n_expert);
            const int64_t nb = (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(i / g.n_expert);
            if (b == nullptr || !(refill_blocking() ? p.cache->fill_slot_blocking(slot, b, e, nb)
                                                    : p.cache->fill_slot_queued(slot, b, e, nb)))
                return false;
            if (!refill_blocking()) srcp->note_async_read(b, nullptr);   // #1237
            host_res[(size_t) i] = slot;
        }
        return true;
    };
    auto refill_wait = [&](PfPart& p, std::string& e) -> bool {
        const strata::core::OnDevice on(p.dev);
        if (!p.cache->sync_queued(e)) return false;
        p.lent.clear();
        p.lent_chunk = 0;
        return true;
    };
    auto refill_one = [&](PfPart& p, std::string& e) -> bool {
        if (p.lent.empty()) return true;
        return refill_issue(p, e) && refill_wait(p, e);
    };
    // #340: with several stages every stage's copies go out first (each card on its own link), then each is
    // waited for - the same copies into the same slots, so the same cache; one stage: refill_one exactly.
    // STRATA_REFILL_SERIAL=1: stage after stage, as 0.1.30/0.1.31.
    static const bool refill_serial = std::getenv("STRATA_REFILL_SERIAL") != nullptr;
    auto refill = [&](std::string& e) -> bool {
        const auto t_rf = Clock::now();
        int64_t n_lent = 0, n_parts = 0;
        for (PfPart& p : pf_parts)
            if (!p.lent.empty()) { n_lent += (int64_t) p.lent.size(); ++n_parts; }
        if (n_parts > 1 && !refill_serial) {
            for (PfPart& p : pf_parts)
                if (!p.lent.empty() && !refill_issue(p, e)) return false;
            for (PfPart& p : pf_parts)
                if (!p.lent.empty() && !refill_wait(p, e)) return false;
        } else {
            for (PfPart& p : pf_parts)
                if (!p.lent.empty() && !refill_one(p, e)) return false;
        }
        if (n_parts > 0) res_upload();
        if (trace && n_parts > 0) {
            std::fprintf(stderr, "strata trace: refilled %lld slots on %lld stage(s) in %.1f ms\n",
                         (long long) n_lent, (long long) n_parts,
                         std::chrono::duration<double, std::milli>(Clock::now() - t_rf).count());
            std::fflush(stderr);
        }
        return true;
    };
    // lend the slots `tokens` batched prompt tokens need: the prompt path's buffers for min(chunk, tokens
    // rounded up to 256), laid out in the last of the slots it may borrow - per participant, out of that
    // participant's own cache, and marking only that participant's own layers
    int64_t seg_lend = 0;   // the loan the last lend left (the largest participant's chunk; 0: none)
    auto lend = [&](int64_t tokens, std::string& e, int64_t exact) -> bool {
        seg_lend = 0;
        if (pf_parts.empty()) return true;                     // its own buffers: nothing to lend
        const auto t_ln = Clock::now();
        // what this segment needs, capped by the configured chunk: a request lends only what its own
        // prompt needs, so a large chunk costs a short prompt nothing
        const int64_t want_full = equal_chunk(tokens, o.prefill_chunk);
        const int64_t want_seg = o.prefill_head_alias ? o.prefill_chunk : want_full;
        // `exact`: a planned segment read on - the loan it was read with (what is left would size another)
        const int64_t want = exact > 0 ? std::min(exact, o.prefill_chunk) : want_seg;
        if (want <= 0) {
            e = "prefill: cannot lend buffers for an empty request segment";
            return false;
        }
        bool any = false;
        for (PfPart& p : pf_parts) {
            if (p.first < 0) continue;
            if (!p.lent.empty()) {
                if (want <= p.lent_chunk) continue;            // its current loan already covers this
                // ONLY this participant's loan goes back: `refill` would return the other participants'
                // loans too, and their buffers are still laid out in their caches - marking those slots
                // resident again would hand the next window a prompt buffer in place of an expert
                if (!refill_one(p, e)) return false;
            }
            const strata::core::OnDevice on(p.dev);
            int64_t want_p = want;
            int32_t first = p.first_now;
            // --prefill-head-alias: the loan laid out at start (slot 1 on) is kept; a hybrid layout cannot relayout
            if (!o.prefill_head_alias) {
                // #2090: the loan a chunk needs is NOT monotonic in the chunk: a chunk above the scan's small_max
                // lays out the byte-budget ring (199 slots at 5120), one at or below it 0.1.39's (384 at 4096), so
                // a request's smaller chunk can need MORE slots than the auto chunk's region (p.first) holds.  The
                // loan starts where the chunk itself needs it (the 128-slot floor aside); a chunk that cannot be
                // lent even then is halved until it can, instead of ending the engine.
                int64_t need_p = part_slots(p, want_p);
                while ((need_p <= 0 || need_p + 128 > p.cache->slots()) && want_p > 256) {
                    want_p = std::max<int64_t>(256, want_p / 2 / 256 * 256);
                    need_p = part_slots(p, want_p);
                }
                if (need_p <= 0 || need_p + 128 > p.cache->slots()) {
                    e = "prefill: no chunk of this request fits the slots that can be lent (#2090)";
                    return false;
                }
                if (want_p != want)
                    std::fprintf(stderr, "strata serve: prompt chunk %lld -> %lld tokens for this request: its "
                                 "buffers do not fit the slots that can be lent (#2090)\n", (long long) want,
                                 (long long) want_p);
                if (need_p > p.cache->slots() - p.first)   // the case that ended the engine before (#2090)
                    std::fprintf(stderr, "strata serve: the loan for a chunk of %lld tokens (%lld slots) is larger than the "
                                 "auto chunk's (%lld slots): starting it earlier (#2090)\n", (long long) want_p,
                                 (long long) need_p, (long long) (p.cache->slots() - p.first));
                first = (int32_t) (p.cache->slots() - need_p);
            }
            if (want_p != p.sp->chunk() || first != p.first_now) {
                if (!p.sp->relayout(want_p, p.cache->device_slot(first), part_bytes(p, first), e)) return false;
                p.first_now = first;
            }
            for (int64_t l = p.lb; l < p.le; ++l)              // THIS participant's layers only
                for (int64_t ex = 0; ex < g.n_expert; ++ex) {
                    const size_t i = (size_t) (l * g.n_expert + ex);
                    if (host_res[i] >= first) {
                        p.lent.emplace_back((int32_t) i, host_res[i]);
                        host_res[i] = strata::core::kNotResident;
                        any = true;
                    }
                }
            p.lent_chunk = want_p;
        }
        for (const PfPart& p : pf_parts)
            if (p.first >= 0) seg_lend = std::max(seg_lend, p.lent_chunk);
        if (any) res_upload();
        if (trace) {
            int64_t n_lent = 0;
            for (const PfPart& p : pf_parts) n_lent += (int64_t) p.lent.size();
            std::fprintf(stderr, "strata trace: lent %lld slots for %lld tokens in %.1f ms\n", (long long) n_lent,
                         (long long) want, std::chrono::duration<double, std::milli>(Clock::now() - t_ln).count());
            std::fflush(stderr);
        }
        return true;
    };
    req_chunk = 0;
    req_chunk_len = 0;
    auto read_part = [&](int64_t a0, int64_t b0, std::string& e) -> bool {
        ConvCheckpoint::PromptPlan plan;
        const bool planned = plan.chunk > 0;
        const bool plan_single = planned && plan.single;
        if (b0 - a0 > req_chunk_len) {
            req_chunk_len = b0 - a0;
            req_chunk = planned ? std::min(plan.chunk, b0 - a0) : std::min(sp->chunk(), b0 - a0);
        }
        seg_plan = plan;   // the checkpoints taken inside it keep it; cleared when the segment is done
        struct SegPlanScope {
            ConvCheckpoint::PromptPlan& p;
            ~SegPlanScope() { p = {}; }
        } seg_plan_scope{seg_plan};
        auto run = [&](int64_t q, int64_t r) -> bool {
            if (planned) sp->force_chunk(plan.chunk, plan_single, plan.chunked);
            return sp->run(ids.data() + q, r - q, q, e);
        };
        return run(a0, b0);
    };
    apply_pending(true);
    if (!adapt_tick(true)) return die("an adaptive refill failed");   // --adapt-async: the round in flight lands before the prompt lends any slot
    {   // aux_cpus=0|1: where the non-pool, non-host threads run for this request (-1 = the start-up --aux-cpus); moves
        // threads only.  An enabled placement is swept at every request: it catches threads created since.
        static const bool aux_start = strata::aux_cpus::enabled();
        const bool was_on = strata::aux_cpus::enabled();
        const bool want = req_aux_cpus < 0 ? aux_start : req_aux_cpus != 0;
        strata::aux_cpus::set_enabled(want);
        if (want != was_on)
            std::fprintf(stderr, "strata serve: aux cpus %s for this request%s\n", strata::aux_cpus::enabled() ? "ON" : "off",
                         want && !strata::aux_cpus::enabled() ? " (asked for, but no CPU is spare: see the start-up line)" : "");
    }
    ver.pause_pcie_balance(req_pcie_set);   // the measured costs keep across requests
    drive.d.pcie_num = std::max(0, std::min(256, (int) (req_pcie_frac * 256.0 + 0.5)));
    cancelled = false;
    tr("prompt start", n - 1);
    // The prompt is read in two parts when it has a turn boundary past `resume`: up to the last <|im_start|>
    // (the conversation so far), a checkpoint there, then the new turn's header.  The next request of the same
    // chat renders the same history - but not always the same header or the thinking of this reply - so that
    // checkpoint is the one it reuses.
    int64_t turn_at = -1;
    if (o.prompt_cache > 0 && o.turn_token >= 0)
        for (int64_t i = n - 1; i > resume; --i)
            if (ids[(size_t) i] == o.turn_token) { turn_at = i; break; }
    // #458 (opt-in): a short turn of --tail-role-token's role right before the new assistant turn (the
    // server's trailing reasoning-effort turn) stays out of the checkpoint, so the next request - another
    // effort, or the next turn of the chat - finds the conversation without it
    if (turn_at > 0 && o.tail_role_token >= 0)
        for (int64_t i = turn_at - 1; i > resume; --i)
            if (ids[(size_t) i] == o.turn_token) {
                if (ids[(size_t) i + 1] == o.tail_role_token) turn_at = i;
                break;
            }
    // A prompt read from token 0 also stops at its FIRST turn boundary: the end of the system prompt (with
    // the tools), which every new chat of the same client shares.  That checkpoint becomes the chain's root,
    // which the retention policy pins (conv_cache.hpp), so the next new chat reads only what comes after it.
    // (PR #65, code-martin.)  Only for a system prompt of --prompt-cache-root tokens or more: a small one
    // is cheaper to read again than the extra part costs (~0.3 s).
    const int64_t last_turn = turn_at;   // ckpt=0 drops the split there, the root is still looked for before it
    if (!req_ckpt) turn_at = -1;
    int64_t root_at = -1;
    if (o.prompt_cache > 0 && o.turn_token >= 0 && o.prompt_cache_root > 0 && read_from == 0)
        for (int64_t i = 1; i < last_turn; ++i)
            if (ids[(size_t) i] == o.turn_token) {
                if (i >= o.prompt_cache_root) root_at = i;
                break;
            }
    static const bool message_checkpoint = [] {
        const char* e = std::getenv("STRATA_CACHE_MESSAGE_BOUNDARY");
        return e != nullptr && std::atoi(e) != 0;
    }();
    // Only add a snapshot; the existing token and image checks still decide reuse.
    const int64_t message_at = message_checkpoint && o.prompt_cache > 0
        ? strata::program::message_checkpoint_boundary(ids, resume, turn_at, o.turn_token) : -1;
    std::vector<int64_t> cuts = {reread_to, root_at, message_at, turn_at, n - 1};
    int64_t at = read_from;
    for (const int64_t to : cuts) {
        if (to <= at) continue;
        err.clear();
        const bool win = windows_ok(at, to);
        if (win && !refill(err)) return die("refilling a lent slot failed: " + err);
        if (!win && !lend(to - at, err, 0)) return die("lending the prompt path its slots failed: " + err);
        const int64_t read_to = to;
        const auto tsp = Clock::now();
        pp_exact = false;   // a batched part is in flight: a cancel may leave the state past pp_reached
        const bool sp_ok = read_to <= at || (win ? read_windows(at, read_to, err) : read_part(at, to, err));
        if (sp_ok) pp_exact = true;
        if (trace) {
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - tsp).count();
            (win ? req_times.windows_ms : req_times.read_ms) += ms;
            std::fprintf(stderr, "strata trace: read %lld tokens (%s) in %.1f ms\n", (long long) (read_to - at),
                         win ? "windows" : "batched", ms);
            std::fflush(stderr);
        }
        if (!sp_ok && nf_hit) {   // #879: DONE nonfinite (the lent slots are refilled below)
            cancelled = true;
            break;
        }
        if (!sp_ok) {
            if (!stop_requested()) {
                std::fprintf(stderr, "strata serve: %s\n", err.c_str());
                // #224: a CUDA fault (an illegal address) poisons the context for the whole process, and unwinding
                // the destructors on it could hang: a dead engine runs none of them
                return die(err);
            }
            cancelled = true;   // stopped while reading the prompt: refill the lent slots below, then DONE cancel
            break;
        }
        at = to;
        if ((to == turn_at || to == root_at || to == message_at) &&
            !checkpoint_at(to, to == message_at && to != turn_at && to != root_at))   // the tail kind leaves first
            return die("saving a conversation checkpoint failed" + ckpt_why);
        if (trace && to == message_at)
            std::fprintf(stderr, "strata serve: message boundary checkpoint: %lld tokens, %lld tail\n",
                         (long long) message_at, (long long) (turn_at - message_at));
    }
    if (!refill(err)) return die("refilling a lent slot failed: " + err);
    if (o.prefill_head_alias) {
        sp->reset();
        const auto head_t0 = Clock::now();
        if (!native_head.load(o.native_head_shards, g.n_embd, n_vocab, err))
            return die("restoring the native output head failed: " + err);
        // the verify windows' captured graphs read the head at the address of their capture
        if (prefill_head_alias_at == nullptr) prefill_head_alias_at = native_head.weights();
        if (native_head.weights() != prefill_head_alias_at)
            return die("--prefill-head-alias: the output head was reloaded at another address, so the captured "
                       "verify graphs would read freed memory");
        std::fprintf(stderr, "strata serve: --prefill-head-alias: output head restored in %.1f ms before decode\n",
                     std::chrono::duration<double, std::milli>(Clock::now() - head_t0).count());
    }
    tr("prompt done (slots refilled)");
    prompt_ms = std::chrono::duration<double, std::milli>(Clock::now() - r0).count();
    // the verify windows: the first holds the last prompt token alone
    p = n - 1;
    x = (int32_t) ids[(size_t) p];
    // (the drafter writes max_t - 1 drafts)
    const size_t DS = (size_t) std::max(S, use_mtp ? mtp.max_t() : 0);
    const size_t WS = (size_t) S;
    drafts.assign(DS, 0);
    window.assign(WS, 0);
    outv.assign(WS, 0);
    dprob.assign(DS, 0.0f);
    sfx_out.assign((size_t) S, 0);
    sfx_on = o.suffix_draft > 0 || o.lookup_chain > 0;
    if (sfx_on) {
        sfx->reset();
        for (int64_t t : ids) sfx->append((int32_t) t);
    }
    if (o.lookup_chain > 0) {
        extra_sources_reset();
        for (int64_t t : ids) { const int32_t t32 = (int32_t) t; extra_sources_append(&t32, 1); }
    }
    first_window = true;
    produced_n = sfx_windows = sfx_drafts = sfx_ok = 0;
    chain_windows = chain_drafts = chain_ok = 0;
    for (int64_t* counts : {t2_rej, t2_hit, dp_tested, dp_ok}) std::fill(counts, counts + 8, 0);
    cbuf.assign((size_t) S, 0);
    ctail.clear();
    lookup_src.emplace(*sfx);
    draft_offered = draft_accepted = 0;
    consumed.clear();
    consumed.reserve((size_t) (session_ctx + S));
    for (int64_t i = 0; i < p; ++i) consumed.push_back((int32_t) ids[(size_t) i]);
    d0 = Clock::now();
    ds0 = dec_snap();
    dt_run = dt_commit = dt_draft = 0;
    dt_adapt_wait = dt_adapt_join = 0;
    dt_sw0 = ajob ? a_swapped : adapt_nswaps;
    dec_windows = dec_T = 0;
    decode_hits0 = drive.d.cache_hits;
    ram0 = src.ram_reads();
    files0 = src.file_reads();
    file_bytes0 = src.file_read_bytes();
    decode_look0 = drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused;
    offload0 = drive.d.offload_entries;
    pcie_auto.mark(drive.d.ms_run, drive.d.multi_misses);   // the prompt's windows are not the decode's
    sp->embd_rows = nullptr;   // the caller's rows live for this call only
    sampling_applied = false;
    drafts_ok = true;
    decode_ok = !cancelled;
    if (!cancelled) return kOk;
    session_error = nf_hit ? "non-finite logits (#879)" : "cancelled";
    return nf_hit ? kInvalidated : kInterrupted;
}

int EngineState::decode_window(int32_t token, const Sampling& sampling, int budget, int32_t* out, int cap) {
    if (dead) return kDead;
    if (!decode_ok) return refuse("decode needs a successful sync first");
    if (budget < 1 || cap < 1) return refuse("decode needs a budget and a cap of at least 1");
    if (token < kNoToken || token >= n_vocab) return refuse("a token id is outside the vocabulary");
    if (first_window != (token == kNoToken))
        return refuse(first_window ? "the first decode after a sync reads the last prompt token: it takes no token"
                                   : "decode needs the token to feed");
    if (p + 2 > session_ctx) return refuse("the context has no room for the token and the one after it");
    if (token != kNoToken) {
        if (token != x) {   // not the engine's own sample: its drafts do not follow it
            x = token;
            drafts_ok = false;
        }
        if (sfx_on) sfx->append(x);
        if (o.lookup_chain > 0) extra_sources_append(&x, 1);
    }
    if (!sampling_applied || !(sampling == applied_sampling)) {
        // per-request sampling for the verify window's head (greedy when temperature is absent)
        strata::kernels::SamplerParams req_sp;
        req_sp.greedy = sampling.temperature <= 0.0f;
        req_sp.temperature = sampling.temperature;
        req_sp.top_p = sampling.top_p;
        req_sp.top_k = sampling.top_k;
        req_sp.seed = sampling.seed ? sampling.seed
                                    : (unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count();
        req_sp.min_p = std::clamp(sampling.min_p, 0.0f, 1.0f);
        req_sp.penalty_last_n = std::max(sampling.penalty_last_n, 0);
        req_sp.penalty_repeat = sampling.penalty_repeat;
        req_sp.penalty_freq = sampling.penalty_freq;
        req_sp.penalty_present = sampling.penalty_present;
        req_sp.counter = 0;
        ver.set_sampling(req_sp);
        if (use_mtp) mtp.set_draft_sampling(req_sp);   // STRATA_SPEC_COUPLED=1: sampled drafts (a no-op otherwise)
        hist_n = std::min(req_sp.penalty_last_n, kPenaltyWindowCap);
        ver.set_history(hist_n > 0 ? d_hist : nullptr, hist_n);
        applied_sampling = sampling;
        sampling_applied = true;
    }
    static const bool chain_fixed = std::getenv("STRATA_LOOKUP_CHAIN_FIXED") != nullptr;   // no policy gate
    const int limit = std::min(budget, cap);
    // the fed token alone: the first window, a token that is not the last window's own sample, the last cells
    const bool fresh = first_window || !drafts_ok || p + S > session_ctx;
    int T = use_mtp ? S_mtp : 1;   // no --mtp: one token a round unless a lookup draft fires
    if (use_mtp && req_spec_min_p > 0.0) {
        T = 1;
        while (T < S_mtp && dprob[(size_t) T - 1] >= (float) req_spec_min_p) ++T;
    }
    if (fresh) T = 1;
    // a repeat of earlier context (prompt lookup) where the MTP's own first guess agrees: the policy takes it
    // when its expected tokens per ms, from the measured acceptance and window costs, beat the MTP window's
    bool from_sfx = false;
    int sfx_match = 0;
    if (o.suffix_draft > 0 && !fresh) {
        const int k = sfx->propose(S - 1, sfx_out.data());
        sfx_match = sfx->last_match();
        if (k > 0 && (!use_mtp || sfx_out[0] == drafts[0])) {
            const strata::spec::DraftPolicy::Pick pk = policy->choose(T, k, sfx_match);
            if (pk.lookup) { T = pk.t; from_sfx = true; }
        }
    }
    // --lookup-chain: what followed an earlier occurrence of the context + the MTP's drafts, after them
    int chain_n = 0, cm = 0;
    if (o.lookup_chain > 0 && !fresh && !from_sfx && T < S) {
        int csrc = -1;
        const int nt = chain_tail(*sfx, drafts.data(), T - 1, ctail);
        chain_n = strata::spec::propose_from_sources(*lookup_src, ctail.data(), nt, T - 1,
                                                     std::min(o.lookup_chain, S - T), o.lookup_chain_min,
                                                     cbuf.data(), &cm, &csrc);
        // the policy keeps the chained tokens that pay at this machine's measured window costs
        if (chain_n > 0 && !chain_fixed) {
            double p_mtp = 1.0;
            for (int i = 0; i < T - 1; ++i) p_mtp *= dprob[(size_t) i];
            chain_n = policy->chain(T, p_mtp, chain_n, cm);
        }
    }
    const int T_mtp = T;
    T += chain_n;
    const bool timed_round = !fresh;
    const Clock::time_point round0 = Clock::now();
    window[0] = x;
    for (int i = 1; i < T_mtp; ++i)
        window[(size_t) i] = from_sfx ? sfx_out[(size_t) i - 1] : drafts[(size_t) i - 1];
    for (int i = 0; i < chain_n; ++i) window[(size_t) (T_mtp + i)] = cbuf[(size_t) i];
    drive.d.layers = 0;
    drive.d.experts = 0;
    drive.d.failed = false;
    // #463: the previous adapt round's copies land first - with a non-blocking query, whether a swapped-in
    // expert ran on the GPU or the CPU (they round differently) depended on the copy's timing
    // (STRATA_ADAPT_NOWAIT=1: 0.1.37's non-blocking query, the A/B)
    const Clock::time_point taw0 = Clock::now();
    if (adapt_nowait()) apply_pending(false);
    else if (aov && !pending.empty() && !aflipped) {
        // 0.1.43 STRATA_ADAPT_OVERLAP: the tables take the swaps now, exactly where the wait would have ended;
        // the copies still landing are waited for per layer inside the window (AdaptGate)
        // (resident RAM mode: the staged evicted experts look committed to the planner at once, the RAM
        // copy's places stay as they are until the copies have landed; see soft_commit_exchanges)
        if (src.complement_ready() && !src.soft_commit_exchanges()) {
            apply_pending(true);   // not possible here: the plain wait
        } else {
            for (const auto& [i, slot] : pending) host_res[(size_t) i] = slot;
            res_upload();
            aflipped = true;
            ++aov_flips;
            agate.active.store(true, std::memory_order_release);
        }
    }
    else if (pending.empty() || ++pending_age >= adapt_lag()) apply_pending(true);
    // STRATA_RAM_ADAPT=<n> (opt-in, the resident RAM mode on one GPU): the RAM copy is chosen once, at
    // start, so an expert it does not hold is read from the file every time the conversation routes to
    // it.  With this, before a window and while no adaptive round is in flight, up to n experts the windows
    // keep routing to (the adaptive tier's decayed count, at least its own candidate floor of 2) take the
    // RAM places of experts of their layer that are not routed to (count below 0.5), one pair per layer,
    // the largest difference first.  One blob read each, on this thread; the RAM stays the same size, and
    // the routing, the weights and the VRAM tier's own rule are untouched.
    static const int ram_adapt_n = [] {
        const char* e = std::getenv("STRATA_RAM_ADAPT");
        return e != nullptr ? std::max(0, std::atoi(e)) : 0;
    }();
    if (ram_adapt_n > 0 && !first_window && !drive.d.usage.empty() && pending.empty() &&
        (!ajob || astate == AState::Idle) && src.complement_ready()) {
        constexpr float kRamAdaptRouted = 2.0f, kRamAdaptUnused = 0.5f;
        struct RamSwap { float gain; int32_t layer, in, out; };
        std::vector<RamSwap> swaps;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const float* u = drive.d.usage.data() + l * g.n_expert;
            const int32_t* r = host_res.data() + l * g.n_expert;
            int32_t in = -1, out = -1;
            for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                if (r[e] >= 0) continue;   // in VRAM: the RAM copy does not hold it
                if (src.has_resident(l, e)) { if (out < 0 || u[e] < u[out]) out = e; }
                else if (u[e] >= kRamAdaptRouted && (in < 0 || u[e] > u[in])) in = e;
            }
            if (in >= 0 && out >= 0 && u[out] < kRamAdaptUnused)
                swaps.push_back({u[in] - u[out], (int32_t) l, in, out});
        }
        std::sort(swaps.begin(), swaps.end(),
                  [](const RamSwap& a, const RamSwap& b) { return a.gain > b.gain; });
        if ((int) swaps.size() > ram_adapt_n) swaps.resize((size_t) ram_adapt_n);
        for (const RamSwap& x : swaps) src.replace_resident(x.layer, x.in, x.out);
    }
    if (!adapt_tick(false)) return die("an adaptive refill failed");   // --adapt-async: the round in flight moves on a step when it can
    dt_adapt_wait += std::chrono::duration<double, std::milli>(Clock::now() - taw0).count();
    if (hist_n > 0) {
        // the tails the penalties count over, ONE PER ROW: the tokens the state has consumed, the
        // fed-back head `x` (it joins `consumed` only after this window commits), then the drafts
        // before that row - what plain decode would have counted there.  (Until 0.1.19 only row 0
        // was staged, and the drafted rows read unwritten slots.)
        strata::kernels::penalty_rows(consumed.data(), (int64_t) consumed.size(), window.data(), T,
                                      hist_n, hist_stage.data());
        cudaMemcpy(d_hist, hist_stage.data(), (size_t) T * (size_t) hist_n * sizeof(int32_t),
                   cudaMemcpyHostToDevice);
    }
    // the elastic K/V: the window's cells and the drafter's beyond them
    if (kvg.on && p + T + 64 > kvg.cells &&
        !kvg_ensure(p + T + 64, adapt_quiesce))
        return die("the K/V cannot grow: no VRAM is left");
    tr("window", p, T);
    const Clock::time_point tw0 = Clock::now();
    // STRATA_SPEC_PROB: the MTP drafts' distributions q, judged by rejection sampling (core/spec_prob.hpp);
    // a suffix window and the lookup chain's tail are point masses and keep the exact-match rule
    // (never on a first window)
    if (use_mtp && mtp.prob() && !from_sfx && !fresh && T_mtp > 1) ver.set_spec_q(mtp.spec_q(), T_mtp - 1);
    if (!ver.run(T, window.data(), p, win_pool_fn, win_pool_user, outv.data(), err) || drive.d.failed)
        return die(drive.d.failed && drive.d.fail ? drive.d.fail : err);
    if (aflipped) apply_pending(true);   // 0.1.43: every layer's copies were waited for in the window; the bookkeeping now
    // Diagnostics: STRATA_DUMP_FIRST_LOGITS=<path> writes the logits of each request's first
    // window to <path>.<n>, so the prompt paths can be compared by KL
    if (first_window) {
        static const char* fl = std::getenv("STRATA_DUMP_FIRST_LOGITS");
        static int fl_n = 0;
        if (fl != nullptr) {
            std::vector<float> row((size_t) ver.vocab());
            const std::string path = std::string(fl) + "." + std::to_string(fl_n++);
            std::FILE* f = ver.copy_logits(0, row.data()) ? std::fopen(path.c_str(), "wb") : nullptr;
            if (f == nullptr || std::fwrite(row.data(), sizeof(float), row.size(), f) != row.size())
                std::fprintf(stderr, "strata serve: STRATA_DUMP_FIRST_LOGITS: cannot write %s\n", path.c_str());
            if (f) std::fclose(f);
        }
    }
    int a = 0;
    while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
    if (from_sfx) { ++sfx_windows; sfx_drafts += T - 1; sfx_ok += a; }
    if (chain_n > 0) { ++chain_windows; chain_drafts += chain_n; chain_ok += std::max(0, a - (T_mtp - 1)); }
    if (!from_sfx && !fresh)
        for (int d = 0; d < T_mtp - 1 && d < 8 && d <= a; ++d) { ++dp_tested[d]; dp_ok[d] += d < a; }
    if (strata::core::MtpDrafter::top2_env() && !from_sfx && !fresh && a < T_mtp - 1 && a < 8) {
        ++t2_rej[a];
        if (mtp.top2(a) == outv[(size_t) a]) ++t2_hit[a];
    }
    const Clock::time_point tw1 = Clock::now();
    std::thread adapt_thr;   // the adaptive tier beside the commit and the draft
    bool adapt_ok = true;
    // (--adapt-async 1: the asynchronous tier above instead, ticked before the window)
    if (!drive.d.usage.empty() && !ajob && ((rounds + 1) % o.adapt_every) == 0)
        adapt_thr = std::thread([&] { strata::aux_cpus::pin_current_thread(); adapt_ok = adapt(); });
    // ruling 11: the outputs this call returns - the accepted a + 1, cut at `limit` and after the first
    // end-of-generation token - and only their rows committed, so the state is exactly the returned history
    int keep = 0;
    bool eos = false;
    while (keep <= a && keep < limit && !eos) {
        eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) keep]) != o.eos_ids.end();
        ++keep;
    }
    if (!ver.commit(keep, err)) {
        if (adapt_thr.joinable()) adapt_thr.join();
        return die(err);
    }
    if (nonfinite_hit(ver, first_window ? "first decode" : "decode")) {   // #879: none of its tokens goes out
        if (adapt_thr.joinable()) adapt_thr.join();
        if (!adapt_ok) return die("an adaptive refill failed");
        decode_ok = false;
        session_error = "non-finite logits (#879)";
        return kInvalidated;
    }
    // the window's first `keep` tokens are in the session now (the last output is not: it is next x)
    for (int i = 0; i < keep; ++i) consumed.push_back(window[(size_t) i]);
    draft_offered += T - 1;
    draft_accepted += a;
    first_window = false;
    for (int i = 0; i < keep; ++i) {
        out[i] = outv[(size_t) i];
        strata::core::progress_beat();
        ++produced_n;
        if (i + 1 == keep) break;   // the drafters take the last output once it is fed (the next call's token)
        if (sfx_on) sfx->append(outv[(size_t) i]);
        if (o.lookup_chain > 0) extra_sources_append(&outv[(size_t) i], 1);
    }
    ++rounds;
    if (drive.d.fs) drive.d.fs->completed.fetch_add(1, std::memory_order_release);
    const Clock::time_point tw2 = Clock::now();
    // coupled drafts with penalties: the next window's row-0 history (`consumed` holds this window's
    // commit, outv[keep - 1] is its row 0) - the drafts extend it on the device as the verify rows will
    if (use_mtp && hist_n > 0 && mtp.coupled() && !eos && keep < budget)
        mtp.set_draft_history(consumed.data(), (int64_t) consumed.size(), outv[(size_t) keep - 1]);
    const bool drafted = !use_mtp || eos || keep >= budget ||
                         mtp.draft(T, outv.data(), p, keep - 1, drafts.data(), err, dprob.data(),
                                   (float) req_spec_min_p);
    {
        const Clock::time_point tw3 = Clock::now();
        auto msd = [](Clock::time_point a0, Clock::time_point b0) { return std::chrono::duration<double, std::milli>(b0 - a0).count(); };
        dt_run += msd(tw0, tw1); dt_commit += msd(tw1, tw2); dt_draft += msd(tw2, tw3);
        ++dec_windows; dec_T += T;
    }
    pcie_auto_step(req_pcie_frac);   // the default share, refined from the CPU pool's speed (PcieAuto); a request's own share is left alone
    if (adapt_thr.joinable()) {
        const Clock::time_point taj0 = Clock::now();
        adapt_thr.join();
        dt_adapt_join += std::chrono::duration<double, std::milli>(Clock::now() - taj0).count();
    }
    if (!adapt_ok) return die("an adaptive refill failed");
    if (!drafted) return die(err);
    if (timed_round && !eos && chain_n == 0)
        policy->observe(from_sfx, T, a, sfx_match,
                        std::chrono::duration<double, std::milli>(Clock::now() - round0).count());
    else if (timed_round && !eos)   // a chained window: its MTP part, its chain, its cost
        policy->observe_chain(T_mtp, chain_n, a, cm,
                              std::chrono::duration<double, std::milli>(Clock::now() - round0).count());
    x = outv[(size_t) keep - 1];
    p += keep;
    drafts_ok = !eos && keep < budget;
    return keep;
}

int EngineState::finish_request() {
    if (!request_open) return dead ? kDead : kOk;
    request_open = false;
    decode_ok = false;
    if (dead) return kDead;
    const int64_t n = (int64_t) cur.size();
    const double decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - d0).count();
    // the last commit (set_commit_async): the session is complete before anything reads or copies it
    if (!ver.wait_commit(err)) return die(err);
    // STRATA_DECODE_TIMING=1: where a request's decode time goes (one line per request)
    static const bool dec_timing = std::getenv("STRATA_DECODE_TIMING") != nullptr;
    if (dec_timing && dec_windows > 0) {
        const DecSnap d1 = dec_snap();
        const double w = (double) dec_windows, L = (double) g.n_layers;
        std::fprintf(stderr, "strata decode timing: %lld windows, avg T %.2f, %.2f tokens/window, %.2f ms/window = "
                             "verify %.2f (GPU-reach wait %.2f + per-layer host %.2f [plan %.2f actq %.2f jobs %.2f "
                             "CPU %.2f] + stage %.2f) + commit/emit %.2f + draft %.2f; per layer-window: CPU experts "
                             "%.2f (%.2f entries), VRAM hits %.2f, PCIe %.2f; adapt wait %.2f join %.2f, %.1f swaps/window\n",
                     (long long) dec_windows, dec_T / w, produced_n / w, decode_ms / w, dt_run / w,
                     (d1.wait - ds0.wait) / w, (d1.pool - ds0.pool) / w, (d1.plan - ds0.plan) / w,
                     (d1.actq - ds0.actq) / w, (d1.jobs - ds0.jobs) / w, (d1.run - ds0.run) / w,
                     (d1.host - ds0.host) / w, dt_commit / w, dt_draft / w, (d1.misses - ds0.misses) / (w * L),
                     (d1.entries - ds0.entries) / (w * L), (d1.hits - ds0.hits) / (w * L), (d1.pcie - ds0.pcie) / (w * L),
                     dt_adapt_wait / w, dt_adapt_join / w, (double) ((ajob ? a_swapped : adapt_nswaps) - dt_sw0) / w);
        if (aov)
            std::fprintf(stderr, "strata decode adapt overlap: %lld windows took swaps still landing, a layer waited %.2f ms in all so far\n",
                         (long long) aov_flips, ver.ms_gate);
        if (const std::string pr = ver.profile_report(); !pr.empty())
            std::fprintf(stderr, "strata decode GPU stages (ms/window):%s\n", pr.c_str());
    }
    if (drive.d.fs) std::fprintf(stderr, "strata serve: %s\n", drive.d.fs->report().c_str());
    if (!cancelled) {
        // a prompt stopped halfway leaves the session somewhere between two chunks: nothing to continue from
        // (the checkpoints taken while reading it are still good)
        live.swap(consumed);
        live_imgs = imgs_below(req_imgs, (int64_t) live.size());
        live_ok = o.prompt_cache > 0 && req_ckpt;   // ckpt=0: nothing to continue (#830)
        if (nf_hit) live_ok = false;   // #879 (the next request's start drops the rest)
    } else if (!nf_hit && pp_exact && pp_reached > 0 && o.prompt_cache > 0 && req_ckpt &&
               pp_reached <= (int64_t) cur.size()) {
        // #1620: stopped between two windows (or before the first token was read): the session is exactly at
        // pp_reached, so it stays the live conversation - the next request continues from it.
        live.assign(cur.begin(), cur.begin() + pp_reached);
        live_imgs = imgs_below(req_imgs, pp_reached);
        live_ok = true;
        std::fprintf(stderr, "strata serve: cancelled request keeps its conversation: %lld tokens live\n",
                     (long long) pp_reached);
    }
    static const bool state_hash = std::getenv("STRATA_STATE_HASH") != nullptr;
    if (state_hash && !cancelled && o.prompt_cache > 0) {   // every finished request, ckpt=0 too (parity gates)
        // DEBUG: a fingerprint of every part of the session over the positions it holds ([0, L)), and
        // separately of what lies past them in the last KV page (stale cells, fine unless something reads them)
        if (cudaDeviceSynchronize() != cudaSuccess) return die("synchronizing state fingerprint");
        const int64_t L = (int64_t) live.size();
        const strata::kernels::QsaShapes qs = strata::core::qsa_shapes(g);
        bool hash_ok = true;
        std::array<uint8_t, 65536> hash_buffer;
        auto hash_dev = [&](const void* p, size_t bytes, uint64_t h) {
            for (size_t offset = 0; hash_ok && offset < bytes;) {
                const size_t n = std::min(hash_buffer.size(), bytes - offset);
                // VRAM or a streamed host copy, with fixed diagnostic workspace.
                if (cudaMemcpy(hash_buffer.data(), static_cast<const uint8_t*>(p) + offset, n, cudaMemcpyDefault) != cudaSuccess) {
                    hash_ok = false;
                    break;
                }
                h = fnv1a(hash_buffer.data(), n, h);
                offset += n;
            }
            return h;
        };
        // the cells [c0, c1) of one int8 K or V pool ([page][kv_head][page_size][head_dim]), bytes per value `w`
        auto hash_cells = [&](const void* pool, int64_t per_cell, int64_t c0, int64_t c1, uint64_t h) {
            const int64_t ps = qs.page_size;
            for (int64_t pg = c0 / ps; pg * ps < c1; ++pg)
                for (int64_t hd = 0; hd < qs.n_head_kv; ++hd) {
                    const int64_t a = std::max(c0, pg * ps) - pg * ps, e = std::min(c1, (pg + 1) * ps) - pg * ps;
                    const size_t off = (size_t) (((pg * qs.n_head_kv + hd) * ps + a) * per_cell);
                    h = hash_dev((const uint8_t*) pool + off, (size_t) ((e - a) * per_cell), h);
                }
            return h;
        };
        const ConvStateSizes z = conv_state_sizes(g, ss);
        uint64_t h_gdn = hash_dev(ss.gdn_state, z.gdn, 1469598103934665603ull);
        if (std::getenv("STRATA_STATE_HASH_GDN") != nullptr && ss.gdn_alloc > 0) {   // per GDN layer: which one differs first
            const size_t per = z.gdn / (size_t) ss.gdn_alloc;
            std::string s;
            char b[8];
            for (int64_t i = 0; i < ss.gdn_alloc; ++i) {
                std::snprintf(b, sizeof(b), "%04llx ", (unsigned long long) (hash_dev((const uint8_t*) ss.gdn_state + i * per, per, 1469598103934665603ull) & 0xffff));
                s += b;
            }
            std::fprintf(stderr, "strata serve: STATE_HASH_GDN %s\n", s.c_str());
        }
        uint64_t h_ple = hash_dev(ss.ple_hist, ss.ple_hist ? z.ple : 0, 1469598103934665603ull);
        uint64_t h_tail = 1469598103934665603ull, h_pool = h_tail, h_kv = h_tail, h_stale = h_tail;
        // pooled= keeps its 0.1.29 meaning: the completed rows [0, L / idx_block) only.  pooled_full= adds the
        // spare row at L / idx_block (the `dead` key the next block completion overwrites), which a
        // conversation restore writes back; dead= is the spare key itself
        uint64_t h_dead = h_tail, h_pool_full = h_tail;
        const int64_t kvb = qs.head_dim, scb = (qs.head_dim / 64) * 2;
        // a state's K/V arrays and their bytes per (cell, head) row: the host copy when it has one
        auto kv_arrays = [&](const strata::core::QsaState& st) {
            const bool h = st.kv_mode != 0;
            std::vector<std::pair<const void*, int64_t>> a;
            if (st.kv_q4) {
                const int64_t q4b = (int64_t) strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                a = {{h ? st.host.k_q4 : st.k_q4, q4b}, {h ? st.host.v_q4 : st.v_q4, q4b}};
            } else if (st.kv_hybrid) {
                const int64_t q4b = (int64_t) strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                a = {{h ? st.host.k_q : st.k_q, kvb}, {h ? st.host.v_q4 : st.v_q4, q4b},
                     {h ? st.host.k_scale : st.k_scale, scb}};
            } else if (st.kv_int8) {
                a = {{h ? st.host.k_q : st.k_q, kvb}, {h ? st.host.v_q : st.v_q, kvb},
                     {h ? st.host.k_scale : st.k_scale, scb}, {h ? st.host.v_scale : st.v_scale, scb}};
            } else {
                a = {{h ? st.host.k_pool : st.k_pool, qs.head_dim * 2},
                     {h ? st.host.v_pool : st.v_pool, qs.head_dim * 2}};
            }
            return a;
        };
        const int64_t end_cell = std::min<int64_t>(((L + qs.page_size - 1) / qs.page_size) * qs.page_size,
                                                   ss.max_cells);   // = the primary state's max_cells
        for (int64_t j = 0; j < ss.qsa_alloc; ++j) {   // this session's owned QSA ordinals only
            const strata::core::QsaState& st = ss.qsa_states[ss.qsa_ord0 + j];
            h_tail = hash_dev(st.idx_tail, z.tail, h_tail);
            h_dead = hash_dev(st.idx_dead, z.dead, h_dead);
            h_pool = hash_dev(st.idx_pooled, (size_t) (L / qs.idx_block) * qs.idx_dim * 4, h_pool);
            h_pool_full = hash_dev(st.idx_pooled, (size_t) (L > 0 ? L / qs.idx_block + 1 : 0) * qs.idx_dim * 4,
                                   h_pool_full);
            // KV streaming: the host copy is the identity layout and holds every cell
            for (const auto& [pool, w] : kv_arrays(st)) {
                h_kv = hash_cells(pool, w, 0, L, h_kv);
                h_stale = hash_cells(pool, w, L, end_cell, h_stale);
            }
        }
        uint64_t h_mtp = 1469598103934665603ull;
        if (use_mtp) {
            const strata::core::QsaState& ms = mtp.kv_state();
            const int64_t mL = std::min<int64_t>(L, ms.max_cells);
            for (const auto& [pool, w] : kv_arrays(ms))
                if (pool != nullptr) h_mtp = hash_cells(pool, w, 0, mL, h_mtp);
        }
        if (!hash_ok) return die("reading state fingerprint");
        std::fprintf(stderr, "strata serve: STATE_HASH L=%lld gdn=%016llx ple=%016llx tail=%016llx pooled=%016llx "
                             "kv=%016llx mtp=%016llx stale=%016llx dead=%016llx pooled_full=%016llx ple_prev=%d,%d\n", (long long) L,
                     (unsigned long long) h_gdn, (unsigned long long) h_ple, (unsigned long long) h_tail,
                     (unsigned long long) h_pool, (unsigned long long) h_kv, (unsigned long long) h_mtp,
                     (unsigned long long) h_stale, (unsigned long long) h_dead,
                     (unsigned long long) h_pool_full, ss.ple_prev[0], ss.ple_prev[1]);
    }
    const int64_t req_hits = drive.d.cache_hits - decode_hits0;
    const int64_t req_look = (drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused) - decode_look0;
    const int64_t req_offload = drive.d.offload_entries - offload0;
    // #471: the prompt tokens this request read - all the fresh ones, or as far as the prompt pass got when a
    // cancel stopped it part-way (a cancelled request used to be logged and counted as having read them all)
    const int64_t fresh = n - resume;
    const int64_t read_n = cancelled ? std::clamp<int64_t>(pp_reached - resume, 0, fresh) : fresh;
    stats.generated = produced_n;
    stats.prompt_tokens = n;
    stats.prompt_ms = prompt_ms;
    stats.decode_ms = decode_ms;
    stats.draft_accepted = draft_accepted;
    stats.draft_offered = draft_offered;
    stats.reused = resume;
    stats.expert_hits = req_hits;
    stats.expert_lookups = req_look;
    stats.ram_reads = src.ram_reads() - ram0;
    stats.file_reads = src.file_reads() - files0;
    stats.file_read_mb = (double) (src.file_read_bytes() - file_bytes0) / 1e6;
    stats.prompt_read = read_n;
    stats.offloaded = req_offload;
    stats.chain_counts = o.lookup_chain > 0;
    stats.chain_accepted = chain_ok;
    stats.chain_offered = chain_drafts;
    stats.suffix_accepted = sfx_ok;
    stats.suffix_offered = sfx_drafts;
    stats.windows = dec_windows;
    stats.chunk = req_chunk;
    stats.experts_streamed = sp->chain_experts_streamed() - streamed0;
    // "12288 of 98179" when cancelled mid-read (#471), the rate from what was read
    char read_txt[64];
    if (cancelled)
        std::snprintf(read_txt, sizeof(read_txt), "%lld of %lld", (long long) read_n, (long long) fresh);
    else
        std::snprintf(read_txt, sizeof(read_txt), "%lld", (long long) fresh);
    std::fprintf(stderr, "strata serve: prompt %lld tokens = %lld reused + %s read in %.0f ms (%.1f tok/s), "
                         "%lld generated in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld, %zu checkpoints%s\n",
                 (long long) n, (long long) resume, read_txt, prompt_ms,
                 prompt_ms > 0 ? 1000.0 * read_n / prompt_ms : 0.0, (long long) produced_n, decode_ms,
                 decode_ms > 0 ? 1000.0 * produced_n / decode_ms : 0.0, (long long) draft_accepted,
                 (long long) draft_offered, checks.size(), cancelled ? " (cancelled)" : "");
    // the VRAM share of the experts the pool looked up while decoding; experts it sent over PCIe for the GPU
    // to read (--pcie-frac) are in neither count - #588: so that share is said beside it (raising --pcie-frac
    // raises the hit rate while the PCIe reads may make the decode slower)
    if (req_look > 0) {
        char off[160] = "";
        if (req_offload > 0)
            std::snprintf(off, sizeof off, "; %lld more read by the GPU over PCIe or from another GPU (%.1f%% of "
                          "all %lld routed)", (long long) req_offload,
                          100.0 * (double) req_offload / (double) (req_look + req_offload),
                          (long long) (req_look + req_offload));
        std::fprintf(stderr, "strata serve: decode expert cache hit rate: %.1f%% (%lld hits / %lld lookups)%s\n",
                     100.0 * (double) req_hits / (double) req_look,
                     (long long) req_hits, (long long) req_look, off);
    }
    if (drive.d.tail_experts > 0)   // STRATA_ROUTE_TAIL_SKIP, cumulative
        std::fprintf(stderr, "strata serve: route tail skip: %lld missed experts skipped (%lld entries) since the "
                             "start\n", (long long) drive.d.tail_experts, (long long) drive.d.tail_skipped);
    // the resident RAM mode, cumulative: experts read from experts.bin since the copy was made (what the plain
    // mmap mode reads through the OS file cache, from the SSD when the RAM could not keep it)
    if (src.complement_ready())
        std::fprintf(stderr, "strata serve: resident RAM: %.2f GiB of experts in RAM, %lld exchanged with the "
                             "VRAM tier, %lld blob reads from the file\n",
                     (double) src.resident_bytes() / 1073741824.0, (long long) src.exchanges(),
                     (long long) src.file_reads());
    if (src.ram_adapts() > 0)   // STRATA_RAM_ADAPT, cumulative
        std::fprintf(stderr, "strata serve: resident RAM: %lld experts moved in from the file in place of "
                             "unused ones (STRATA_RAM_ADAPT)\n", (long long) src.ram_adapts());
    if (ajob)   // --adapt-async, cumulative (a round may still be in flight: it lands before the next request)
        std::fprintf(stderr, "strata serve: asynchronous adaptive tier: %lld rounds, %lld experts swapped in, "
                             "%.1f ms per round (start to flip, between windows)",
                     (long long) a_rounds, (long long) a_swapped, a_rounds > 0 ? a_ms / (double) a_rounds : 0.0);
    if (ajob) {
        if (a_arena)
            std::fprintf(stderr, " (arena, copy-in only, batch %zu, wait %d, pick %s %.2f ms/round)\n", a_batch,
                         (int) a_wait, a_inline ? "inline" : "on the job", a_rounds > 0 ? a_pick_ms / (double) a_rounds : 0.0);
        else
            std::fprintf(stderr, "\n");
    }
    if (src.exchange_rotation())
        std::fprintf(stderr, "strata serve: exchange rotation: %llu blocks, %llu host memcpy bytes avoided (cumulative payload)\n",
                     (unsigned long long)src.rotated_exchanges(),
                     (unsigned long long)src.avoided_exchange_copy_bytes());
    // CS-T: the tiers, cumulative - GPU cache hits (the decode lookups above), RAM copy, files (SSD / OS cache)
    if (srcp == &src) {
        std::fprintf(stderr, "strata serve: expert tiers: GPU %lld hits this request; since the start RAM %lld blobs, files %lld blobs "
                             "%.1f MB read%s\n", (long long) req_hits, (long long) src.ram_reads(),
                     (long long) src.file_reads(), (double) src.file_read_bytes() / 1e6,
                     src.gguf_mode() ? " (the GGUF in place)" : "");
        // what the OS did for this request (since the previous one): the drive's share of the file tier's reads
        // against the page cache's, and what the I/O path did about it
        static strata::platform::ProcIo io_prev;
        static strata::core::FileExpertSource::IoCounters ic_prev;
        static uint64_t fb_prev = 0;
        const strata::platform::ProcIo io_now = strata::platform::proc_io_sample();
        const strata::core::FileExpertSource::IoCounters ic = src.io_counters();
        const uint64_t fb = src.file_read_bytes();
        if (io_now.valid) {
            std::fprintf(stderr, "strata serve: file tier I/O this request: the OS read %.1f MB from storage (%llu major faults) for "
                                 "%.1f MB of expert reads", (double) (io_now.read_bytes - io_prev.read_bytes) / 1e6,
                         (unsigned long long) (io_now.major_faults - io_prev.major_faults), (double) (fb - fb_prev) / 1e6);
            if (src.io_stats())
                std::fprintf(stderr, "; page cache held %.1f MB at the read, not %.1f MB",
                             (double) (ic.cached_bytes - ic_prev.cached_bytes) / 1e6,
                             (double) (ic.uncached_bytes - ic_prev.uncached_bytes) / 1e6);
            if (src.io_prefetch())
                std::fprintf(stderr, "; io prefetch: %llu read ahead (%.1f MB by pread), %llu used, %llu unused, %llu skipped "
                                     "(cached), %llu dropped; decode waited %.1f ms in %llu fetches",
                             (unsigned long long) (ic.pf_read_blobs - ic_prev.pf_read_blobs),
                             (double) (ic.pread_bytes - ic_prev.pread_bytes) / 1e6,
                             (unsigned long long) (ic.pf_used - ic_prev.pf_used),
                             (unsigned long long) (ic.pf_unused - ic_prev.pf_unused),
                             (unsigned long long) (ic.pf_resident_skips - ic_prev.pf_resident_skips),
                             (unsigned long long) (ic.pf_dropped - ic_prev.pf_dropped),
                             (double) (ic.crit_us - ic_prev.crit_us) / 1e3,
                             (unsigned long long) (ic.crit_n - ic_prev.crit_n));
            std::fprintf(stderr, "\n");
            io_prev = io_now;
            ic_prev = ic;
            fb_prev = fb;
        }
    }
    // STRATA_SPLIT_TIMING: where the verifier's host time went, cumulative per window since the start
    // (waiting for its GPU to ring a layer, the CPU pool and plan per layer, staging the window)
    if (static const bool st_timing = std::getenv("STRATA_SPLIT_TIMING") != nullptr; st_timing) {
        const double w = ver.windows > 0 ? (double) ver.windows : 1.0;
        std::fprintf(stderr, "strata serve: stage 0: %lld windows; per window: wait for the GPU %.3f ms, "
                             "pool + plan %.3f ms, host staging %.3f ms, commit %.3f ms\n",
                     (long long) ver.windows, ver.ms_wait / w, ver.ms_pool / w, ver.ms_host / w, ver.ms_commit / w);
    }
    if (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1) {
        // KV streaming, cumulative over the process: blocks the selections named vs blocks read from RAM
        uint64_t miss = 0, look = 0;
        bool over = false;
        for (int64_t j = 0; j < ss.qsa_alloc; ++j) {
            const strata::kernels::KvStreamCounters c =
                strata::kernels::kv_stream_counters(ss.qsa_states[ss.qsa_ord0 + j].map);
            miss += c.misses; look += c.lookups; over = over || c.overflow;
        }
        std::fprintf(stderr, "strata serve: KV streaming: %.2f%% of %llu block reads hit VRAM, %.1f MiB read "
                             "from RAM%s\n", look ? 100.0 * (double) (look - miss) / (double) look : 100.0,
                     (unsigned long long) look, (double) miss * 4224.0 / 1048576.0,
                     over ? " - OVERFLOW (too few resident cells)" : "");
    }
    if (sfx_windows > 0)
        std::fprintf(stderr, "strata serve: suffix drafts: %lld windows, %lld of %lld drafts accepted\n",
                     (long long) sfx_windows, (long long) sfx_ok, (long long) sfx_drafts);
    if (o.lookup_chain > 0)
        std::fprintf(stderr, "strata serve: lookup chain: %lld of %lld windows, %lld of %lld chained drafts accepted\n",
                     (long long) chain_windows, (long long) dec_windows, (long long) chain_ok, (long long) chain_drafts);
    if (std::getenv("STRATA_SPEC_DEPTH") != nullptr)
        std::fprintf(stderr, "strata serve: spec depth: accepted/tested %lld/%lld %lld/%lld %lld/%lld %lld/%lld %lld/%lld\n",
                     (long long) dp_ok[0], (long long) dp_tested[0], (long long) dp_ok[1], (long long) dp_tested[1],
                     (long long) dp_ok[2], (long long) dp_tested[2], (long long) dp_ok[3], (long long) dp_tested[3],
                     (long long) dp_ok[4], (long long) dp_tested[4]);
    if (strata::core::MtpDrafter::top2_env())
        std::fprintf(stderr, "strata serve: mtp top2: rejected/runner-up by depth %lld/%lld %lld/%lld %lld/%lld %lld/%lld "
                             "of %lld windows\n", (long long) t2_rej[0], (long long) t2_hit[0], (long long) t2_rej[1],
                     (long long) t2_hit[1], (long long) t2_rej[2], (long long) t2_hit[2], (long long) t2_rej[3],
                     (long long) t2_hit[3], (long long) dec_windows);
    busy_scope.reset();
    return kOk;
}

void EngineState::invalidate_session() {
    if (request_open) finish_request();
    live_ok = false;
    checks.clear();
    tail_ckpt_len = -1;
}

bool EngineState::shut_down() {
    if (dead) return false;
    if (!ready) return true;
    if (finish_request() != kOk) return false;
    if (!adapt_tick(true)) {
        die("an adaptive refill failed");
        return false;
    }
    save_profile("exit");   // #477: QUIT, or the server closed stdin
    return true;
}

void EngineState::drop_host_state() {
    checks.clear();
    checks.shrink_to_fit();
    ckpt_pool.reset();
    live = {};
    consumed = {};
}

namespace strata::program {

namespace {
// set by the first open and never reset: a re-open is not safe (most VRAM is never freed, process-wide library
// state and the client's threads survive an engine)
std::atomic<bool> g_engine_opened{false};
const std::string g_closed = "the engine is closed";
}

struct Engine::State {
    EngineState s;
};

Engine::Engine() : state_(std::make_unique<State>()) {}

Engine::~Engine() { close(); }

std::unique_ptr<Engine> Engine::open(int argc, char** argv, OpenError& error) {
    error = OpenError{};
    if (g_engine_opened.exchange(true)) {
        error.status = kExitFailure;
        error.message = "an engine was already opened in this process (one engine per process)";
        return nullptr;
    }
    std::unique_ptr<Engine> e(new Engine());
    EngineState& s = e->state_->s;
    int status = s.start_up(argc, argv);
    if (status == kExitOk && !s.nothing_to_serve) status = s.serve_setup();
    if (status == kExitOk && !s.nothing_to_serve) return e;
    error.status = status;
    error.message = s.error_message;
    return nullptr;
}

std::unique_ptr<Session> Engine::create_session(int64_t ctx, std::string& error) {
    const EngineState& s = state_->s;
    if (s.dead) {
        error = "the engine is dead: " + s.session_error;
        return nullptr;
    }
    if (session_ != nullptr) {
        error = "the engine already has its session (one per engine)";
        return nullptr;
    }
    if (ctx < 1 || ctx > s.o.max_context) {
        error = "a session holds 1 to " + std::to_string((long long) s.o.max_context) + " tokens";
        return nullptr;
    }
    std::unique_ptr<Session> session(new Session(*this, ctx));
    state_->s.session_ctx = ctx;
    session_ = session.get();
    return session;
}

int Engine::close() {
    if (!state_) return closed_dead_ ? kDead : kOk;
    if (session_ != nullptr) {
        session_->engine_ = nullptr;
        session_ = nullptr;
    }
    EngineState& s = state_->s;
    s.shut_down();
    closed_dead_ = s.dead;
    closed_message_ = s.session_error;
    if (s.dead) {
        // the destructors would call CUDA on a context that may be poisoned: the rest goes with the process
        s.drop_host_state();
        (void) state_.release();
    } else {
        state_.reset();
    }
    strata::core::set_native_embed(nullptr);
    return closed_dead_ ? kDead : kOk;
}

bool Engine::dead() const { return state_ ? state_->s.dead : closed_dead_; }
const std::string& Engine::dead_message() const { return state_ ? state_->s.session_error : closed_message_; }
const std::string& Engine::info() const { return state_->s.info; }

int64_t Engine::spec() const { return state_->s.o.spec; }
int64_t Engine::prefill_chunk() const { return state_->s.o.prefill_chunk; }
int64_t Engine::max_context() const { return state_->s.o.max_context; }
int64_t Engine::vocab_size() const { return state_->s.n_vocab; }
int64_t Engine::embd_dim() const { return state_->s.g.n_embd; }
int64_t Engine::layer_count() const { return state_->s.g.n_layers; }
bool Engine::has_vision() const { return state_->s.o.vision; }
const std::vector<int64_t>& Engine::eos_ids() const { return state_->s.o.eos_ids; }

Session::Session(Engine& engine, int64_t ctx) : engine_(&engine), ctx_(ctx) {}

Session::~Session() {
    if (engine_ == nullptr) return;
    EngineState& s = engine_->state_->s;
    s.invalidate_session();
    s.session_ctx = 0;
    engine_->session_ = nullptr;
}

int Session::sync(const int32_t* ids, int64_t n, const ImageSpan* images, size_t image_count,
                  const ProgressFn& progress, const CancelFn& cancel) {
    if (engine_ == nullptr) return kDead;
    EngineState& s = engine_->state_->s;
    s.progress_fn = progress;
    s.cancel_fn = cancel;
    const int status = s.sync_request(ids, n, images, image_count);
    s.progress_fn = nullptr;
    s.cancel_fn = nullptr;
    return status;
}

int64_t Session::reused() const { return engine_ != nullptr ? engine_->state_->s.resume : 0; }

int64_t Session::held() const {
    if (engine_ == nullptr) return -1;
    const EngineState& s = engine_->state_->s;
    return s.pp_exact ? s.pp_reached : -1;
}

int Session::decode(int32_t token, const Sampling& sampling, int budget, int32_t* out, int cap) {
    if (engine_ == nullptr) return kDead;
    return engine_->state_->s.decode_window(token, sampling, budget, out, cap);
}

int Session::finish() { return engine_ != nullptr ? engine_->state_->s.finish_request() : kDead; }

const RequestStats& Session::stats() const {
    static const RequestStats none;
    return engine_ != nullptr ? engine_->state_->s.stats : none;
}

void Session::invalidate() {
    if (engine_ != nullptr) engine_->state_->s.invalidate_session();
}

int64_t Session::ctx() const { return ctx_; }

const std::string& Session::error() const { return engine_ != nullptr ? engine_->state_->s.session_error : g_closed; }

uint64_t image_rows_key(const float* rows, size_t floats) { return ::fnv1a(rows, floats * sizeof(float)); }

}  // namespace strata::program
