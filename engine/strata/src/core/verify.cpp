// src/core/verify.cpp - see include/strata/core/verify.hpp.
#include "strata/core/verify.hpp"
#include "strata/core/host_graph_launch.hpp"
#include "strata/core/dma_batch.hpp"
#include "strata/core/spec_prob.hpp"

#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/route_prior.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/pdl.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/core/progress.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <optional>
#include <thread>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <immintrin.h>

namespace strata::core {
namespace {

constexpr float EPS = 1e-6f;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
const bool g_dbg = std::getenv("STRATA_VERIFY_DEBUG") != nullptr;
inline bool g_lfuse() { static const bool on = [] { const char* v = std::getenv("STRATA_LFUSE"); return v != nullptr && v[0] == '1'; }(); return on; }
// STRATA_ROUTE_RESIDENT=<margin logits> (+ STRATA_ROUTE_RESIDENT_RANKS=lo-hi, default 6-9): residency-biased routing.
struct RouteResidentCfg;
RouteResidentCfg& route_resident_cfg();
struct RouteResidentCfg {
    float margin = 0.0f;
    int lo = 6, hi = 9;
    // #1578: one counter block per GPU.  A layer split inits a verifier per stage; a block allocated on whichever device
    // was current at the first init is unreachable from the other stage's kernels without P2P (illegal memory access).
    // The Verifier asks for its device's block in init (outside graph capture) and keeps the pointer: a launch never
    // picks the block by the host thread's current device.  nullptr (the allocation failed): the kernel runs without counters.
    std::mutex mu;
    std::map<int, unsigned long long*> blocks;
    unsigned long long* stats_on(int dev) {
        if (margin <= 0.0f || dev < 0) return nullptr;
        std::lock_guard<std::mutex> lk(mu);
        const auto it = blocks.find(dev);
        if (it != blocks.end()) return it->second;
        unsigned long long* p = nullptr;
        {
            const OnDevice on(dev);
            if (cudaMalloc((void**) &p, 8 * sizeof(unsigned long long)) != cudaSuccess ||
                cudaMemset(p, 0, 8 * sizeof(unsigned long long)) != cudaSuccess) {
                (void) cudaGetLastError();
                if (p != nullptr) cudaFree(p);
                p = nullptr;
                std::fprintf(stderr, "route-resident: the counters could not be allocated on GPU %d; it runs without them\n", dev);
            }
        }
        if (blocks.empty()) {
            std::atexit([] {
                RouteResidentCfg& c = route_resident_cfg();
                for (const auto& kv : c.blocks) {
                    if (kv.second == nullptr) continue;
                    unsigned long long h[8] = {};
                    if (cudaSetDevice(kv.first) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
                        cudaMemcpy(h, kv.second, sizeof(h), cudaMemcpyDeviceToHost) != cudaSuccess) {
                        (void) cudaGetLastError();
                        continue;
                    }
                    char who[32] = "";
                    if (c.blocks.size() > 1) std::snprintf(who, sizeof who, " (GPU %d)", kv.first);
                    std::fprintf(stderr, "route-resident%s: margin=%g ranks=%d-%d windows T>8 (prompt reads): tail_nonres=%llu swaps=%llu (%.1f%%) nonres_entries before=%llu after=%llu\n",
                                 who, c.margin, c.lo, c.hi, h[0], h[1], h[0] ? 100.0 * h[1] / h[0] : 0.0, h[2], h[3]);
                    std::fprintf(stderr, "route-resident%s: margin=%g ranks=%d-%d windows T<=8 (decode): tail_nonres=%llu swaps=%llu (%.1f%%) nonres_entries before=%llu after=%llu\n",
                                 who, c.margin, c.lo, c.hi, h[4], h[5], h[4] ? 100.0 * h[5] / h[4] : 0.0, h[6], h[7]);
                }
            });
        }
        blocks[dev] = p;
        return p;
    }
};
RouteResidentCfg& route_resident_cfg() {
    static RouteResidentCfg c;
    static const bool parsed = [] {
        if (const char* v = std::getenv("STRATA_ROUTE_RESIDENT")) c.margin = (float) std::atof(v);
        if (const char* v = std::getenv("STRATA_ROUTE_RESIDENT_RANKS")) std::sscanf(v, "%d-%d", &c.lo, &c.hi);
        return true;
    }();
    (void) parsed;
    return c;
}
inline bool g_lfuse_gate() { static const bool on = [] { const char* v = std::getenv("STRATA_LFUSE_GATE"); return v == nullptr || v[0] != '0'; }(); return on; }
inline bool g_lfuse_pair() { static const bool on = [] { const char* v = std::getenv("STRATA_LFUSE_PAIR"); return v == nullptr || v[0] != '0'; }(); return on; }
inline bool g_qdedup() { static const bool on = [] { const char* v = std::getenv("STRATA_VERIFY_QDEDUP"); return v != nullptr && std::atoi(v) != 0; }(); return on; }
// S26 STRATA_QFUSE=1: activation q8_1 images written by their producers (the GDN output norm) - the same bytes
// fork F4 (Eddoursul): 2-4 token dense projections read an interleaved copy of the q8_1 activations (bitwise the same
// outputs); STRATA_MMVQ_IL=0 keeps native_mmvq's multi-column kernels
inline bool g_mmvq_il() { static const bool on = [] { const char* v = std::getenv("STRATA_MMVQ_IL"); return v == nullptr || v[0] != '0'; }(); return on; }
// 143-amdk: STRATA_QFUSE_HC - only the hyper-connection read writes its consumer's q8_1 image (not the GDN recurrence): gfx12 default (=0 / 1 forces off / on)
inline bool g_qfuse_hc() {
    return false;
}
// fork 34a35db + b063c13 (Eddoursul), STRATA_SIDE_MIXER=1 / =2 (opt-in): the mixer's work that reads only the layer's
// input on branches of the window's graph (bitwise the same outputs); 0 / unset = in line, and no branch stream is created
inline int g_side_mixer() { static const int m = [] { const char* v = std::getenv("STRATA_SIDE_MIXER"); return v ? std::max(0, std::min(2, std::atoi(v))) : 0; }(); return m; }
inline bool g_qfuse() { static const bool on = [] { const char* v = std::getenv("STRATA_QFUSE"); return v != nullptr && std::atoi(v) != 0; }(); return on; }
#define VDBG(...) do { if (g_dbg) { std::fprintf(stderr, "verify dbg: " __VA_ARGS__); std::fflush(stderr); } } while (0)

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

// #649 (HIP, opt-in A/Bs for the gfx1030 verify timeouts; CUDA never reads them)
//   STRATA_VERIFY_COHERENT=1  the handshake words and rows in explicitly coherent (fine-grained) host memory
//   STRATA_DOORBELL_STORE=1   the GPU stores each step's ring instead of read-modify-writing it over PCIe
bool env_on(const char* name) {
    const char* e = std::getenv(name);
    return e != nullptr && e[0] != 0 && e[0] != '0';
}
const bool g_trace = env_on("STRATA_VERIFY_TRACE");
const bool g_no_multi_gr = env_on("STRATA_NO_MULTI_GR");   // #783 PR-g: per-token GR reads/writes and generic-T kernels again
const bool g_no_batch_kv = env_on("STRATA_NO_BATCH_KV_STEP");   // #783 PR-d: per-token K/V and indexer appends again

// The shared expert runs on its own stream, forked off and joined back per layer (`sh_fork` in record_window).  The
// overlap pays off on CUDA, where it was added (cfd3b72).  On HIP a cross-stream event dependency costs more than the
// shared expert takes to run, so there the fork starts off: same binary, same 48,067-token prompt, three runs each on a
// Radeon AI PRO R9700 (gfx1201, ROCm 6.4.3), the fork off decodes at 62.9 t/s against 43.4 t/s with it on.  The
// profiler turns the fork off to time the stages, so a profiled run never showed that.  #816 reports the same fall on a
// gfx1030 (RX 6800, Windows, 42.5 -> 56.4 t/s).  STRATA_SH_STREAM overrides either default: `=1` forks, `=0` does not.
// Strix Halo (gfx1151, unified memory) is the exception: the fork pays there (+1.8% UD-IQ4_XS, +6.7% UD-Q4_K_XL decode
// at 8K, ids identical), so the gfx1151 arch table sets STRATA_SH_STREAM=1.  Read on first use, after that table ran.
bool sh_stream_on() {
    static const bool on = [] {
        if (const char* v = std::getenv("STRATA_SH_STREAM")) return v[0] != '0';
        return true;
    }();
    return on;
}

// A one-token window always keeps its token, so its graph advances the sequence state itself (the GDN conv history
// and recurrence state) and Verifier::commit launches no commit graph after it (eddoursul's fork, F7).  CUDA only:
// HIP keeps the commit graph after every window (STRATA_ONE_TOKEN_COMMIT=0 does too).
// The head's hyper-connection read for the window's tokens in one launch set (CUDA; HIP keeps gr_read per token;
// STRATA_HEAD_MIX_MULTI=0 does too): bitwise the same sums.
bool head_mix_multi_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_HEAD_MIX_MULTI");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

// 143-amdk: the wait for a host flag folded into the copy that follows it (wait_copy_i32_from_mapped,
// copy_rows_from_mapped_after): same reads, same values, one kernel node fewer each.  A dependent kernel costs ~2.4 us in a
// gfx1201 graph whatever it does; measured on gfx12 only, so it is on there by default.  STRATA_FUSE_WAIT=0 / 1 forces it
// off / on (HIP builds; a CUDA build never fuses).
bool fuse_wait_on() {
    // CUDA: STRATA_FUSE_WAIT=1 folds both waits (the plan's and the CPU rows'), =2 only the plan's, =0 neither.  Turing
    // default 2 (RTX 2080 Ti, Flash-Next Q2_0, two runs each: +0.6..+1.4 % decode); the rows' version makes every row
    // block of the copy poll the flag over PCIe and was -1.5 % there, so it stays opt-in.
    static const int env = [] { const char* v = std::getenv("STRATA_FUSE_WAIT"); return v != nullptr ? std::atoi(v) : -1; }();
    if (env >= 0) return env == 1 || env == 2;
    return false;
}
// 143-amdk's rows version of the folded wait (copy_rows_from_mapped_after): HIP with fuse_wait_on, CUDA STRATA_FUSE_WAIT=1
bool fuse_rows_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_FUSE_WAIT"); return v != nullptr && std::atoi(v) == 1; }();
    return on;
}

// 143-amdk: STRATA_FUSE_CONVAB=1 - the GDN conv and the a/b projections in one launch.  Opt-in: measured -0.3% on gfx1201 (the a/b dot chain is the longer one).
bool fuse_convab_on() {
    return false;
}

// 143-amdk: STRATA_FUSE_HITS - moe_hit_add folded into the combine (gfx12 default; =0 / 1 forces off / on)
bool fuse_hits_on() {
    return false;
}

bool one_token_self_commit() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_ONE_TOKEN_COMMIT");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

bool mapped(size_t bytes, void** h, void** d) {
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped) != cudaSuccess) return false;
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

strata::kernels::QsaShapes shapes_of(const ModelGeometry& g) { return qsa_shapes(g); }

const WeightRef* need(const LayerView& v, const char* suffix, std::string& err) {
    const WeightRef* r = v.get(suffix);
    if (r == nullptr && err.empty()) err = v.name(suffix) + " is missing";
    return r;
}

bool native_of(const WeightRef* w, const std::string& name, std::string& err) {
    if (w == nullptr) return false;
    if (w->native_data == nullptr) {
        err = "verify: " + name + " is not served natively (run with --native)";
        return false;
    }
    return true;
}

}  // namespace

namespace {
std::atomic<const Verifier*> g_diag_verifier{nullptr};
void diag_active_verifier(std::FILE* f) {
    if (const Verifier* v = g_diag_verifier.load()) v->diag(f);
}
// #267: every live verifier (a layer split has one per stage), for the release before the engine ends
constexpr int kLiveMax = 16;
std::atomic<Verifier*> g_live[kLiveMax];
void release_live_verifiers(std::FILE* f) {
    for (auto& slot : g_live)
        if (Verifier* v = slot.load()) {
            const Clock::time_point t0 = Clock::now();
            const bool done = v->release_gpu_waits(5000);
            if (f != nullptr)
                std::fprintf(f, "strata: released the verify window's GPU waits (#267): the GPU %s\n",
                             done ? ("finished in " + std::to_string((long long) ms_since(t0)) + " ms").c_str()
                                  : "did not finish within 5 s");
        }
    if (f != nullptr) std::fflush(f);
}
// #649: the host side of STRATA_VERIFY_TRACE - a ring of the last kTraceN handshake events of every verifier
struct TraceEv {
    int64_t t_ns;
    const void* who;
    int64_t window, step, layer, aux;
    uint32_t seq, flag, a, b;
    const char* what;
};
constexpr uint64_t kTraceN = 4096;
TraceEv g_trace_ring[kTraceN];
std::atomic<uint64_t> g_trace_next{0};
int64_t trace_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
std::string released_note(bool drained) {
    return drained ? "; its GPU waits were released and the GPU finished (#267)"
                   : "; its GPU waits were released but the GPU did not finish within 5 s (#267)";
}
// #267 test hook: STRATA_TEST_VERIFY_STALL=N withholds the last layer's flag in the N-th window (1-based), so the
// GPU spins on a flag nobody raises - the bounded window wait and the release are then what ends it.  Unset: never.
const int64_t g_test_stall = [] {
    const char* e = std::getenv("STRATA_TEST_VERIFY_STALL");
    return e != nullptr ? (int64_t) std::atoll(e) : (int64_t) 0;
}();
}  // namespace

bool Verifier::release_gpu_waits(int timeout_ms) {
    released_.store(true);
    trace_ev("RELEASE", -1, -1, timeout_ms);
    // the words the spin kernels read (wait_flag_ge, wait_flag_ge_or) are mapped host memory, so a store here
    // reaches them with no API call; UINT32_MAX is past every ring.  (E-6's skip words are device memory, but
    // wait_flag_ge_or also returns on its flag.)  A host function raising flag B later only raises.
    for (uint32_t* p : {h_flag_, h_flagA_, h_flagB_})
        if (p != nullptr) *(volatile uint32_t*) p = UINT32_MAX;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    for (cudaStream_t s : {cs_, copy_}) {
        if (s == nullptr) continue;
        while (cudaStreamQuery(s) == cudaErrorNotReady) {
            if (ms_since(t0) > timeout_ms) {
                trace_ev("RELEASE-NOT-DRAINED", -1, -1, (int64_t) ms_since(t0));
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    trace_ev("RELEASE-DRAINED", -1, -1, (int64_t) ms_since(t0));   // aux: ms the GPU took to finish once released
    return true;
}

void Verifier::trace_ev(const char* what, int64_t step, int64_t layer, int64_t aux) const {
    if (!g_trace) return;
    auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
    TraceEv& e = g_trace_ring[g_trace_next.fetch_add(1) % kTraceN];
    e = {trace_now_ns(), this, windows, step, layer, aux, rd(h_seq_), rd(h_flag_), rd(h_flagA_), rd(h_flagB_), what};
}

void Verifier::trace_dump(std::FILE* f) const {
    if (!g_trace || f == nullptr) return;
    static const char* names[kProfPer] = {"pre", "hc-read0", "qkv gemv", "conv", "ab", "z", "rec", "kv-idx",
                                          "k/v rope", "kv append", "q", "scores+topk", "kv-resolve", "attention",
                                          "gate", "", "out-proj", "router+ring", "shared", "waitA", "VRAM hits", "waitB",
                                          "PCIe grp", "waitCPU", "combine", "", "head", "", "", "", "", "", ""};
    const uint64_t end = g_trace_next.load();
    const uint64_t n = end < 160 ? end : 160;
    std::fprintf(f, "strata verify trace (#649): the last %llu handshake events (host clock, us before the last; "
                    "words: seq = GPU rang, flag = CPU served, A = plan, B = copies)\n", (unsigned long long) n);
    const int64_t t_last = n ? g_trace_ring[(end - 1) % kTraceN].t_ns : 0;
    for (uint64_t i = end - n; i < end; ++i) {
        const TraceEv& e = g_trace_ring[i % kTraceN];
        if (e.what == nullptr) continue;
        std::fprintf(f, "  %10.1f %-20s v%p window %lld step %lld layer %lld aux %lld | seq %u flag %u A %u B %u\n",
                     (double) (t_last - e.t_ns) / 1000.0, e.what, e.who, (long long) e.window, (long long) e.step,
                     (long long) e.layer, (long long) e.aux, e.seq, e.flag, e.a, e.b);
    }
    if (trace_h_ == nullptr || g_ == nullptr) return;
    // the breadcrumbs: every stamp point the GPU passed in this window (ns of the GPU clock after the window's first)
    unsigned long long t0 = ~0ull;
    for (size_t i = 0; i < trace_n_; ++i) {
        const unsigned long long t = *(const volatile unsigned long long*) (trace_h_ + i);
        if (t != 0 && t < t0) t0 = t;
    }
    if (t0 == ~0ull) {
        std::fprintf(f, "strata verify trace: no GPU breadcrumb in this window (the GPU never started it)\n");
        return;
    }
    int64_t last_l = -1;
    for (int64_t l = 0; l <= g_->n_layers; ++l)
        for (int i = 0; i < kProfPer; ++i)
            for (int grp = 0; grp < 2; ++grp)
                if (trace_h_[(size_t) ((l * kProfPer + i) * 2 + grp)] != 0) last_l = l;
    std::fprintf(f, "strata verify trace: GPU breadcrumbs (us after the window's first), the GPU got as far as layer "
                    "%lld:\n", (long long) last_l);
    for (int64_t l = std::max<int64_t>(0, last_l - 2); l <= std::min<int64_t>(g_->n_layers, last_l + 1); ++l)
        for (int grp = 0; grp < 2; ++grp) {
            std::string line;
            char b[64];
            for (int i = 0; i < kProfPer; ++i) {
                const unsigned long long t = trace_h_[(size_t) ((l * kProfPer + i) * 2 + grp)];
                if (t == 0) continue;
                std::snprintf(b, sizeof b, " %s(%d)=%.1f", names[i], i, (double) (t - t0) / 1000.0);
                line += b;
            }
            if (!line.empty())
                std::fprintf(f, "  layer %lld group %d:%s\n", (long long) l, grp, line.c_str());
        }
    std::fflush(f);
}

void Verifier::diag(std::FILE* f) const {
    auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
    // #251: outside a verify stage these are the LAST window's numbers (it finished), not the stalled work's
    const char* where = progress().where.load();
    const bool current = where != nullptr && std::strncmp(where, "verify window", 13) == 0;
    std::fprintf(f, "  verify window%s: %d tokens at position %lld, host at layer step %u; the GPU rang %u; flags: "
                    "served %u, plan (A) %u, copies (B) %u\n", current ? "" : " (last window, not the current stage)",
                 last_t_, (long long) last_pos0_, cur_layer_ + 1, rd(h_seq_), rd(h_flag_), rd(h_flagA_), rd(h_flagB_));
    trace_dump(f);   // #649: STRATA_VERIFY_TRACE=1 only
}


Verifier::~Verifier() {
    const Verifier* self = this;
    g_diag_verifier.compare_exchange_strong(self, nullptr);
    for (auto& slot : g_live) {
        Verifier* me = this;
        slot.compare_exchange_strong(me, nullptr);
    }
    if (cs_) cudaStreamSynchronize(cs_);
    if (sh_cs_) cudaStreamSynchronize(sh_cs_);
    for (auto& e : exec_)
        if (e) cudaGraphExecDestroy(e);
    for (auto& e : exec_nr_)
        if (e) cudaGraphExecDestroy(e);
    if (commit_exec_) cudaGraphExecDestroy(commit_exec_);
    if (qcnt_) cudaFree(qcnt_);
    if (d_spec_) cudaFree(d_spec_);
    if (logit_bias_device_) cudaFree(logit_bias_device_);
    if (cs_) cudaStreamDestroy(cs_);
    if (sh_cs_) cudaStreamDestroy(sh_cs_);
    if (copy_) { cudaStreamSynchronize(copy_); cudaStreamDestroy(copy_); }
    if (side_) cudaStreamDestroy(side_);
    if (side2_) cudaStreamDestroy(side2_);
    for (cudaEvent_t e : {mfork_, mjoin_, pfork_, pjoin_})
        if (e) cudaEventDestroy(e);
    if (commit_done_) cudaEventDestroy(commit_done_);
    if (ev_fork_) cudaEventDestroy(ev_fork_);
    if (ev_join_) cudaEventDestroy(ev_join_);
    for (cudaStream_t s : df_side_)
        if (s) cudaStreamDestroy(s);
    if (df_fork_) cudaEventDestroy(df_fork_);
    for (cudaEvent_t e : df_join_)
        if (e) cudaEventDestroy(e);
    if (arena_) cudaFree(arena_);
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_commit_, h_ple_, h_out_, h_x_, h_ids_, h_w_, h_seq_, h_flag_, h_ymiss_,
                     h_flagA_, h_plan_, h_flagB_, h_plan_err_, h_nf_, h_bal_};
    for (void* h : hosts)
        if (h) cudaFreeHost(h);
}

bool Verifier::set_logit_bias(const std::vector<float>& bias, std::string& err) {
    const OnDevice on_device(device_);
    if (!bias.empty() && bias.size() != (size_t) n_vocab_) {
        err = "logit_bias vocabulary size mismatch"; return false;
    }
    if (bias != logit_bias_host_ && !bias.empty()) {
        const size_t bytes = bias.size() * sizeof(float);
        if ((!logit_bias_device_ && cudaMalloc((void**) &logit_bias_device_, bytes) != cudaSuccess) ||
            cudaMemcpy(logit_bias_device_, bias.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "cannot upload logit_bias"; return false;
        }
    }
    logit_bias_host_ = bias;
    sampling_.logit_bias = bias.empty() ? nullptr : logit_bias_device_;
    return true;
}

bool Verifier::init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, const VerifyHits& hits,
                    const NativeHead* head, int max_t, std::string& err) {
    g_diag_verifier.store(this);
    diag_verify_fn().store(&diag_active_verifier);
    for (auto& slot : g_live) {
        Verifier* none = nullptr;
        if (slot.load() == this || slot.compare_exchange_strong(none, this)) break;
    }
    release_gpu_fn().store(&release_live_verifiers);
    cudaGetDevice(&device_);   // a layer split's stage on another GPU: its streams, graphs and buffers live there
    rr_stats_ = route_resident_cfg().stats_on(device_);   // STRATA_ROUTE_RESIDENT: this GPU's counters, outside graph capture
    strata::kernels::fused_gr_check();   // once per card: which bitwise-equal hyper-connection read runs there
    wt_ = &wt;
    g_ = &g;
    ss_ = &ss;
    hits_ = hits;
    head_ = head;
    max_t_ = max_t;
    sampling_.greedy = true;      // a fresh verifier samples greedily until set_sampling says otherwise
    sampling_.temperature = 0.0f;
    if (max_t < 2 || max_t > strata::kernels::kVerifyMaxT || max_t > strata::kernels::cpu::MAXT) {
        err = "verify: the window must hold 2.." + std::to_string(strata::kernels::kVerifyMaxT) + " tokens";
        return false;
    }
    if (hits.d_res == nullptr || hits.cache_base == nullptr || hits.blob <= 0) {
        err = "verify: needs the profile-filled VRAM expert tier (--expert-profile and --expert-cache); with "
              "--expert-cache auto, no VRAM was left for it - the 'no VRAM is left for the expert cache' line above "
              "says how much is short and what makes room (a smaller --max-context, --kv q4_0, setup --draft-vocab en, "
              "images on the CPU)";
        return false;
    }
    std::string why;
    if (!layer_verify_compatible(why)) {
        err = "verify: " + why + " (the verify window reproduces the default native decode path)";
        return false;
    }
    if (!strata::kernels::fused_gr_supported(g.n_embd, g.hc, g.hc_lr) || ss.k != 10 || g.ssm_state_size != 128 ||
        g.ssm_d_conv != 4) {
        err = "verify: geometry differs from the artifact's";
        return false;
    }
    if (le_ < 0) le_ = g.n_layers;
    if (lb_ < 0 || lb_ >= le_ || le_ > g.n_layers || le_ < g.n_layers) {
        err = "verify: the stage's layer range or its hand-off buffers are wrong";
        return false;
    }
    const WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) { err = "verify: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;

    const strata::kernels::QsaShapes s = shapes_of(g);
    cap_ = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    max_blocks_ = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, s);

    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t C = (uint64_t) g.ssm_conv_channels, ZV = (uint64_t) g.ssm_value_dim, HV = (uint64_t) g.ssm_v_heads;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t IQ = (uint64_t) g.idx_q_heads, ID = (uint64_t) g.idx_key_dim;
    const uint64_t nG = (uint64_t) g.n_gdn_layers(), nQ = (uint64_t) g.n_qsa_layers();
    const uint64_t HS = (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM;
    const uint64_t TS = (uint64_t) (s.idx_block - 1) * ID;
    const int max_in = (int) std::max<uint64_t>(std::max<uint64_t>(N, ZV), NH * HD);

    // ---- mapped staging
    bool ok = mapped(T * 4, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(T * strata::kernels::kStepCount * 4, (void**) &h_step_, (void**) &m_step_) &&
              mapped(T * (NH + NKV + IQ) * 4, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped((2 + T) * 4 + 16, (void**) &h_commit_, (void**) &m_commit_) &&
              mapped(T * N * 4, (void**) &h_ple_, (void**) &m_ple_) &&
              mapped(T * 4 + 16, (void**) &h_out_, (void**) &m_out_) &&
              mapped((size_t) strata::kernels::kVerifyMaxT * strata::kernels::kNonfiniteBlocks * 4, (void**) &h_nf_,
                     (void**) &m_nf_) &&
              mapped(T * N * 4, (void**) &h_x_, (void**) &m_x_) &&
              mapped(T * K * 4, (void**) &h_ids_, (void**) &m_ids_) &&
              mapped(T * K * 4, (void**) &h_w_, (void**) &m_w_) &&
              mapped(64, (void**) &h_seq_, (void**) &m_seq_) &&
              mapped(64, (void**) &h_flag_, (void**) &m_flag_) &&
              mapped(64, (void**) &h_flagA_, (void**) &m_flagA_) &&
              mapped(64, (void**) &h_flagB_, (void**) &m_flagB_) &&
              mapped(64, (void**) &h_plan_err_, (void**) &m_plan_err_) &&
              mapped(T * K * N * 4, (void**) &h_ymiss_, (void**) &m_ymiss_);
    if (!ok) { err = "verify: mapped staging allocation failed"; return false; }
    if (bal_ != nullptr) {   // set_pcie_balance: four stamp words per (layer, group)
        if (!mapped((size_t) g.n_layers * 2 * 4 * 8, (void**) &h_bal_, (void**) &m_bal_)) {
            err = "verify: mapped balance stamps allocation failed";
            return false;
        }
        bal_p_.assign((size_t) g.n_layers * 2, 0.0);
        bal_nc_.assign((size_t) g.n_layers * 2, 0);
        bal_np_.assign((size_t) g.n_layers * 2, 0);
        bal_ok_.assign((size_t) g.n_layers * 2, 0);
        bal_->on = true;
    }
    // the GPU plan: counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) | start2(cap+1)
    {
        const int64_t cap = (int64_t) (T * K);
        const int64_t i32 = 4 + (cap + 1) + cap + cap;
        const int64_t ptr_off = (i32 + 1) & ~1ll;
        plan_i32_ = ptr_off + 4 * cap + (cap + 1) + 1;
        if (!mapped((size_t) plan_i32_ * 4 * 2 + 64, (void**) &h_plan_, (void**) &m_plan_)) {
            err = "verify: mapped plan allocation failed";
            return false;
        }
        sink_.counts = h_plan_;
        sink_.start = h_plan_ + 4;
        sink_.dst = sink_.start + cap + 1;
        sink_.tok = sink_.dst + cap;
        sink_.ptr = (unsigned long long*) (h_plan_ + ptr_off);
        sink_.ptr2 = sink_.ptr + cap;
        sink_.start2 = h_plan_ + ptr_off + 4 * cap;
        sink_.cap = cap;
        sink_.publish = &Verifier::publish_plan;
        sink_.fetch = &Verifier::fetch_dma;
        sink_.ctx = this;
    }

    // ---- the device arena: the same sequence counted, then carved
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T); step_ = b.take<int32_t>(T * strata::kernels::kStepCount);
        pos_ = b.take<int32_t>(T * (NH + NKV + IQ)); commit_ = b.take<int32_t>(2 + T);
        ple_ = b.take<float>(T * N); emb_ = b.take<float>(T * N); R_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N); bo_ = b.take<float>(T * N);
        inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC); xn_ = b.take<float>(T * HC * N);
        xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes(max_in, (int) T));
        sh_xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes(max_in, (int) T));
        xil_ = b.take<uint8_t>(strata::kernels::native_q8_1_il_bytes(max_in, 4));
        qkv_L_ = b.take<float>(nG * T * C); h_L_ = b.take<float>(nG * T * C);
        gate_L_ = b.take<float>(nG * T * HV); beta_L_ = b.take<float>(nG * T * HV);
        z_ = b.take<float>(T * ZV); y_ = b.take<float>(T * ZV); y_dummy_ = b.take<float>(T * ZV);
        qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
        idx_raw_L_ = b.take<float>(nQ * T * ID); qidx_ = b.take<float>(T * IQ * ID);
        scores_ = b.take<float>(T * (uint64_t) max_blocks_); sel_ = b.take<int32_t>(T * (uint64_t) cap_);
        attn_ = b.take<float>(T * NH * HD); attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>(T * (uint64_t) attn_scratch_floats_);
        tail_snap_ = b.take<float>(nQ * TS);
        logits_ = b.take<float>(T * (uint64_t) g.n_expert); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
        shared_ = b.take<float>(T * N); parts_ = b.take<float>(T * K * N); hit_out_ = b.take<float>(T * K * N);
        hit_slot_ = b.take<int32_t>(T * K); hit_dst_ = b.take<int32_t>(T * K); hit_count_ = b.take<int32_t>(4);
        plan_ = b.take<int32_t>(2 * ((uint64_t) plan_i32_ + 16));
        staging_ = b.take<uint8_t>((uint64_t) kStagingBlobs * strata::kernels::cpu::expert_layout().max_blob);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34); hit_xs_ = b.take<float>(T * (N / 32));
        nat_xq_ = b.take<uint8_t>(T * (N / 32) * 36);
        hit_scratch_ = b.take<uint8_t>(std::max<uint64_t>(
            strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff),
            strata::kernels::native_expert_scratch_bytes((int64_t) (T * K), g.n_ff)));
        head_mixed_ = b.take<float>(T * N); head_inj_ = b.take<float>(HC);
        sh_bf16_ = b.take<uint16_t>(T * N); sh_gate_ = b.take<float>(T * (uint64_t) g.n_ff);
        sh_up_ = b.take<float>(T * (uint64_t) g.n_ff); sh_g_ = b.take<float>(T + 4);
        head_logits_ = b.take<float>(T * (uint64_t) n_vocab_);
        arg_scratch_ = b.take<uint8_t>(strata::kernels::argmax_rows_scratch_bytes((int) T));
        one_ = b.take<int32_t>(4);
        hist_snap_ = b.take<float>(T * HS);
        ple_key_ = b.take<float>(T * (uint64_t) strata::kernels::NG_HC_DIM); ple_val_ = b.take<float>(T * N);
    };
    Bump count;
    carve(count);
    if (cudaMalloc(&arena_, count.used) != cudaSuccess) {
        err = "verify: the device arena (" + std::to_string(count.used >> 20) + " MiB) does not fit";
        return false;
    }
    cudaMemset(arena_, 0, count.used);
    if (g_trace && trace_h_ == nullptr) {   // #649: the breadcrumbs, mapped so they read while the GPU hangs
        trace_n_ = (size_t) (g.n_layers + 1) * kProfPer * 2;
        if (!mapped(trace_n_ * 8, (void**) &trace_h_, (void**) &trace_m_)) {
            (void) cudaGetLastError();
            trace_h_ = trace_m_ = nullptr;
            trace_n_ = 0;
        }
        std::fprintf(stderr, "strata verify trace (#649): on\n");
    }
    prof_on_ = std::getenv("STRATA_VERIFY_PROFILE") != nullptr;
    if (prof_on_) {
        const size_t np = (size_t) g.n_layers * kProfPer + 4;
        if (cudaMalloc((void**) &prof_, np * 8) != cudaSuccess) { prof_on_ = false; prof_ = nullptr; cudaGetLastError(); }
        else { cudaMemset(prof_, 0, np * 8); prof_h_.assign(np, 0); }
    }
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    {
        const int32_t one = 1;
        if (cudaMemcpy(one_, &one, sizeof one, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "verify: the arena could not be set";
            return false;
        }
    }
    sink_.staging = (unsigned long long) staging_;
    sink_.staging_cap = kStagingBlobs;
    (void) TS;
    // STRATA_PIPED_CONCURRENT=1 (opt-in): a prompt can run on the same GPU while the batch windows run, so the windows'
    // streams get the device's highest priority (HIP: the default priority, as the drafter's)
    auto make_stream = [](cudaStream_t* s) -> cudaError_t {
        static const bool hi = [] {   // (STRATA_PIPED_CONCURRENT_PRIO=0: the default priority)
            const char* v = std::getenv("STRATA_PIPED_CONCURRENT");
            const char* p = std::getenv("STRATA_PIPED_CONCURRENT_PRIO");
            return v != nullptr && std::atoi(v) != 0 && (p == nullptr || std::atoi(p) != 0);
        }();
        int lo = 0, top = 0;
        if (hi) {
            if (cudaDeviceGetStreamPriorityRange(&lo, &top) == cudaSuccess &&
                cudaStreamCreateWithPriority(s, cudaStreamNonBlocking, top) == cudaSuccess)
                return cudaSuccess;
            cudaGetLastError();   // (only the failed priority call's error: the default path clears nothing)
        }
        return cudaStreamCreateWithFlags(s, cudaStreamNonBlocking);
    };
    bool copy_made = false;
    if (!copy_made && make_stream(&copy_) != cudaSuccess) {
        err = "verify: copy stream create failed";
        return false;
    }
    if (make_stream(&cs_) != cudaSuccess) cs_ = nullptr;
    if (cs_ == nullptr || make_stream(&sh_cs_) != cudaSuccess) {
        err = "verify: stream create failed";
        return false;
    }
    if (cudaEventCreateWithFlags(&commit_done_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&ev_fork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&ev_join_, cudaEventDisableTiming) != cudaSuccess) {
        err = "verify: event create failed";
        return false;
    }
    {   // STRATA_DF_BRANCH=1 (opt-in): the mixer's side branches.  Off by default: on Linux (open modules, GSP) a
        // stage's window with the branches stopped inside its graph when an NVML query (nvidia-smi) came between
        // requests (2x RTX 3090 and 2x RTX 2080 Ti, #905); never seen on Windows.  0 / unset: one stream, as before
        const char* v = std::getenv("STRATA_DF_BRANCH");
        df_branch_ = v != nullptr && std::atoi(v) != 0;
        if (df_branch_ && df_fork_ == nullptr) {
            bool okb = cudaEventCreateWithFlags(&df_fork_, cudaEventDisableTiming) == cudaSuccess;
            for (int i = 0; i < 2 && okb; ++i)
                okb = cudaStreamCreateWithFlags(&df_side_[i], cudaStreamNonBlocking) == cudaSuccess &&
                      cudaEventCreateWithFlags(&df_join_[i], cudaEventDisableTiming) == cudaSuccess;
            if (!okb) { cudaGetLastError(); df_branch_ = false; }
        }
    }
    // Check whether 100% of experts across [lb_, le_) are resident in this stage's VRAM cache.
    // When true, every layer plans on device and writes directly into parts_ without any CPU doorbells,
    // wait_flag_ge spins, PCIe empty launches, or moe_hit_add copies.
    all_resident_ = false;
    ar_off_ = false;
    h_res_ = hits.h_res;
    if (h_plan_err_ != nullptr) *(volatile uint32_t*) h_plan_err_ = 0;
    if (hits.h_res != nullptr && hits.d_res != nullptr && hits.cache_base != nullptr) {
        const char* v_ar = std::getenv("STRATA_VERIFY_ALL_RESIDENT");
        if (v_ar == nullptr || std::atoi(v_ar) != 0) {
            bool all_ok = true;
            for (int64_t l = lb_; l < le_ && all_ok; ++l) {
                for (int64_t e = 0; e < g.n_expert; ++e) {
                    if (hits.h_res[l * g.n_expert + e] < 0) {
                        all_ok = false;
                        break;
                    }
                }
            }
            all_resident_ = all_ok;
        }
    }
    if (g_side_mixer() > 0 &&
        (cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&mfork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&mjoin_, cudaEventDisableTiming) != cudaSuccess ||
        cudaStreamCreateWithFlags(&side2_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&pfork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&pjoin_, cudaEventDisableTiming) != cudaSuccess)) {
        err = "verify: the mixer branch's stream / events could not be created";
        return false;
    }
    // E-6: a layer whose routed experts are all resident is planned on the device (STRATA_VERIFY_DEVICE_PLAN=1: on;
    // exact, but neutral on RIBPC 1-2 GPUs: off by default)
    {
        const char* v = std::getenv("STRATA_VERIFY_DEVICE_PLAN");
        device_plan_ = !all_resident_ && (v != nullptr && std::atoi(v) != 0);
    }
    // (halo's STRATA_VERIFY_RESIDENT=1 is this window's all_resident_ graph above, which 0.1.39 has on by default)
    if (g_qfuse() || g_qfuse_hc()) {   // S26: the HC read's q8_1 group counters, zeroed once (each launch leaves them zero)
        if (cudaMalloc((void**) &qcnt_, sizeof(unsigned) * (size_t) (g.n_embd / 32)) != cudaSuccess ||
            cudaMemset(qcnt_, 0, sizeof(unsigned) * (size_t) (g.n_embd / 32)) != cudaSuccess) {
            cudaGetLastError();
            if (qcnt_) cudaFree(qcnt_);
            qcnt_ = nullptr;
        }
    }
    if (all_resident_ || device_plan_) {
        bool ok2 = true;
        if (device_plan_)
            ok2 = cudaMalloc((void**) &skip_, 64) == cudaSuccess && cudaMemset(skip_, 0, 64) == cudaSuccess;
        if (ok2 && hits.slot_off != nullptr && hits.n_slots > 0) {
            ok2 = cudaMalloc((void**) &slot_off_d_, (size_t) hits.n_slots * sizeof(unsigned long long)) == cudaSuccess &&
                  cudaMemcpy(slot_off_d_, hits.slot_off, (size_t) hits.n_slots * sizeof(unsigned long long),
                             cudaMemcpyHostToDevice) == cudaSuccess;
        }
        if (!ok2) { cudaGetLastError(); all_resident_ = false; device_plan_ = false; }
    }
    std::fprintf(stderr, "strata verify: window up to %d tokens, %.1f MiB of device buffers%s\n", max_t,
                 (double) count.used / 1048576.0,
                 all_resident_ ? " (100% VRAM resident: zero-doorbell graph)" : "");
    return true;
}

const float* Verifier::final_R(int t) const { return R_ + (size_t) t * (size_t) (g_->hc * g_->n_embd); }

// ================================ THE WINDOW, AS CAPTURED ================================
//
// Plan v0.3 P6 (split window): with `groups_ == 2` the window's tokens are cut into two groups A = [0, T/2 up) and
// B = the rest, and the stream is ordered
//
//     pre(0,A) pre(0,B) | post(0,A) pre(1,A) | post(0,B) pre(1,B) | post(1,A) pre(2,A) | ...
//
// so the CPU computes A's experts of layer l while the GPU runs B's mixer and router of layer l, and B's experts
// while the GPU combines A and runs A's layer l+1.  B's mixer only needs A's mixer of the same layer (K/V, GDN
// state), never A's experts, so nothing waits that did not wait before.  Every token's arithmetic is unchanged.
bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const WeightTable& wt = *wt_;
    SessionState& ss = *ss_;
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, C = g.ssm_conv_channels, ZV = g.ssm_value_dim;
    const int64_t HV = g.ssm_v_heads, HK = g.ssm_k_heads, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const int64_t IQ = g.idx_q_heads, ID = g.idx_key_dim, NE = g.n_expert, MT = max_t_;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    const int64_t TS = (s.idx_block - 1) * ID;
    const bool ple_on = ss.ple.ready() && ple_stage();
    auto Rt = [&](int t) { return R_ + (size_t) t * HC * N; };
    const int G = (split_ && T >= 2) ? 2 : 1;
    const bool self_commit = T == 1 && !g_qfuse() && one_token_self_commit();   // see Verifier::commit
    static const bool dec_batch = [] { const char* v = std::getenv("STRATA_DEC_BATCH"); return v == nullptr || std::atoi(v) != 0; }();
    // fork 34a35db: the GDN gates and the QSA indexer beside the projections (STRATA_SIDE_MIXER=1; 0 / unset = in line)
    const int side_mixer_mode = g_side_mixer();
    const bool side_mixer = side_mixer_mode >= 1;
    const bool side_proj = side_mixer_mode >= 2;   // fork b063c13: the GDN gate projection, the QSA queries
    auto branch = [&](cudaEvent_t e, cudaStream_t from, cudaStream_t to, const char* what) {
        if (cudaEventRecord(e, from) == cudaSuccess && cudaStreamWaitEvent(to, e, 0) == cudaSuccess) return true;
        err = std::string("verify: the mixer branch could not ") + what;
        return false;
    };
    const bool sidem = side_mixer;
    auto stamp = [&](int64_t l, int i, int grp) {
        if (prof_on_ && grp == 0) gpu_stamp(prof_, (int) (l * kProfPer + i), cs);
        if (trace_m_ != nullptr) gpu_stamp(trace_m_, (int) ((l * kProfPer + i) * 2 + grp), cs);   // #649
    };
    const int tb_[2] = {0, (T + 1) / 2}, te_[2] = {G == 2 ? (T + 1) / 2 : T, T};
    // S26 STRATA_LFUSE=1: where the shared expert's gate / scale fusions apply (the paths they replace are the ones taken)
    auto lfuse_on = [&](int n) {
        return g_lfuse() && ar_on() && native_moe_combine_enabled() && dec_batch && n > 1 && n <= 8 &&
               shared_expert_native_bf16_enabled();
    };
    bool sg_gated_[2] = {false, false};   // per group: the combine applies the shared gate
    groups_[T] = G;
    if (G == 1) pdl_supported();   // pdl.hpp: its first call per device clears a pending CUDA error
    // STRATA_DF_BRANCH: mixer work that reads only the layer's input runs on a side stream inside the graph.  fork(i)
    // makes side i depend on everything captured on `cs` so far, join(i) makes `cs` depend on everything captured on
    // side i: a branch never races with what came before its fork or after its join.  Not in a split window,
    // nor under the stage profiler (whose stamps are on `cs`).
    const bool br = df_branch_ && G == 1 && !prof_on_ && !side_mixer;   // STRATA_SIDE_MIXER wins
    auto fork = [&](int i) {
        cudaEventRecord(df_fork_, cs);
        cudaStreamWaitEvent(df_side_[i], df_fork_, 0);
    };
    auto join = [&](int i) {
        cudaEventRecord(df_join_[i], df_side_[i]);
        cudaStreamWaitEvent(cs, df_join_[i], 0);
    };
    // ---- the window's inputs, from mapped staging.  With branches the steps and positions (first read by the first
    // layer's mixer) cross PCIe on side 1 beside the embedding and the first hyper-connection read.
    copy_i32_from_mapped(tok_, m_tok_, T, cs);
    bool inputs_pending = false;
    cudaStream_t in_s = cs;
    if (br) {
        fork(1);
        in_s = df_side_[1];
        inputs_pending = true;
    }
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * kStepCount, in_s);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) MT * (NH + NKV + IQ), in_s);
    // per-ROW positions of the K rows [t][NKV] and the indexer query rows [t][IQ] (for batched RoPE)
    const int32_t* pos_k = pos_ + MT * NH;
    const int32_t* pos_i = pos_ + MT * (NH + NKV);

    // ---- the embeddings, broadcast to the hc streams - or, in a later stage of a layer split, the previous stage's
    // residual, pending write and inject (see set_stage)
    if (const NativeEmbed* ne = native_embed()) {       // plan v0.3 P6: the GGUF-form table
        ne->gather_dev(tok_, T, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    } else {
        const WeightRef* w = wt.find("token_embd.weight");
        if (w == nullptr || w->codebook_iq4nl || (w->code_bits != 2 && w->code_bits != 4 && w->code_bits != 8)) {
            err = "verify: token_embd.weight is missing or not an S2/S4/S8 tensor";
            return false;
        }
        const auto* codes = (const uint8_t*) w->data;
        const auto* scales = (const float*) (codes + w->codes_bytes);
        const auto* offsets = w->has_offset ? (const float*) (codes + w->codes_bytes + w->scales_bytes) : nullptr;
        const uint64_t row_codes = (uint64_t) (w->ne0 / (8 / w->code_bits));
        const uint64_t row_groups = (uint64_t) (w->ne0 / w->group_elems);
        embedding_gather_dev(codes, scales, offsets, tok_, T, w->ne0, w->code_bits, w->code_bias, w->group_elems,
                             row_codes, row_groups, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    }

    // per-layer state indices (GDN and QSA layers are numbered separately)
    std::vector<int64_t> gdn_idx((size_t) g.n_layers, -1), qsa_idx((size_t) g.n_layers, -1);
    {
        int64_t qi = 0, gi = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (is_qsa_layer(g, l)) qsa_idx[(size_t) l] = qi++;
            else gdn_idx[(size_t) l] = gi++;
        }
    }

    // ---------------------------------------------------------------- pre(l, group): up to the ring
    auto pre = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        stamp(l, 0, grp);
        const LayerView v(wt, l);
        const char* pfx[2] = {"hc_attn_", "hc_ffn_"};
        const WeightRef *wn[2], *wd[2], *wu[2], *wi[2];
        for (int h = 0; h < 2; ++h) {
            wn[h] = need(v, (std::string(pfx[h]) + "norm.weight").c_str(), err);
            wd[h] = need(v, (std::string(pfx[h]) + "down.weight").c_str(), err);
            wu[h] = need(v, (std::string(pfx[h]) + "up.weight").c_str(), err);
            wi[h] = need(v, (std::string(pfx[h]) + "inject.weight").c_str(), err);
            if (!wn[h] || !wd[h] || !wu[h] || !wi[h]) return false;
        }
        // the previous layer's FFN write, folded into this layer's first read (a control vector after it has
        // already applied it)
        bool pending = l > 0 && !cvec().covers(l - 1);
        if (l == 1 && ple_on) {
            if (grp == 0) {
                if (ar_on()) wait_flag_ge(m_flag_, 1, cs);
                copy_from_mapped(ple_, m_ple_, (int64_t) T * N, cs);
            }
            float* normalized = (float*) ((uint8_t*) ss.ple.scratch + ple_block_scratch_bytes());
            static const bool ple_batch_env = [] {
                const char* e = std::getenv("STRATA_PLE_BATCH");
                return !e || e[0] != '0';
            }();
            const bool ple_batch_kv = ple_batch_env && dec_batch && n > 1 && ss.ple.w.key_bf16 != nullptr &&
                                      ss.ple.w.value_bf16 != nullptr && ple_native_bf16_enabled() &&
                                      ple_native_postops_enabled();
            // S25 (STRATA_PLE_BATCH=1): the key and value projections depend only on each row's n-gram embedding, so
            // the group's rows run through the multi-row forms of the same kernels (the weights read once); every
            // row's arithmetic is the single-row call's. The history-dependent rest stays row by row.
            static const bool ple_batch = [] { const char* e = std::getenv("STRATA_PLE_BATCH"); return e && e[0] == '1'; }();
            PleWeights pw = ss.ple.w;
            const bool nkey = pw.key_native_data != nullptr && pw.key_bf16 == nullptr;
            const bool batched = !ple_batch_kv && ple_batch && n > 1 && (pw.key_bf16 != nullptr || nkey) && ple_native_bf16_enabled();
            if (ple_batch_kv) {
                try {
                    bf16_gemv_fp32_mmvf_multi(ple_ + (size_t) tb * N, N, ss.ple.w.key_bf16,
                                              xn_ + (size_t) tb * HC * N, HC * N, N, HC * N, n, cs);
                    bf16_gemv_fp32_mmvf_multi(ple_ + (size_t) tb * N, N, ss.ple.w.value_bf16,
                                              z_ + (size_t) tb * N, N, N, N, n, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE multi: ") + e.what();
                    return false;
                }
            } else if (batched) {
                try {
                    if (pw.key_bf16 != nullptr)
                        bf16_gemv_fp32_mmvf_multi(ple_ + tb * N, N, pw.key_bf16, ple_key_ + (size_t) tb * NG_HC_DIM,
                                                  NG_HC_DIM, N, NG_HC_DIM, n, cs);
                    else {
                        native_quantize_q8_1(ple_ + tb * N, xq_, (int) N, n, cs);
                        native_mmvq(pw.key_native_type, pw.key_native_data, xq_, ple_key_ + (size_t) tb * NG_HC_DIM,
                                    (int) N, NG_HC_DIM, n, cs);
                    }
                    bf16_gemv_fp32_mmvf_multi(ple_ + tb * N, N, pw.value_bf16, ple_val_ + tb * N, N, N, N, n, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE batch: ") + e.what();
                    return false;
                }
            }
            // STRATA_DF_PLE (opt-in: STRATA_DF_PLE=1): the window's post-projection steps in one pass, bitwise the per-token loop
            // below (the norms divide at run time as its norms do; the history advances token by token), which also
            // writes the commit's history snapshots.  Scratch: layer 0's expert rows, combined by now, and the
            // hyper-connection read's `lo_`, which the read below rewrites.
            static const bool ple_post_env = [] {
                const char* e = std::getenv("STRATA_DF_PLE");
                return e != nullptr && std::atoi(e) != 0;
            }();
            const bool ple_post = ple_batch_kv && ple_post_env;
            // #783 PR-g (stuchapin909): the window's residual writes in one launch (STRATA_NO_MULTI_GR=1: per token)
            const bool multi_write = !ple_post && dec_batch && !g_no_multi_gr && n > 1;
            if (multi_write) gr_write_multi(Rt(tb), bo_ + tb * N, inj2_ + tb * HC, gs, Rt(tb), n, cs);
            if (ple_post) {
                float* qn = parts_ + (size_t) tb * K * N;           // n x 10240 (K * N = 25600 per row)
                float* gated = hit_out_ + (size_t) tb * K * N;      // n x 10240
                float* pgate = lo_ + (size_t) tb * g.hc_lr;         // n x 4
                try {
                    if (gr_native_mmvf_enabled())   // gr_write's native write for the n rows in one launch
                        native_gr_post_multi(Rt(tb), bo_ + tb * N, inj2_ + tb * HC, Rt(tb), (int) N, (int) HC, n, HC * N,
                                             N, HC, cs);
                    else
                        for (int t = tb; t < te; ++t) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
                    native_ple_postops_batch_snap(xn_ + (size_t) tb * HC * N, Rt(tb), z_ + (size_t) tb * N, ss.ple.hist,
                                                  ss.ple.w, qn, gated, pgate, n, hist_snap_ + (size_t) tb * HS, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE (window): ") + e.what();
                    return false;
                }
            } else
            for (int t = tb; t < te; ++t) {
                if (batched) { pw.pre_key = ple_key_ + (size_t) t * NG_HC_DIM; pw.pre_value = ple_val_ + t * N; }
                if (!multi_write) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
                PleOut po;
                po.normalized = normalized;
                po.result = Rt(t);
                float* hist = ss.ple.hist;
                try {
                    if (ple_batch_kv) {
                        ple_block_projected(xn_ + (size_t) t * HC * N, z_ + (size_t) t * N, Rt(t), hist,
                                            ss.ple.w, po, ss.ple.scratch, cs);
                    } else {
                        ple_block(ple_ + t * N, Rt(t), hist, pw, po, ss.ple.scratch, cs);
                    }
                    ple_history_advance(hist, normalized, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE: ") + e.what();
                    return false;
                }
                copy_from_mapped(hist_snap_ + (size_t) t * HS, hist, HS, cs);
            }
            pending = false;
        }
        auto gr_read_group = [&](int half, bool apply, float* inj_prev, float* inj_out) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = tb; t < te; ++t) {
                FusedGrArgs& a = fa[t - tb];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = apply;
                a.bo_prev = bo_ + t * N; a.inj_prev = inj_prev + t * HC;
                a.w_norm = (const float*) wn[half]->data; a.w_down = (const uint16_t*) wd[half]->data;
                a.w_up = (const uint16_t*) wu[half]->data; a.w_inject = (const uint16_t*) wi[half]->data;
                a.q8_down = (const uint8_t*) wd[half]->hc_q8; a.q8_up = (const uint8_t*) wu[half]->hc_q8;
                a.q8_inject = (const uint8_t*) wi[half]->hc_q8;
                a.s_down = wd[half]->q8_scale; a.s_up = wu[half]->q8_scale;
                a.eps = EPS; a.lo = lo_ + t * g.hc_lr; a.rs = rs_ + t * HC;
                a.inject_out = inj_out + t * HC; a.mixed = mixed_ + t * N;
                if (qcnt_ != nullptr) {   // S26 STRATA_QFUSE: the consumer's q8_1 image written by the read itself
                    a.q8_cnt = qcnt_;
                    if (half == 0) a.q8_mixed = xq_ + (size_t) (t - tb) * (N / 32) * 36;
                    else if (strata::kernels::cpu::expert_layout().native) a.q8_mixed = nat_xq_ + (size_t) t * (N / 32) * 36;
                }
            }
            return fused_gr_read_multi(fa, n, xn_ + (size_t) tb * HC * N, cs, (prof_on_ && grp == 0) ? prof_ : nullptr,
                                (int) (l * kProfPer + (half == 0 ? 27 : 30)));
        };
        const bool q8_attn = gr_read_group(0, pending, inj2_, inj_);   // true: xq_ holds mixed's q8_1 (STRATA_QFUSE)
        {   // measurement only: STRATA_DUMMY_NODES=k adds k empty kernel nodes per layer (what one node costs in a window)
            static const int dummy = [] { const char* e = std::getenv("STRATA_DUMMY_NODES"); return e != nullptr ? std::atoi(e) : 0; }();
            for (int d = 0; d < dummy; ++d) gpu_noop(cs);
        }
        if (inputs_pending) {   // the window's steps and positions, before the first mixer reads them
            join(1);
            inputs_pending = false;
        }
        stamp(l, 1, grp);
        float* xm = mixed_ + tb * N;
        bool il_ready = false;   // xil_ holds xq_'s interleaved copy (reset whenever xq_ is rewritten)
        // a projection input's q8_1 blocks and their interleaved copy in one launch (native_quantize_q8_1_il, the same
        // bytes): Turing default (+0.5..+1 % decode, two launches fewer per GDN layer and per QSA layer); STRATA_QIL=0 / 1
        // forces it off / on
        auto quant_in = [&](const float* x, int n_in) {
            static const int qil_env = [] { const char* v = std::getenv("STRATA_QIL"); return v != nullptr ? std::atoi(v) : -1; }();
            const bool qil = qil_env > 0;
            if (qil && n >= 2 && n <= 4 && g_mmvq_il()) {
                strata::kernels::native_quantize_q8_1_il(x, xq_, xil_, n_in, n, cs);
                il_ready = true;
                return;
            }
            native_quantize_q8_1(x, xq_, n_in, n, cs);
            il_ready = false;
        };
        auto mm = [&](const WeightRef* w, float* out, int n_in, int n_out) {
            if (g_mmvq_il() && strata::kernels::native_mmvq_il_supported(w->native_type, n, n_out)) {
                if (!il_ready && n >= 2) {   // one column reads the plain q8_1 blocks
                    strata::kernels::native_q8_1_interleave(xq_, xil_, n_in, n, cs);
                    il_ready = true;
                }
                strata::kernels::native_mmvq_il(w->native_type, w->native_data, xq_, xil_, out, n_in, n_out, n, cs);
            } else {
                native_mmvq(w->native_type, w->native_data, xq_, out, n_in, n_out, n, cs);
            }
        };
        try {
            if (!is_qsa_layer(g, l)) {
                // ======================= GDN =======================
                const WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                *wout = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                *wsa = need(v, "ssm_a", err);
                if (!wqkv || !wg || !wout || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                if (!native_of(wqkv, v.name("attn_qkv.weight"), err) || !native_of(wg, v.name("attn_gate.weight"), err) ||
                    !native_of(wout, v.name("ssm_out.weight"), err))
                    return false;
                const int64_t gi = gdn_idx[(size_t) l];
                float* state = ss.gdn_state + (size_t) (gi - ss.gdn_ord0) * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                float* qkv = qkv_L_ + (size_t) gi * MT * C;
                float* hb = h_L_ + (size_t) gi * MT * C;
                float* gate = gate_L_ + (size_t) gi * MT * HV;
                float* beta = beta_L_ + (size_t) gi * MT * HV;
                if (sidem) {   // the gates (a few blocks, latency-bound) beside the projections, joined below
                    if (!branch(mfork_, cs, side_, "fork")) return false;
                    gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                                 (const float*) wsa->data, gate + (size_t) tb * HV, beta + (size_t) tb * HV, (int) N,
                                 (int) HV, n, side_);
                    if (cudaEventRecord(mjoin_, side_) != cudaSuccess) { err = "verify: the GDN gates' branch could not join"; return false; }
                }
                if (!q8_attn) quant_in(xm, (int) N);
                if (side_proj) {   // the gate projection (compute-bound for i-quants) beside qkv and the conv
                    // the branch reads F4's interleaved copy too (written on cs before the fork; bitwise native_mmvq's
                    // output), and qkv is recorded before the branch's launch, so its grid is dispatched first and the
                    // conv / recurrence that wait for it start sooner (z is read only after the recurrence)
                    const bool g_il = g_mmvq_il() && strata::kernels::native_mmvq_il_supported(wg->native_type, n, (int) ZV);
                    if (g_il && !il_ready && n >= 2) {
                        strata::kernels::native_q8_1_interleave(xq_, xil_, (int) N, n, cs);
                        il_ready = true;
                    }
                    if (!branch(pfork_, cs, side2_, "fork")) return false;
                    mm(wqkv, qkv + (size_t) tb * C, (int) N, (int) C);
                    if (g_il) strata::kernels::native_mmvq_il(wg->native_type, wg->native_data, xq_, xil_, z_ + (size_t) tb * ZV,
                                                              (int) N, (int) ZV, n, side2_);
                    else native_mmvq(wg->native_type, wg->native_data, xq_, z_ + (size_t) tb * ZV, (int) N, (int) ZV, n, side2_);
                    if (cudaEventRecord(pjoin_, side2_) != cudaSuccess) { err = "verify: the gate projection's branch could not join"; return false; }
                }
                if (br) {   // STRATA_DF_BRANCH: a/b and z beside q/k/v and the conv; all of them read only this layer's input
                    fork(0);
                    gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                                 (const float*) wsa->data, gate + (size_t) tb * HV, beta + (size_t) tb * HV, (int) N,
                                 (int) HV, n, df_side_[0]);
                    fork(1);
                    native_mmvq(wg->native_type, wg->native_data, xq_, z_ + (size_t) tb * ZV, (int) N, (int) ZV, n,
                                df_side_[1]);
                }
                const bool fuse_ab = fuse_convab_on() && !br && !prof_on_ && !sidem;   // the side mixer computes the gates on its branch
                if (!side_proj) mm(wqkv, qkv + (size_t) tb * C, (int) N, (int) C);   // with side_proj: above, first
                stamp(l, 2, grp);
                // a one-token window keeps its token: it advances the conv history and the state itself (commit)
                if (fuse_ab)   // 143-amdk: the conv and the a/b projections in one launch
                    gdn_conv_ab_multi(conv, qkv, (const float*) wc->data, hb, (int) C, (int) (2 * HK), EPS, n, tb, self_commit,
                                      xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                                      (const float*) wsa->data, gate + (size_t) tb * HV, beta + (size_t) tb * HV, (int) N,
                                      (int) HV, cs);
                else
                gdn_conv_l2_multi(conv, qkv, (const float*) wc->data, hb, (int) C, (int) (2 * HK), EPS, n, cs, tb,
                                  self_commit);
                stamp(l, 3, grp);
                if (br) {   // gate, beta and z, before the recurrence reads them
                    join(0);
                    join(1);
                } else {
                if (!fuse_ab && !sidem)   // the gates: the HIP fused conv kernel (143-amdk) or the side mixer's branch, else here
                gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                             (const float*) wsa->data, gate + (size_t) tb * HV, beta + (size_t) tb * HV, (int) N, (int) HV,
                             n, cs);
                stamp(l, 4, grp);
                if (!side_proj) mm(wg, z_ + (size_t) tb * ZV, (int) N, (int) ZV);
                stamp(l, 5, grp);
                if (side_proj && cudaStreamWaitEvent(cs, pjoin_, 0) != cudaSuccess) {
                    err = "verify: the gate projection's branch could not join";
                    return false;
                }
                if (sidem && cudaStreamWaitEvent(cs, mjoin_, 0) != cudaSuccess) {
                    err = "verify: the GDN gates' branch could not join";
                    return false;
                }
                }
                // the recurrence from the untouched state over tokens [0, te); outputs only for this group's
                gdn_step_norm_multi(state, hb, (int) C, gate, beta, z_, (const float*) wnm->data, EPS, y_, (int) HK,
                                    (int) HV, te, self_commit ? one_ : nullptr, cs, tb, g_qfuse() ? (void*) xq_ : nullptr);
                stamp(l, 6, grp);
                // STRATA_QFUSE: done above by gdn_step_norm_multi
                if (!g_qfuse()) quant_in(y_ + (size_t) tb * ZV, (int) ZV);
                else il_ready = false;
                mm(wout, bo_ + tb * N, (int) ZV, (int) N);
            } else {
                // ======================= QSA =======================
                const int64_t qi = qsa_idx[(size_t) l];
                const QsaState& st = ss.qsa_states[qi];
                const WeightRef *wik = need(v, "indexer.k_proj.weight", err), *wq = need(v, "attn_q.weight", err),
                                *wk = need(v, "attn_k.weight", err), *wv = need(v, "attn_v.weight", err),
                                *wo = need(v, "attn_output.weight", err), *wiq = need(v, "indexer.q_proj.weight", err),
                                *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                *wiqn = need(v, "indexer.q_norm.weight", err), *wikn = need(v, "indexer.k_norm.weight", err);
                if (!wik || !wq || !wk || !wv || !wo || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                if (!native_of(wq, v.name("attn_q.weight"), err) || !native_of(wk, v.name("attn_k.weight"), err) ||
                    !native_of(wv, v.name("attn_v.weight"), err) || !native_of(wo, v.name("attn_output.weight"), err))
                    return false;
                auto norm_rope_on = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos,
                                        cudaStream_t sx) {
                    if (native_qsa_enabled()) native_qsa_rms_norm_weighted(data, (const float*) norm->data, data, cols, rows, EPS, sx);
                    else rms_norm_weighted(data, (const float*) norm->data, rows, cols, EPS, sx);
                    if (native_rope_enabled()) native_rope_apply(data, data, rows, cols, (int) s.n_rot, rope_scaling(), pos, sx);
                    else rope_neox_apply(data, data, rows, cols, (int) s.n_rot, st.cos_tab, st.sin_tab, pos, sx);
                };
                // #783 PR-f (stuchapin909): the per-head RMSNorm and the rope in one launch, bit-identical to the pair
                // (rope_parity check 6); STRATA_NO_NORM_ROPE=1 keeps the two; off on HIP until its parity check passes
                const bool fuse_nr = native_qsa_enabled() && native_rope_enabled() && native_norm_rope_usable((int) HD, (int) s.n_rot);
                auto norm_rope = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos) {
                    if (fuse_nr && native_norm_rope_usable(cols, (int) s.n_rot)) {
                        native_qsa_rms_norm_rope(data, cols, (const float*) norm->data, data, rows, cols, (int) s.n_rot, EPS,
                                                 rope_scaling(), pos, cs);
                        return;
                    }
                    norm_rope_on(data, norm, rows, cols, pos, cs);
                };
                float* idx_raw = idx_raw_L_ + (size_t) qi * MT * ID;
                // the per-token GEMVs / norms / RoPEs / copies of this layer as one launch over the
                // window's rows each - row-wise identical arithmetic (STRATA_DEC_BATCH=0: token by token)
                // STRATA_DF_QB_Q4 (default on since 0.1.45; STRATA_DF_QB_Q4=0 gives the old token-by-token path): a Q4_0 KV cache
                // takes the batched window too - its one per-token extra, the query's rotation, is row-wise and the batched
                // path rotates (270650e).  The generated ids are identical (p620 gfx1201, A4000 sm_86, P100 sm_60, short and
                // long prompts), and the 6-token verify graph is 1950 nodes instead of 2610 (#2403: an RX 6900 XT faulted in
                // the first verify graph launch with --kv q4_0 + KV streaming + MTP; whether the graph size is the cause is
                // not confirmed)
                static const bool qb_q4 = [] {
                    const char* e = std::getenv("STRATA_DF_QB_Q4");
                    return e == nullptr || std::atoi(e) != 0;
                }();
                const bool qb = dec_batch && n > 1 && native_qsa_enabled() && native_rope_enabled() && (!st.kv_q4 || qb_q4);
                if (!q8_attn) quant_in(xm, (int) N);
                // STRATA_DF_BRANCH: the query side (q and its norm/RoPE/rotation on side 0, the indexer query on side 1)
                // beside the K/V side; the indexer query joins before the block scores, the query before attention
                const bool qbr = br && qb;
                if (qbr) {
                    fork(0);
                    native_mmvq(wq->native_type, wq->native_data, xq_, qfull_ + tb * NH * 2 * HD, (int) N,
                                (int) (NH * 2 * HD), n, df_side_[0]);
                    copy_rows_strided(qcur_ + tb * NH * HD, qfull_ + tb * NH * 2 * HD, (int64_t) n * NH, HD, 2 * HD,
                                      df_side_[0]);
                    norm_rope_on(qcur_ + tb * NH * HD, wqn, (int) (n * NH), (int) HD, pos_ + tb * NH, df_side_[0]);
                    if (st.kv_rot) fwht256_inplace_cuda(qcur_ + tb * NH * HD, (int64_t) n * NH, df_side_[0]);   // <Hq, Hk> = <q, k>
                    fork(1);
                    bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wiq->data, qidx_ + tb * IQ * ID,
                                              IQ * ID, N, IQ * ID, n, df_side_[1]);
                    norm_rope_on(qidx_ + tb * IQ * ID, wiqn, (int) (n * IQ), (int) ID, pos_i + tb * IQ, df_side_[1]);
                }
                // STRATA_SIDE_MIXER=1 / =2 (fork Eddoursul 34a35db + b063c13): the indexer's key projection,
                // the tail's snapshot, its keys appended, its query projection, norm and RoPE, the block scores and the
                // top-k read only the mixer's input and the indexer's own state - =1 runs them on a branch beside the
                // attention's projections; =2 also the queries' chain on a second branch.  Each step below is
                // the in-line code with its stream as a parameter; with the switch off they run on cs in the same order
                // as before (STRATA_DF_BRANCH, the same idea for fewer of these steps, is off whenever this is on).
                auto norm_rope_s = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos, cudaStream_t ns) {
                    if (fuse_nr && native_norm_rope_usable(cols, (int) s.n_rot)) {
                        native_qsa_rms_norm_rope(data, cols, (const float*) norm->data, data, rows, cols, (int) s.n_rot, EPS,
                                                 rope_scaling(), pos, ns);
                        return;
                    }
                    norm_rope_on(data, norm, rows, cols, pos, ns);
                };
                auto idx_key = [&](cudaStream_t ns) {
                    if (qb) bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wik->data, idx_raw + tb * ID, ID, N, ID, n, ns);
                    else for (int t = tb; t < te; ++t)
                        bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wik->data, idx_raw + t * ID, (int) N, (int) ID, ns);
                };
                auto idx_snap = [&](cudaStream_t ns) {
                    if (grp == 0) copy_from_mapped(tail_snap_ + (size_t) qi * TS, st.idx_tail, TS, ns);
                };
                auto idx_append = [&](cudaStream_t ns) {
                    if (dec_batch && !g_no_batch_kv) {
                        const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                        native_qsa_indexer_append_steps(idx_raw + tb * ID, step_ + tb * kStepCount + kStepPos, kStepCount,
                                                        n, 0, (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                        rope_scaling(), ns);
                    } else {
                        for (int t = tb; t < te; ++t) {
                            const QsaState& sx = ss.qsa_states[qi];
                            const QsaIndexerBuffers ib{sx.idx_tail, sx.idx_dead, sx.idx_pooled, sx.idx_block_pos};
                            native_qsa_indexer_append(idx_raw + t * ID, step_ + t * kStepCount + kStepPos, 0,
                                                      (const float*) wikn->data, EPS, ib, s, sx.max_cells,
                                                      rope_scaling(), ns);
                        }
                    }
                };
                auto idx_query = [&](cudaStream_t ns) {
                    if (qb) {
                        bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wiq->data, qidx_ + tb * IQ * ID, IQ * ID,
                                                  N, IQ * ID, n, ns);
                        norm_rope_s(qidx_ + tb * IQ * ID, wiqn, (int) (n * IQ), (int) ID, pos_i + tb * IQ, ns);
                    } else
                        for (int t = tb; t < te; ++t) {
                            float* qx = qidx_ + t * IQ * ID;
                            bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wiq->data, qx, (int) N, (int) (IQ * ID), ns);
                            norm_rope_s(qx, wiqn, (int) IQ, (int) ID, pos_ + t * NH, ns);
                        }
                };
                auto idx_select = [&](cudaStream_t ns) {
                    qsa_block_scores(st.idx_pooled, st.idx_dead, qidx_ + tb * IQ * ID, step_ + tb * kStepCount, n, max_blocks_,
                                     s, scores_ + (size_t) tb * max_blocks_, ns);
                    qsa_block_topk(scores_ + (size_t) tb * max_blocks_, step_ + tb * kStepCount, n, max_blocks_, cap_, s,
                                   sel_ + (size_t) tb * cap_, ns);
                };
                // the attention's queries: the projection (mm on cs, which may read F4's interleaved copy; on a branch
                // F4's kernel too, its copy written on cs before the fork - bitwise the same), the q/gate
                // split with norm and RoPE, the rotation
                const bool q_il = side_proj && g_mmvq_il() &&
                                  strata::kernels::native_mmvq_il_supported(wq->native_type, n, (int) (NH * 2 * HD));
                auto q_chain = [&](cudaStream_t ns) -> bool {
                    if (ns == cs) mm(wq, qfull_ + tb * NH * 2 * HD, (int) N, (int) (NH * 2 * HD));
                    else if (q_il) strata::kernels::native_mmvq_il(wq->native_type, wq->native_data, xq_, xil_,
                                                                   qfull_ + tb * NH * 2 * HD, (int) N, (int) (NH * 2 * HD), n, ns);
                    else native_mmvq(wq->native_type, wq->native_data, xq_, qfull_ + tb * NH * 2 * HD, (int) N,
                                     (int) (NH * 2 * HD), n, ns);
                    if (qb) {
                        if (fuse_nr) {   // the q/gate split reads q straight out of the q|gate rows (stride 2 * HD)
                            native_qsa_rms_norm_rope(qfull_ + tb * NH * 2 * HD, (int) (2 * HD), (const float*) wqn->data,
                                                     qcur_ + tb * NH * HD, (int) (n * NH), (int) HD, (int) s.n_rot, EPS,
                                                     rope_scaling(), pos_ + tb * NH, ns);
                        } else {
                            copy_rows_strided(qcur_ + tb * NH * HD, qfull_ + tb * NH * 2 * HD, (int64_t) n * NH, HD, 2 * HD, ns);
                            norm_rope_s(qcur_ + tb * NH * HD, wqn, (int) (n * NH), (int) HD, pos_ + tb * NH, ns);
                        }
                        if (st.kv_rot) fwht256_inplace_cuda(qcur_ + tb * NH * HD, (int64_t) n * NH, ns);   // <Hq, Hk> = <q, k>
                    } else {
                        for (int t = tb; t < te; ++t) {
                            float* qc = qcur_ + t * NH * HD;
                            if (fuse_nr) {
                                native_qsa_rms_norm_rope(qfull_ + t * NH * 2 * HD, (int) (2 * HD), (const float*) wqn->data, qc,
                                                         (int) NH, (int) HD, (int) s.n_rot, EPS, rope_scaling(), pos_ + t * NH, ns);
                            } else {
                                if (cudaMemcpy2DAsync(qc, (size_t) HD * 4, qfull_ + t * NH * 2 * HD, (size_t) HD * 2 * 4,
                                                      (size_t) HD * 4, (size_t) NH, cudaMemcpyDeviceToDevice, ns) != cudaSuccess) {
                                    err = "verify: the q/gate split failed";
                                    return false;
                                }
                                norm_rope_s(qc, wqn, (int) NH, (int) HD, pos_ + t * NH, ns);
                            }
                            if (st.kv_rot) fwht256_inplace_cuda(qc, NH, ns);   // <Hq, Hk> = <q, k>
                        }
                    }
                    return true;
                };
                if (sidem) {
                    if (!branch(mfork_, cs, side_, "fork")) return false;
                    idx_key(side_);
                    idx_snap(side_);
                    idx_append(side_);
                    idx_query(side_);
                    idx_select(side_);
                    if (cudaEventRecord(mjoin_, side_) != cudaSuccess) { err = "verify: the indexer's branch could not join"; return false; }
                }
                if (side_proj) {   // the queries beside the keys and values (fork b063c13)
                    if (q_il && !il_ready && n >= 2) {   // the copy before the fork (mm(wk) / mm(wv) on cs reuse it)
                        strata::kernels::native_q8_1_interleave(xq_, xil_, (int) N, n, cs);
                        il_ready = true;
                    }
                    if (!branch(pfork_, cs, side2_, "fork") || !q_chain(side2_)) return false;
                    if (cudaEventRecord(pjoin_, side2_) != cudaSuccess) { err = "verify: the queries' branch could not join"; return false; }
                }
                if (!sidem) idx_key(cs);
                stamp(l, 7, grp);
                mm(wk, kcur_ + tb * NKV * HD, (int) N, (int) (NKV * HD));
                mm(wv, vcur_ + tb * NKV * HD, (int) N, (int) (NKV * HD));
                if (qb) norm_rope(kcur_ + tb * NKV * HD, wkn, (int) (n * NKV), (int) HD, pos_k + tb * NKV);
                else for (int t = tb; t < te; ++t) norm_rope(kcur_ + t * NKV * HD, wkn, (int) NKV, (int) HD, pos_ + t * NH);
                if (st.kv_rot) {   // K and V rotated before they are stored (kv_q4.hpp)
                    fwht256_inplace_cuda(kcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                    fwht256_inplace_cuda(vcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                } else if (st.kv_hybrid) {   // K8V4: only V is rotated
                    fwht256_inplace_cuda(vcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                }
                stamp(l, 8, grp);
                if (!sidem) idx_snap(cs);
                // #783 PR-d (stuchapin909): the window's K/V cells in one launch per pool instead of one per token
                if (dec_batch && !g_no_batch_kv) {
                    const int32_t* step_b = step_ + tb * kStepCount;
                    const float* kc_b = kcur_ + tb * NKV * HD;
                    const float* vc_b = vcur_ + tb * NKV * HD;
                    if (st.kv_hybrid) {   // K8V4: the unused half's lanes folded onto the used pool (layer.cpp)
                        // #1188: a streamed state's host copy gets the window's cells too, as the per-token path below
                        // writes them - without it a block that was not resident when its cells were appended came
                        // back from the host copy without them, and free generation degenerated into repetition
                        const KvHostPools hk = kv_hybrid_k_half(st.host), hv = kv_hybrid_v_half(st.host);
                        const bool mirror = st.host.present();
                        kv_append_q8_steps(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, step_b, kStepCount,
                                           kc_b, kc_b, (int) (NKV * HD), n, s, cs, mirror ? &hk : nullptr);
                        kv_append_q4_steps(st.v_q4, st.v_q4, st.page_table, step_b, kStepCount, n, vc_b, vc_b, s, cs,
                                           mirror ? &hv : nullptr);
                    } else if (st.kv_q4)
                        kv_append_q4_steps(st.k_q4, st.v_q4, st.page_table, step_b, kStepCount, n, kc_b, vc_b, s, cs,
                                           &st.host);
                    else if (st.kv_int8)
                        kv_append_q8_steps(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step_b, kStepCount,
                                           kc_b, vc_b, (int) (NKV * HD), n, s, cs, &st.host);
                    else
                        for (int t = tb; t < te; ++t)
                            kv_append_step(st.k_pool, st.v_pool, st.page_table, step_ + t * kStepCount,
                                           kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st.host);
                } else
                for (int t = tb; t < te; ++t) {
                    const QsaState& st = ss.qsa_states[qi];
                    const int32_t* step_t = step_ + t * kStepCount;
                    if (st.kv_hybrid) {   // K8V4: the unused half's lanes folded onto the used pool (layer.cpp)
                        const KvHostPools hk = kv_hybrid_k_half(st.host), hv = kv_hybrid_v_half(st.host);
                        const bool mirror = st.host.present();   // streamed: the host copy too
                        kv_append_q8_step(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, step_t,
                                          kcur_ + t * NKV * HD, kcur_ + t * NKV * HD, s, cs, mirror ? &hk : nullptr);
                        kv_append_q4_step(st.v_q4, st.v_q4, st.page_table, step_t, vcur_ + t * NKV * HD,
                                          vcur_ + t * NKV * HD, s, cs, mirror ? &hv : nullptr);
                    } else if (st.kv_q4)
                        kv_append_q4_step(st.k_q4, st.v_q4, st.page_table, step_t, kcur_ + t * NKV * HD,
                                          vcur_ + t * NKV * HD, s, cs, &st.host);
                    else if (st.kv_int8)
                        kv_append_q8_step(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step_t,
                                          kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st.host);
                    else
                        kv_append_step(st.k_pool, st.v_pool, st.page_table, step_t, kcur_ + t * NKV * HD,
                                       vcur_ + t * NKV * HD, s, cs, &st.host);
                }
                if (!sidem) idx_append(cs);
                stamp(l, 9, grp);
                if (qbr) {
                    join(1);   // the indexer query, for the block scores
                } else {
                    if (!side_proj && !q_chain(cs)) return false;
                    if (!sidem) idx_query(cs);
                }
                stamp(l, 10, grp);
                if (!sidem) idx_select(cs);
                else if (cudaStreamWaitEvent(cs, mjoin_, 0) != cudaSuccess) {
                    err = "verify: the indexer's branch could not join";
                    return false;
                }
                if (side_proj && cudaStreamWaitEvent(cs, pjoin_, 0) != cudaSuccess) {
                    err = "verify: the queries' branch could not join";
                    return false;
                }
                stamp(l, 11, grp);
                // KV streaming: the n selections' blocks resident (device-side, inside the graph)
                qsa_kv_resolve(st, *g_, sel_ + (size_t) tb * cap_, step_ + tb * kStepCount, n, cap_, cs);
                stamp(l, 12, grp);
                if (qbr) join(0);   // the query, for attention
                const QsaAttnPools pools = qsa_attn_pools(st);
                // STRATA_QSA_SPLIT_ATTN=1: the split tensor-core attention where it applies (int8 KV, sm_80+)
                static const bool split_attn = [] { const char* v = std::getenv("STRATA_QSA_SPLIT_ATTN"); return v != nullptr && std::atoi(v) != 0; }();
                if (!(split_attn && qsa_decode_attn_split(qcur_ + tb * NH * HD, pools, sel_ + (size_t) tb * cap_,
                                                          step_ + tb * kStepCount, cap_, s,
                                                          attn_scratch_ + (size_t) tb * attn_scratch_floats_,
                                                          (uint64_t) n * (uint64_t) attn_scratch_floats_, attn_ + tb * NH * HD, n, cs)))
                qsa_decode_attn_batch(qcur_ + tb * NH * HD, pools, sel_ + (size_t) tb * cap_, step_ + tb * kStepCount, cap_,
                                      s, attn_scratch_ + (size_t) tb * attn_scratch_floats_, attn_ + tb * NH * HD, n, cs);
                stamp(l, 13, grp);
                if (st.kv_rot || st.kv_hybrid) fwht256_inplace_cuda(attn_ + tb * NH * HD, (int64_t) n * NH, cs);   // back: H^-1 = H
                if (qb) native_qsa_gate_apply(attn_ + tb * NH * HD, qfull_ + tb * NH * 2 * HD, attn32_ + tb * NH * HD,
                                              (int) (n * NH), (int) HD, cs);
                else
                for (int t = tb; t < te; ++t) {
                    if (native_qsa_enabled())
                        native_qsa_gate_apply(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, attn32_ + t * NH * HD,
                                              (int) NH, (int) HD, cs);
                    else
                        qsa_gate_apply_f32(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, s, attn32_ + t * NH * HD, cs);
                }
                stamp(l, 14, grp);
                quant_in(attn32_ + tb * NH * HD, (int) (NH * HD));
                mm(wo, bo_ + tb * N, (int) (NH * HD), (int) N);
            }
        } catch (const std::exception& e) {
            err = "verify layer " + std::to_string(l) + ": " + e.what();
            return false;
        }
        stamp(l, 16, grp);
        const bool q8_ffn = gr_read_group(1, true, inj_, inj2_);   // true: nat_xq_ holds the MoE input's q8_1 (STRATA_QFUSE)
        stamp(l, 32, grp);   // the profile only: the FFN-side read ends here, the router starts
        // with CPU experts in the window the shared expert forks off once the doorbell is published (it no longer
        // competes with it for the GPU's first microseconds); STRATA_SH_FORK_LATE=0 forks at the top as before
        static const bool sh_fork_late_env = [] {
            const char* e = std::getenv("STRATA_SH_FORK_LATE");
            return !e || e[0] != '0';
        }();
        const bool sh_fork = sh_stream_on() && !prof_on_ && sh_cs_ != nullptr && ev_fork_ != nullptr && ev_join_ != nullptr;
        const bool sh_fork_late = sh_fork && sh_fork_late_env && !ar_on();
        cudaStream_t sh_stream = sh_fork ? sh_cs_ : cs;
        if (sh_fork && !sh_fork_late) {
            cudaEventRecord(ev_fork_, cs);
            cudaStreamWaitEvent(sh_cs_, ev_fork_, 0);
        }
        // the window's rows routed in 2 launches (one router GEMV reading the weight once, one
        // top-10) instead of 2 per token; every row's arithmetic is the single-token call's (STRATA_DEC_BATCH=0: old)
        const WeightRef* w_router = v.get("ffn_gate_inp.weight");
        // S26 STRATA_LFUSE=1: the shared expert's gate row rides in the router GEMV launch, its sigmoid + row scale
        // move into the combine, gate + up share one launch (all bitwise; resident combine path, 2-8 rows only)
        const WeightRef* w_sgi = lfuse_on(n) && g_lfuse_gate() ? v.get("ffn_gate_inp_shexp.weight") : nullptr;
        bool sg_ready = false;
        if (dec_batch && n > 1 && w_router != nullptr && native_router_enabled() && NE == 512 && K == 10) {
            try {
                if (w_sgi != nullptr)
                    sg_ready = bf16_gemv_fp32_mmvf_multi_aux(mixed_ + tb * N, N, (const uint16_t*) w_router->data,
                                                             logits_ + tb * NE, NE, N, NE, n,
                                                             (const uint16_t*) w_sgi->data, sh_g_ + tb, 1, cs);
                if (!sg_ready)
                bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) w_router->data, logits_ + tb * NE, NE, N,
                                          NE, n, cs);
                native_router_top10_multi(logits_ + tb * NE, ids_ + tb * K, w_ + tb * K, n, cs);
            } catch (const std::exception& e) { err = "verify router: " + std::string(e.what()); return false; }
        } else
        for (int t = tb; t < te; ++t) {
            MoEBuffers mb = ss.moe;
            mb.logits = logits_ + t * NE; mb.ids = ids_ + t * K; mb.weights = w_ + t * K;
            if (!moe_route(wt, g, l, K, mb, mixed_ + t * N, cs, err, nullptr)) return false;
        }
        // STRATA_ROUTE_PRIOR=lambda (opt-in, changes the output): the top-10 picked again with a logit bonus for the
        // resident experts; the weights stay the router's (route_prior.cu)
        static const float route_prior = [] { const char* v = std::getenv("STRATA_ROUTE_PRIOR"); return v ? (float) std::atof(v) : 0.0f; }();
        if (route_prior > 0.0f && NE == 512 && K == 10 && hits_.d_res != nullptr)
            route_prior_top10(logits_ + tb * NE, hits_.d_res + l * g.n_expert, route_prior, ids_ + tb * K, w_ + tb * K, n, cs);
        if (route_resident_cfg().margin > 0.0f && NE == 512 && K == 10 && hits_.d_res != nullptr) {
            // STRATA_ROUTE_RESIDENT: after the router, before the plan/doorbell read ids_/w_ (opt-in, changes the output)
            try {
                native_route_resident(logits_ + tb * NE, ids_ + tb * K, w_ + tb * K, hits_.d_res + l * g.n_expert, n,
                                      route_resident_cfg().margin, route_resident_cfg().lo, route_resident_cfg().hi,
                                      rr_stats_ != nullptr ? rr_stats_ + (n <= 8 ? 4 : 0) : nullptr, cs);
            } catch (const std::exception& e) { err = "verify route-resident: " + std::string(e.what()); return false; }
        }
        if (ar_on()) {
            resident_plan(ids_ + tb * K, n * (int) K, (int) K, hits_.d_res + l * g.n_expert, (int) g.n_expert,
                          hits_.cache_base, slot_off_d_, (long long) hits_.blob,
                          plan_ + (size_t) grp * (size_t) (plan_i32_ + 16), (long long) max_t_ * K, nullptr, 0, cs, m_plan_err_);
        } else {
            if (device_plan_)   // E-6: every routed expert resident: this group's plan without the host
                resident_plan(ids_ + tb * K, n * (int) K, (int) K, hits_.d_res + l * g.n_expert, (int) g.n_expert,
                              hits_.cache_base, slot_off_d_, (long long) hits_.blob,
                              plan_ + (size_t) grp * (size_t) (plan_i32_ + 16), (long long) max_t_ * K, skip_ + grp,
                              (uint32_t) ((l - lb_) * G + grp + 1), cs);
            const int32_t* layer_res = hits_.d_res != nullptr ? (hits_.d_res + l * g.n_expert) : nullptr;
            doorbell_publish_res(xm, ids_ + tb * K, layer_res, (int) g.n_expert, (int64_t) n * N, (int64_t) n * K,
                                 m_x_ + tb * N, m_ids_ + tb * K, m_seq_, cs);
            if (sh_fork_late) {
                cudaEventRecord(ev_fork_, cs);
                cudaStreamWaitEvent(sh_cs_, ev_fork_, 0);
            }
        }
        stamp(l, 17, grp);
        bool qdedup = false;   // STRATA_VERIFY_QDEDUP took effect: the experts' q8_1 image of xm is already made
        {
            const WeightRef *wgi = need(v, "ffn_gate_inp_shexp.weight", err), *wsg = need(v, "ffn_gate_shexp.weight", err),
                            *wsu = need(v, "ffn_up_shexp.weight", err), *wsd = need(v, "ffn_down_shexp.weight", err);
            if (!wgi || !wsg || !wsu || !wsd) return false;
            if (!native_of(wsg, v.name("ffn_gate_shexp.weight"), err) || !native_of(wsu, v.name("ffn_up_shexp.weight"), err) ||
                !native_of(wsd, v.name("ffn_down_shexp.weight"), err))
                return false;
            NativeSharedWeights nsw;
            nsw.gate_type = wsg->native_type; nsw.gate_data = wsg->native_data;
            nsw.up_type = wsu->native_type; nsw.up_data = wsu->native_data;
            nsw.down_type = wsd->native_type; nsw.down_data = wsd->native_data;
            nsw.q8_1 = sh_fork ? sh_xq_ : xq_;
            // STRATA_VERIFY_QDEDUP=1 (not with the forked shared-expert stream, which quantizes into its own buffer): the
            // experts' q8_1 image of xm is made first and the shared expert's gate/up read it (quantize_q8_1_rows and
            // native_quantize_q8_1 write the same bytes)
            qdedup = g_qdedup() && !sh_fork && strata::kernels::cpu::expert_layout().native;
            if (qdedup && !q8_ffn) quantize_q8_1_rows(xm, n, N, nat_xq_ + (size_t) tb * (N / 32) * 36, cs);
            if (!shared_expert_native_bf16_enabled()) {
                if (dec_batch) f32_to_bf16_bulk(mixed_ + tb * N, sh_bf16_ + tb * N, (int64_t) n * N, sh_stream);   // contiguous rows
                else for (int t = tb; t < te; ++t) f32_to_bf16_bulk(mixed_ + t * N, sh_bf16_ + t * N, N, sh_stream);
            }
            try {
                shared_expert_multi(n, xm, sh_bf16_ + tb * N, nsw, (const uint16_t*) wgi->data, sh_gate_ + (size_t) tb * g.n_ff,
                                    sh_up_ + (size_t) tb * g.n_ff, sh_g_ + tb, shared_ + tb * N, N, g.n_ff, sh_stream,
                                    qdedup ? (const void*) (nat_xq_ + (size_t) tb * (N / 32) * 36) : nullptr,
                                    (sg_ready ? 1 : 0) | (lfuse_on(n) && g_lfuse_pair() ? 2 : 0));
                sg_gated_[grp] = sg_ready;
            } catch (const std::exception& e) {
                err = std::string("verify shared expert: ") + e.what();
                return false;
            }
            if (sh_fork) cudaEventRecord(ev_join_, sh_cs_);
        }
        if (strata::kernels::cpu::expert_layout().native) {
            if (!qdedup)
                if (!q8_ffn) quantize_q8_1_rows(xm, n, N, nat_xq_ + (size_t) tb * (N / 32) * 36, cs);
        } else
            quantize_q8_0_scaled(xm, hit_xq_ + (size_t) tb * (N / 32) * 34, hit_xs_ + (size_t) tb * (N / 32), (int64_t) n * N, cs);
        stamp(l, 18, grp);
        return true;
    };

    // ---------------------------------------------------------------- post(l, group): experts, combine
    static const bool fuse_head_gr_env = [] {
        const char* e = std::getenv("STRATA_FUSE_HEAD_GR");
        return e && e[0] == '1';
    }();
    const bool fuse_head_gr = fuse_head_gr_env && (head_ != nullptr && head_->loaded()) &&
                              !cvec().covers(g.n_layers - 1);
    auto post = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        const uint32_t ring = (uint32_t) ((l - lb_) * G + grp + 1);
        const int64_t cap = (int64_t) n * K, capx = (int64_t) max_t_ * K;
        int32_t* pl = plan_ + (size_t) grp * (size_t) (plan_i32_ + 16);
        const int32_t* p_counts = pl;
        const int32_t* p_start = pl + 4;
        const int32_t* p_dst = p_start + capx + 1;
        const int32_t* p_tok = p_dst + capx;
        const int64_t ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
        const unsigned long long* p_ptr = (const unsigned long long*) (pl + ptr_off);
        const unsigned long long* p_ptr2 = p_ptr + capx;
        const int32_t* p_start2 = pl + ptr_off + 4 * capx;
        float* hit_out = hit_out_ + (size_t) tb * K * N;
        float* parts_out = parts_ + (size_t) tb * K * N;
        const auto& lay = strata::kernels::cpu::expert_layout();
        // plan v0.3 P6: the VRAM groups now; the PCIe groups once the copy engine has landed them in staging.
        // `gy`: the native launch's groups side by side (0: cap, one block row per possible group).
        auto grouped = [&](const unsigned long long* gp, const int32_t* gs, const int32_t* gn, int64_t gy, float* dst_buf) {
            if (lay.native) {
                // the layer's GGUF formats (i-quant gate/up, Q2_0 / IQ4_NL down)
                const auto& f = lay.fmt[(size_t) l];
                const NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
                native_expert_grouped(L, gp, gs, gn, p_dst, p_tok, cap, cap,
                                      nat_xq_ + (size_t) tb * (N / 32) * 36, hit_scratch_, dst_buf, cs, gy);
            } else {
                moe_grouped_s2(gp, gs, gn, p_dst, p_tok, cap, cap, hit_xq_ + (size_t) tb * (N / 32) * 34,
                               hit_xs_ + (size_t) tb * (N / 32), hit_scratch_, dst_buf, cs);
            }
        };
        bool hits_in_combine = false;
        if (ar_on()) {
            stamp(l, 19, grp);
            grouped(p_ptr, p_start, p_counts, 0, parts_out);
            stamp(l, 20, grp);
        } else {
            if (device_plan_) {   // E-6: skipped when the device planned this group (all its experts resident)
                wait_flag_ge_or(m_flagA_, ring, skip_ + grp, cs);
                copy_i32_from_mapped_unless(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, skip_ + grp, ring, cs);
            } else {
                if (fuse_wait_on() && !prof_on_ && m_bal_ == nullptr) {   // 143-amdk: one launch for the wait and the copy
                    wait_copy_i32_from_mapped(m_flagA_, ring, pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, cs);
                } else {
                if (m_bal_ != nullptr)   // set_pcie_balance: the GPU's share starts as the plan arrives
                    wait_flag_ge_stamped(m_flagA_, ring, nullptr, m_bal_ + ((l - lb_) * 2 + grp) * 4, cs);
                else
                    wait_flag_ge(m_flagA_, ring, cs);              // the pool published this group's GPU plan
                copy_i32_from_mapped(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, cs);
                }
            }
            stamp(l, 19, grp);
            grouped(p_ptr, p_start, p_counts, 0, hit_out);
            stamp(l, 20, grp);
            if (device_plan_) wait_flag_ge_or(m_flagB_, ring, skip_ + grp, cs);
            else if (m_bal_ != nullptr)   // set_pcie_balance: the PCIe part starts
                wait_flag_ge_stamped(m_flagB_, ring, m_bal_ + ((l - lb_) * 2 + grp) * 4 + 1, nullptr, cs);
            else wait_flag_ge(m_flagB_, ring, cs);                 // the PCIe share is in staging (DMA) or mapped
            if (sink_.pcie_mode == 2) {                            // stage it with a copy kernel, then point at staging
                const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
                uint8_t* stage = staging_ + (size_t) (grp * per) * lay.max_blob;
                fetch_blobs(p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), (int) per, cs);
                rebase_ptrs((unsigned long long*) p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), cs);
            }
            stamp(l, 21, grp);
            // the PCIe share is pcie_frac of the misses: a few groups when the cache is cold, usually none (always none at
            // pcie_frac 0), so its launch is kPcieGroupRows block rows striding over the groups, not cap of them
            grouped(p_ptr2, p_start2, p_counts + 2, kPcieGroupRows, hit_out);
            stamp(l, 22, grp);
            if (device_plan_) {   // no CPU share when the device planned the group: its rows are zeros
                wait_flag_ge_or(m_flag_, ring, skip_ + grp, cs);
                copy_or_zero_from_mapped(parts_out, m_ymiss_ + (size_t) tb * K * N, (long long) n * K * N,
                                         skip_ + grp, ring, cs);
            } else {
                const bool fuse_rows = fuse_rows_on() && !prof_on_ && dec_batch && m_bal_ == nullptr;   // 143-amdk
                if (!fuse_rows) {
                    if (m_bal_ != nullptr)   // set_pcie_balance: ... and ends as it waits for the CPU's rows
                        wait_flag_ge_stamped(m_flag_, ring, m_bal_ + ((l - lb_) * 2 + grp) * 4 + 2, nullptr, cs);
                    else
                        wait_flag_ge(m_flag_, ring, cs);           // the CPU's share is in the mapped rows
                }
                stamp(l, 23, grp);
                if (fuse_rows)   // the wait folded into the copy's blocks
                    copy_rows_from_mapped_after(m_flag_, ring, parts_out, m_ymiss_ + (size_t) tb * K * N, (int64_t) n * K, N,
                                                p_dst, p_counts + 1, cs);
                else if (dec_batch)   // only the CPU rows cross PCIe (p_dst[0, counts[1]) = the GPU's own rows)
                    copy_rows_from_mapped(parts_out, m_ymiss_ + (size_t) tb * K * N, (int64_t) n * K, N,
                                          p_dst, p_counts + 1, cs);
                else
                    copy_from_mapped(parts_out, m_ymiss_ + (size_t) tb * K * N, (int64_t) n * K * N, cs);
            }
            // 143-amdk: the add of the GPU's hit rows is done by the combine that follows (one launch fewer)
            hits_in_combine = fuse_hits_on() && !prof_on_ && !sg_gated_[grp] && dec_batch && n > 1 &&
                              native_moe_combine_enabled() && n <= 8 &&
                              native_moe_combine_hits_supported(parts_ + (size_t) tb * K * N, shared_ + tb * N, bo_ + tb * N, N, K, n);
            if (!hits_in_combine) moe_hit_add(parts_out, hit_out, p_dst, p_counts + 1, cap, N, cs);
        }
        // Same condition as `sh_fork` above, which is per group and out of scope here: wait only when the fork
        // actually ran (a wait on an event never recorded is a no-op, but saying it outright reads better).
        if (sh_stream_on() && !prof_on_ && sh_cs_ != nullptr && ev_fork_ != nullptr && ev_join_ != nullptr) {
            cudaStreamWaitEvent(cs, ev_join_, 0);
        }
        if (sg_gated_[grp]) {   // STRATA_LFUSE: the shared expert's gate was not applied: the combine applies it (all rows are hits)
            try {
                native_moe_combine_multi_hits_gated(parts_out, w_ + tb * K, shared_ + tb * N, sh_g_ + tb, bo_ + tb * N, N, K, n, cs);
            } catch (const std::exception& e) { err = "verify combine: " + std::string(e.what()); return false; }
        } else
        if (hits_in_combine) {
            try {
                native_moe_combine_multi_hits_added(parts_ + (size_t) tb * K * N, hit_out, p_dst, p_counts + 1, w_ + tb * K,
                                                    shared_ + tb * N, bo_ + tb * N, N, K, n, cs);
            } catch (const std::exception& e) { err = "verify combine: " + std::string(e.what()); return false; }
        } else
        if (dec_batch && n > 1 && native_moe_combine_enabled()) {   // one launch for the window's rows
            try {
                native_moe_combine_multi(parts_ + (size_t) tb * K * N, w_ + tb * K, shared_ + tb * N, bo_ + tb * N, N, K, n, cs);
            } catch (const std::exception& e) { err = "verify combine: " + std::string(e.what()); return false; }
        } else
        for (int t = tb; t < te; ++t) {
            MoEBuffers mb = ss.moe;
            mb.weights = w_ + t * K; mb.shared = shared_ + t * N;
            if (!moe_combine_parts(g, l, K, mb, parts_ + (size_t) t * K * N, bo_ + t * N, cs, err)) return false;
        }
        stamp(l, 24, grp);
        if (l == g.n_layers - 1) {
            if (!fuse_head_gr) {
                if (dec_batch && !g_no_multi_gr) gr_write_multi(Rt(tb), bo_ + tb * N, inj2_ + tb * HC, gs, Rt(tb), n, cs);
                else for (int t = tb; t < te; ++t) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
                if (cvec().covers(l)) cvec_apply(Rt(tb), l, n, HC * N, nullptr, 0, nullptr, 0, false, cs);
            }
        } else if (cvec().covers(l)) {
            cvec_apply(Rt(tb), l, n, HC * N, bo_ + tb * N, N, inj2_ + tb * HC, HC, true, cs);
        }
        return true;
    };

    for (int grp = 0; grp < G; ++grp)
        if (!pre(lb_, grp)) return false;
    for (int64_t l = lb_; l < le_; ++l)
        for (int grp = 0; grp < G; ++grp) {
            if (!post(l, grp)) return false;
            if (l + 1 < le_ && !pre(l + 1, grp)) return false;
        }

    // ---- the head, T columns, and the argmax of each
    stamp(g.n_layers, 0, 0);
    {
        const WeightRef *hn = wt.find("output_hc_norm.weight"), *hd = wt.find("output_hc_down.weight"),
                        *hu = wt.find("output_hc_up.weight");
        if (!hn || !hd || !hu) { err = "verify: an output_hc_* weight is missing"; return false; }
        // S25 (STRATA_HC_Q8=1): the final mixer from the GGUF's Q8_0 projections too, through the same multi read
        // (no pending write, no inject)
        const bool mix_q8 = hd->hc_q8 != nullptr && hu->hc_q8 != nullptr && head_ != nullptr && head_->loaded();
        if (mix_q8) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                FusedGrArgs& a = fa[t];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = false;
                a.w_norm = (const float*) hn->data; a.w_down = (const uint16_t*) hd->data;
                a.w_up = (const uint16_t*) hu->data; a.w_inject = nullptr;
                a.q8_down = (const uint8_t*) hd->hc_q8; a.q8_up = (const uint8_t*) hu->hc_q8;
                a.eps = EPS; a.lo = lo_ + t * g.hc_lr; a.rs = rs_ + t * HC;
                a.mixed = head_mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        }
        else if (fuse_head_gr) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                FusedGrArgs& a = fa[t];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = true;
                a.bo_prev = bo_ + t * N; a.inj_prev = inj2_ + t * HC;
                a.w_norm = (const float*) hn->data;
                a.w_down = (const uint16_t*) hd->data;
                a.w_up = (const uint16_t*) hu->data;
                a.w_inject = nullptr;
                a.eps = EPS;
                a.lo = lo_ + t * g.hc_lr;
                a.rs = rs_ + t * HC;
                a.inject_out = head_inj_;
                a.mixed = head_mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        } else if (head_mix_multi_enabled() && head_ != nullptr && head_->loaded()) {
            // the final mixer, the window's tokens in one read (fused_gr_read_multi without the pending write: the
            // last layer's write was done above; its sums are gr_read's)
            if (hn->kind != WeightKind::F32 || hd->kind != WeightKind::Bf16InF32 || hu->kind != WeightKind::Bf16InF32) {
                err = "verify: the output_hc_* weights have the wrong engine forms";
                return false;
            }
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = Rt(t); fa[t].R_out = Rt(t); fa[t].apply = false;
                fa[t].w_norm = (const float*) hn->data; fa[t].w_down = (const uint16_t*) hd->data;
                fa[t].w_up = (const uint16_t*) hu->data; fa[t].eps = EPS;
                fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC; fa[t].mixed = head_mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        } else {
            for (int t = 0; t < T; ++t) {
                BlockBuffers bb = ss.block;
                bb.R = Rt(t);
                bb.mixed = head_mixed_ + t * N;
                if (head_ != nullptr && head_->loaded()) {
                    if (!lm_head_mix(wt, g, bb, cs, err)) return false;
                } else if (!lm_head(wt, g, bb, head_logits_ + (size_t) t * n_vocab_, cs, err)) {
                    return false;
                }
            }
        }
        if (head_ != nullptr && head_->loaded()) {
            try {
                native_quantize_q8_1(head_mixed_, xq_, (int) N, T, cs);
                if (g_mmvq_il() && strata::kernels::native_mmvq_il_supported(head_->type(), (int) T, (int) n_vocab_)) {
                    if (T >= 2) strata::kernels::native_q8_1_interleave(xq_, xil_, (int) N, (int) T, cs);
                    strata::kernels::native_mmvq_il(head_->type(), head_->weights(), xq_, xil_, head_logits_, (int) N,
                                                    (int) n_vocab_, (int) T, cs);
                } else
                native_mmvq(head_->type(), head_->weights(), xq_, head_logits_, (int) N, (int) n_vocab_, T, cs);
            } catch (const std::exception& e) {
                err = std::string("verify head: ") + e.what();
                return false;
            }
        }
        // Greedy, the default, is recorded here as before (no extra launch or sync per window). A request that
        // samples or penalizes is sampled again host-side after the replay (run()) with its own parameters and a
        // fresh draw counter: a captured sampler would bake them in and replay the same draws forever.
        if (argmax_rows_wanted()) {   // sm_80 to sm_89: the pick over many blocks a row (bitwise the one-block kernel's)
            argmax_rows(head_logits_, T, (int) n_vocab_, arg_scratch_, m_out_, cs);
        } else {
            SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            sample_tokens(head_logits_, T, (int) n_vocab_, nullptr, 0, sp, m_out_, cs);
        }
        // #879: whether each row's logits are all finite, beside the pick (read after the window's own sync)
        if (nan_guard_on()) strata::kernels::logits_nonfinite_rows(head_logits_, T, (int) n_vocab_, m_nf_, cs);
    }
    stamp(g.n_layers, 1, 0);
    return true;
}

std::string Verifier::profile_report() {
    if (!prof_on_ || prof_windows_ == 0) return std::string();
    static const char* names[kProfPer] = {"-", "hc-read0", "q8+qkv/q-idx gemv", "conv", "ab", "z", "rec", "q8+kv-idx",
                                          "k/v+norm-rope", "kv+idx append", "q+q-idx", "scores+topk", "kv-resolve",
                                          "attention", "gate", "", "out-proj", "router+plan+bell", "shared+quant",
                                          "waitA", "VRAM hits", "waitB", "PCIe grp", "waitCPU", "copy+combine",
                                          "(gap)", "head", "  hc0 norm", "  hc0 down", "  hc0 up", "  hc1 norm", "  hc1 down", "  hc1 up"};
    std::string out;
    char b[80];
    double total = 0;
    for (int k = 0; k < 2; ++k) {
        out += k == 0 ? " GDN layers:" : " | QSA layers:";
        for (int i = 0; i < kProfPer; ++i) {
            if (prof_sum_[k][i] <= 0) continue;
            total += prof_sum_[k][i];
            std::snprintf(b, sizeof b, " %s %.2f", names[i], prof_sum_[k][i] / 1e6 / (double) prof_windows_);
            out += b;
        }
    }
    std::snprintf(b, sizeof b, " | total %.2f ms/window over %lld windows", total / 1e6 / (double) prof_windows_, (long long) prof_windows_);
    out += b;
    for (auto& r : prof_sum_) for (double& d : r) d = 0;
    prof_windows_ = 0;
    return out;
}

// #871: is the zero-doorbell graph right for the next window?  Only while every expert of [lb_, le_) is in VRAM now
// (the host table is the one the device copy was uploaded from; a loan, a VRAM shrink or a swap in flight has
// marked some -1).  Not: the doorbell graph, which asks the pool for what the device does not hold.
void Verifier::refresh_ar() {
    if (!all_resident_ || h_res_ == nullptr || g_ == nullptr) return;
    const int64_t ne = g_->n_expert;
    const int32_t* r = h_res_ + lb_ * ne;
    const int64_t n = (le_ < 0 ? g_->n_layers : le_) * ne - lb_ * ne;
    int32_t lo = 0;
    for (int64_t i = 0; i < n; ++i) lo = std::min(lo, r[i]);
    const bool off = lo < 0;
    if (off != ar_off_) {
        ar_off_ = off;
        std::fprintf(stderr, "strata verify: layers [%lld, %lld) %s\n", (long long) lb_, (long long) (le_ < 0 ? g_->n_layers : le_),
                     off ? "have experts out of VRAM (a prompt loan, a shrink or a swap): the doorbell graph runs the windows"
                         : "are 100% VRAM resident again: the zero-doorbell graph");
    }
}

bool Verifier::capture(int T, std::string& err) {
    cudaGraphExec_t& exec_t = ar_off_ ? exec_nr_[T] : exec_[T];
    if (exec_t != nullptr) return true;
    {   // said before the capture: a process that exits inside it (#1275: Windows, 313 MiB free) leaves this line as the trace
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess)
            std::fprintf(stderr, "strata verify: capturing the %d-token window (%zu MiB of VRAM free)\n", T, free_b >> 20);
        else
            (void) cudaGetLastError();
    }
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin capture failed";
        return false;
    }
    std::string rerr;
    const bool ok = record_window(T, cs_, rerr);
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        err = rerr;
        return false;
    }
    if (ce != cudaSuccess) {
        err = std::string("verify: end capture: ") + cudaGetErrorString(ce);
        return false;
    }
    if (std::getenv("STRATA_VERIFY_NODES") != nullptr) {   // what the window graph holds
        size_t nn = 0;
        cudaGraphGetNodes(graph, nullptr, &nn);
        std::vector<cudaGraphNode_t> nodes(nn);
        cudaGraphGetNodes(graph, nodes.data(), &nn);
        std::map<std::string, int> kinds;
        for (cudaGraphNode_t nd : nodes) {
            cudaGraphNodeType ty;
            cudaGraphNodeGetType(nd, &ty);
            std::string name = "type" + std::to_string((int) ty);
            if (ty == cudaGraphNodeTypeKernel) {
                cudaKernelNodeParams kp{};
                if (cudaGraphKernelNodeGetParams(nd, &kp) == cudaSuccess) {
#if CUDART_VERSION >= 12030   // cudaFuncGetName arrived in CUDA 12.3
                    const char* fn = nullptr;
                    if (cudaFuncGetName(&fn, kp.func) == cudaSuccess && fn) name = fn;
#endif
                }
            } else if (ty == cudaGraphNodeTypeMemcpy) name = "memcpy";
            else if (ty == cudaGraphNodeTypeMemset) name = "memset";
            ++kinds[name];
        }
        std::vector<std::pair<int, std::string>> v;
        for (auto& [k2, c] : kinds) v.push_back({c, k2});
        std::sort(v.rbegin(), v.rend());
        std::fprintf(stderr, "strata verify: the %d-token window graph has %zu nodes:", T, nn);
        for (size_t i = 0; i < v.size() && i < 40; ++i) std::fprintf(stderr, " %d x %.60s;", v[i].first, v[i].second.c_str());
        std::fprintf(stderr, "\n");
#if CUDART_VERSION >= 12030   // the programmatic (PDL) edges: CUDA allows them only from a kernel node to a kernel node
        size_t ne = 0;
#if CUDART_VERSION >= 13000
        cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &ne);
#else
        cudaGraphGetEdges_v2(graph, nullptr, nullptr, nullptr, &ne);
#endif
        std::vector<cudaGraphNode_t> from(ne), to(ne);
        std::vector<cudaGraphEdgeData> ed(ne);
#if CUDART_VERSION >= 13000
        const cudaError_t ge = cudaGraphGetEdges(graph, from.data(), to.data(), ed.data(), &ne);
#else
        const cudaError_t ge = cudaGraphGetEdges_v2(graph, from.data(), to.data(), ed.data(), &ne);
#endif
        size_t prog = 0, bad = 0;
        for (size_t i = 0; ge == cudaSuccess && i < ne; ++i) {
            if (ed[i].type != cudaGraphDependencyTypeProgrammatic) continue;
            ++prog;
            cudaGraphNodeType ta, tb2;
            if (cudaGraphNodeGetType(from[i], &ta) != cudaSuccess || cudaGraphNodeGetType(to[i], &tb2) != cudaSuccess ||
                ta != cudaGraphNodeTypeKernel || tb2 != cudaGraphNodeTypeKernel)
                ++bad;
        }
        if (ge != cudaSuccess) cudaGetLastError();
        std::fprintf(stderr, "strata verify: %zu edges, %zu programmatic (PDL), %zu of them not kernel to kernel%s\n", ne,
                     prog, bad, ge != cudaSuccess ? " (the edge query failed)" : "");
#endif
    }
    const bool made = instantiate_evicting(exec_t, graph, "verify: instantiate: ", err);
    cudaGraphDestroy(graph);
    if (!made) return false;
    const cudaError_t ue = cudaGraphUpload(exec_t, cs_);
    const cudaError_t us = cudaStreamSynchronize(cs_);
    std::fprintf(stderr, "strata verify: captured the %d-token window (upload %s, sync %s)\n", T,
                 cudaGetErrorString(ue), cudaGetErrorString(us));
    return true;
}

bool Verifier::capture_commit(std::string& err) {
    if (commit_exec_ != nullptr) return true;
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const QsaShapes s = shapes_of(g);
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, ID = g.idx_key_dim, MT = max_t_;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin commit capture failed";
        return false;
    }
    bool ok = true;
    try {
        copy_i32_from_mapped(commit_, m_commit_, 2 + MT, cs_);
        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < lb_; ++l) (is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        // engine-latency (STRATA_COMMIT_BATCH, default on; =0 keeps the per-layer loop): every GDN layer's conv history and
        // state in two launches instead of two per layer (~70 graph nodes fewer; the same arithmetic per layer, bitwise).
        // The commit runs between a window and the draft round, which on HIP waits for it.
        static const bool commit_batch = [] { const char* e = std::getenv("STRATA_COMMIT_BATCH"); return !e || e[0] != '0'; }();
        bool gdn_batched = false;
        if (commit_batch) {
            int64_t n_gdn = 0;
            bool norms_ok = true;   // the per-layer loop checks each layer's ssm_norm.weight: keep that check
            for (int64_t l = lb_; l < le_ && norms_ok; ++l)
                if (!is_qsa_layer(g, l)) {
                    ++n_gdn;
                    norms_ok = need(LayerView(*wt_, l), "ssm_norm.weight", err) != nullptr;
                }
            if (!norms_ok) ok = false;
            else if (n_gdn > 0)
                gdn_batched = gdn_commit_layers(
                    ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats, (int64_t) gdn_floats,
                    (int64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size, qkv_L_ + (size_t) gdn_index * MT * C,
                    MT * C, h_L_ + (size_t) gdn_index * MT * C, MT * C, gate_L_ + (size_t) gdn_index * MT * HV,
                    beta_L_ + (size_t) gdn_index * MT * HV, MT * HV, (int) C, (int) g.ssm_k_heads, (int) HV, (int) MT,
                    (int) n_gdn, commit_, cs_);
        }
        // the QSA layers' tail restore + indexer append, one launch for all of them (one block per layer)
        bool qsa_batched = false;
        if (commit_batch && ok && !g_no_batch_kv) {
            QsaIndexerLayers il;
            int nq = 0;
            int64_t mc = -1;
            bool fits = true;
            for (int64_t l = lb_, qi = qsa_index; l < le_ && fits; ++l) {
                if (!is_qsa_layer(g, l)) continue;
                const WeightRef* wikn = need(LayerView(*wt_, l), "indexer.k_norm.weight", err);
                if (!wikn) { ok = false; fits = false; break; }
                const QsaState& st = ss.qsa_states[qi];
                if (nq >= QsaIndexerLayers::kMax || (mc >= 0 && st.max_cells != mc)) { fits = false; break; }
                mc = st.max_cells;
                il.gamma[nq] = (const float*) wikn->data;
                il.tail_src[nq] = tail_snap_ + (size_t) qi * TS;
                il.tail[nq] = st.idx_tail; il.dead[nq] = st.idx_dead; il.pooled[nq] = st.idx_pooled;
                il.block_pos[nq] = st.idx_block_pos;
                ++nq;
                ++qi;
            }
            if (ok && fits && nq > 0) {
                native_qsa_indexer_append_steps_layers(idx_raw_L_ + (size_t) qsa_index * MT * ID, MT * ID, commit_ + 2, 1,
                                                       (int) MT, 0, EPS, il, nq, s, mc, rope_scaling(), cs_);
                qsa_batched = true;
            }
        }
        for (int64_t l = lb_; l < le_ && ok; ++l) {
            const LayerView v(*wt_, l);
            if (is_qsa_layer(g, l) && qsa_batched) {
                ++qsa_index;
            } else if (!is_qsa_layer(g, l) && gdn_batched) {
                ++gdn_index;
            } else if (!is_qsa_layer(g, l)) {
                const WeightRef* wnm = need(v, "ssm_norm.weight", err);
                if (!wnm) { ok = false; break; }
                float* state = ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                const float* qkv = qkv_L_ + (size_t) gdn_index * MT * C;
                gdn_conv_commit(conv, qkv, (int) C, commit_, cs_);
                gdn_step_norm_multi(state, h_L_ + (size_t) gdn_index * MT * C, (int) C, gate_L_ + (size_t) gdn_index * MT * HV,
                                    beta_L_ + (size_t) gdn_index * MT * HV, z_, (const float*) wnm->data, EPS, y_dummy_,
                                    (int) g.ssm_k_heads, (int) HV, (int) MT, commit_, cs_,
                                    (int) MT);   // S26: t_out_begin = MT - the replay's outputs (y_dummy_) are never read
                ++gdn_index;
            } else {
                const QsaState& st = ss.qsa_states[qsa_index];
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn) { ok = false; break; }
                copy_from_mapped(st.idx_tail, tail_snap_ + (size_t) qsa_index * TS, TS, cs_);
                const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                if (!g_no_batch_kv) {
                    native_qsa_indexer_append_steps(idx_raw_L_ + (size_t) qsa_index * MT * ID, commit_ + 2, 1, (int) MT, 0,
                                                    (const float*) wikn->data, EPS, ib, s, st.max_cells, rope_scaling(), cs_);
                } else {
                    for (int64_t t = 0; t < MT; ++t)
                        native_qsa_indexer_append(idx_raw_L_ + (size_t) (qsa_index * MT + t) * ID, commit_ + 2 + t, 0,
                                                  (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                  rope_scaling(), cs_);
                }
                ++qsa_index;
            }
        }
        if (ok && ss.ple.ready() && ple_stage()) copy_indexed(ss.ple.hist, hist_snap_, HS, commit_ + 1, HS, cs_);
    } catch (const std::exception& e) {
        err = std::string("verify commit: ") + e.what();
        ok = false;
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess || cudaGraphInstantiate(&commit_exec_, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("verify: commit capture: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    return true;
}

void Verifier::stage_inputs(int T, const int32_t* tokens, int64_t pos0) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const QsaShapes s = shapes_of(g);
    int32_t* const pk = h_pos_ + (size_t) max_t_ * g.n_head;
    int32_t* const pi = pk + (size_t) max_t_ * g.n_head_kv;
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        qsa_step_fill(h_step_ + t * kStepCount, pos0 + t, s);
        const int32_t pos_t = (int32_t) (pos0 + t);
        for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = pos_t;
        for (int64_t h = 0; h < g.n_head_kv; ++h) pk[t * g.n_head_kv + h] = pos_t;
        for (int64_t h = 0; h < g.idx_q_heads; ++h) pi[t * g.idx_q_heads + h] = pos_t;
    }
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
    last_t_ = T;
    last_pos0_ = pos0;
    for (int t = 0; t < T; ++t) last_tokens_[t] = tokens[t];
    staged_ = true;
}

void Verifier::collect_profile() {
    cudaMemcpy(prof_h_.data(), prof_, prof_h_.size() * 8, cudaMemcpyDeviceToHost);
    accumulate_profile(prof_h_.data());
}

void Verifier::accumulate_profile(const unsigned long long* stamps) {
    const ModelGeometry& g = *g_;
    const int64_t L = g.n_layers;
    auto at = [&](int64_t l, int i) { return stamps[(size_t) (l * kProfPer + i)]; };
    // D8: the derived columns below read stamps the hc-read kernels write themselves or the next
    // layer's first.  A slot no kernel stamped is 0 and its unsigned difference wrapped to ~1e19 ns -
    // which is why (gap)/head/hc0 printed ~1e14 ms per window.  A missing or out-of-order stamp now
    // contributes nothing.
    const auto gap = [](unsigned long long to, unsigned long long from) {
        return (from != 0 && to != 0 && to >= from) ? (double) (to - from) : 0.0;
    };
    // only this stage's layers [lb_, le_) are ever stamped (a layer split); the head only on the last stage
    for (int64_t l = lb_; l < le_; ++l) {
        const int kind = is_qsa_layer(g, l) ? 1 : 0;
        unsigned long long prev = at(l, 0);
        for (int i = 1; i <= 24; ++i) {
            const unsigned long long x = at(l, i);
            if (x == 0 || x < prev) continue;
            prof_sum_[kind][i] += (double) (x - prev);
            prev = x;
        }
        {   // the FFN-side read split like hc-read0: norm 30, down 31, up 32; what is left of 17 is the router, plan, doorbell
            const double n1 = gap(at(l, 30), at(l, 16)), d1 = gap(at(l, 31), at(l, 30)), u1 = gap(at(l, 32), at(l, 31));
            if (n1 > 0 && d1 > 0 && u1 > 0) {
                prof_sum_[kind][30] += n1;
                prof_sum_[kind][31] += d1;
                prof_sum_[kind][32] += u1;
                prof_sum_[kind][17] -= gap(at(l, 32), at(l, 16));
            }
        }
        if (l + 1 < le_) prof_sum_[kind][25] += gap(at(l + 1, 0), at(l, 24));
        const double dn = gap(at(l, 27), at(l, 0)), dd = gap(at(l, 28), at(l, 27)), du = gap(at(l, 1), at(l, 28));
        if (dn > 0 && dd > 0 && du > 0) {   // the split exists: show it split, not twice
            prof_sum_[kind][27] += dn;      // hc-read0: norm
            prof_sum_[kind][28] += dd;      //           down
            prof_sum_[kind][29] += du;      //           up (through both halves, as before)
            prof_sum_[kind][1] -= gap(at(l, 1), at(l, 0));   // (hc-read0 shown split)
        }
    }
    prof_sum_[0][26] += gap(at(L, 1), at(L, 0));
    ++prof_windows_;
}

// 0.1.43 (STRATA_ADAPT_OVERLAP): hold the host until the adaptive tier's copies up to `layer` have landed.  The wait
// is a spin on an atomic the copy stream's host functions raise (no driver call here); 20 s is the window's own limit.
bool Verifier::adapt_gate_wait(int32_t layer, std::string& err) {
    const Clock::time_point t0 = Clock::now();
    uint32_t spins = 0;
    while (adapt_gate_->reached() < layer) {
        _mm_pause();
        if ((++spins & 1023u) != 0) continue;
        if (Clock::now() - t0 > std::chrono::seconds(20)) {
            err = "verify: the adaptive tier's copies for layer " + std::to_string(layer) + " never landed" +
                  released_note(release_gpu_waits(5000));
            return false;
        }
    }
    ms_gate += ms_since(t0);
    return true;
}

bool Verifier::run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out,
                   std::string& err) {
    using namespace strata::kernels;
    const OnDevice on_device(device_);
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU (#267); restart the engine"; return false; }
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    if (pos0 + T > ss.qsa_states[ss.qsa_primary()].max_cells) { err = "verify: the window runs past the context"; return false; }
    if (pre_launch_ && !pre_launch_(cs_, err)) return false;
    refresh_ar();
    if (!capture(T, err) || !capture_commit(err)) return false;
    VDBG("captured; staging\n");
    const Clock::time_point t0 = Clock::now();
    if (!staged_) stage_inputs(T, tokens, pos0);
    staged_ = false;
    const bool do_ple = ss.ple.ready() && ple_stage();
    uint32_t ple_rows[kVerifyMaxT * PLE_N_HEADS];
    if (do_ple) {
        int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
        for (int t = 0; t < T; ++t) {
            ngram_rows(&tokens[t], prev, 1, ss.ple.consts, ple_rows + t * PLE_N_HEADS);
            prev[0] = prev[1];
            prev[1] = tokens[t];
            ss.ple.table->prefetch_rows(ple_rows + t * PLE_N_HEADS);
        }
    }
    std::optional<HostGraphLaunch> launch_o;   // (not set when the previous stage launched this window already)
    if (trace_h_ != nullptr) std::memset(trace_h_, 0, trace_n_ * 8);   // #649: this window's breadcrumbs only
    std::atomic_thread_fence(std::memory_order_seq_cst);
    trace_ev("WINDOW", -1, -1, pos0 * 16 + T);
    ms_host += ms_since(t0);
    VDBG("staged; launching\n");
    if (h_bal_ != nullptr) std::memset(h_bal_, 0, (size_t) bal_p_.size() * 4 * 8);   // this window's stamps only
    if (ar_on() && h_plan_err_ != nullptr) *(volatile uint32_t*) h_plan_err_ = 0;
    // 0.1.43: with the all-resident graph the host serves no layer, so nothing can hold a layer back: every copy first
    if (adapt_gate_ != nullptr && adapt_gate_->active.load(std::memory_order_acquire) && ar_on() &&
        !adapt_gate_wait((int32_t) g.n_layers, err)) return false;
    // A blocking instrumented launch cannot also serve the graph's CPU doorbells.
    static const bool threaded_launch = [] {
        const char* value = std::getenv("STRATA_VERIFY_ASYNC_LAUNCH");
        return value != nullptr && value[0] == '1';
    }();
    launch_o.emplace(ar_off_ ? exec_nr_[T] : exec_[T], cs_, device_, threaded_launch,
                     [this] { (void) release_gpu_waits(5000); });
    const cudaError_t le = launch_o->status();
    trace_ev("LAUNCHED", -1, -1, (int64_t) le);
    if (le != cudaSuccess) { err = std::string("verify: launch: ") + cudaGetErrorString(le); return false; }
    (void) launch_o->query();
    VDBG("launched\n");
    volatile uint32_t* const seq = h_seq_;
    volatile uint32_t* const flag = h_flag_;
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
    const int64_t steps = (le_ - lb_) * G;
    const bool test_stall = g_test_stall > 0 && windows + 1 == g_test_stall;   // #267 test hook (off: false)
    if (ar_on() && !test_stall) {
        if (do_ple) {
            const Clock::time_point tp = Clock::now();
            if (!ss.ple.table->gather_batch(ple_rows, (size_t) T, h_ple_, err)) return false;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            _mm_sfence();
            *flag = 1;
            ms_host += ms_since(tp);
        } else {
            *flag = 1;
        }
    } else
    for (int64_t k = 0; k < steps; ++k) {
        const int64_t l = lb_ + k / G;
        const int grp = (int) (k % G);
        const uint32_t want = (uint32_t) (k + 1);
        const Clock::time_point a = Clock::now();
        auto last_flush = a;
        uint32_t spins = 0;
        progress_at("verify window: waiting for the GPU to reach layer", l);
        while (*seq < want) {
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            const auto now = Clock::now();
            if (now - last_flush > std::chrono::microseconds(2000)) {
                last_flush = now;
                const cudaError_t q = launch_o ? launch_o->query() : cudaStreamQuery(cs_);
                if (q != cudaErrorNotReady && *seq < want) {
                    trace_ev("NEVER-RANG", k, l, (int64_t) q);
                    trace_dump(stderr);
                    err = "verify: layer " + std::to_string(l) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    return false;
                }
            }
            if (now - a > std::chrono::seconds(20)) {
                // #267: the caller ends the engine; no spin kernel may outlive it
                trace_ev("TIMEOUT", k, l, (int64_t) (launch_o ? launch_o->query() : cudaStreamQuery(cs_)));
                if (g_trace) {
                    diag(stderr);   // the words and the breadcrumbs before the release ...
                    const bool drained = release_gpu_waits(5000);
                    trace_dump(stderr);   // ... and after it: did the GPU move once its waits were raised?
                    err = "verify: timed out at layer " + std::to_string(l) + released_note(drained);
                    return false;
                }
                err = "verify: timed out at layer " + std::to_string(l) + released_note(release_gpu_waits(5000));
                return false;
            }
        }
        const Clock::time_point b = Clock::now();
        if (g_trace) trace_ev("RANG", k, l, (int64_t) std::chrono::duration_cast<std::chrono::microseconds>(b - a).count());
        VDBG("layer %lld rang\n", (long long) l);
        if (adapt_gate_ != nullptr && adapt_gate_->active.load(std::memory_order_acquire) && l >= 0 &&
            (size_t) l < adapt_gate_->has.size() && adapt_gate_->has[(size_t) l] != 0 &&
            !adapt_gate_wait((int32_t) l, err)) return false;
        cur_layer_ = want - 1;
        set_plan_slot(grp);
        const int tb = gtb[grp], n = gte[grp] - gtb[grp];
        progress_at("verify window: the CPU experts of layer", l);
        if (pool != nullptr)
            pool(user, h_x_ + (size_t) tb * g.n_embd, h_ids_ + (size_t) tb * ss.k, n, ss.k,
                 h_ymiss_ + (size_t) tb * ss.k * g.n_embd, l);
        VDBG("layer %lld served\n", (long long) l);
        if (g_trace) trace_ev(*(volatile uint32_t*) h_flagA_ == want ? "SERVED" : "SERVED-NO-PLAN-YET", k, l,
                              (int64_t) ms_since(b));   // aux: ms the CPU experts took
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (h_bal_ != nullptr) {   // set_pcie_balance: the step's plan, its CPU and its PCIe experts
            const size_t bi = (size_t) ((k / G) * 2 + grp);
            bal_ok_[bi] = *(volatile uint32_t*) h_flagA_ == want;
            bal_nc_[bi] = bal_ok_[bi] ? sink_.cpu_jobs : 0;
            bal_np_[bi] = bal_ok_[bi] ? sink_.counts[2] : 0;
            sink_.cpu_jobs = 0;
        }
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        // Layer 1's pre(1, 0) copies h_ple_ -> ple_ after Layer 0's wait_flag_ge(m_flag_, 1).
        // By collecting PLE here at k == 0 (after publishing flagA/flagB for Layer 0 so the GPU can run
        // Layer 0's VRAM experts, and before raising *flag = 1), the NVMe PLE reads overlap with both
        // MtpDrafter::draft and Layer 0's attention + router + expert execution!
        if (k == 0 && do_ple) {
            const Clock::time_point tp = Clock::now();
            if (!ss.ple.table->gather_batch(ple_rows, (size_t) T, h_ple_, err)) return false;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            _mm_sfence();
            ms_host += ms_since(tp);
        }
        if (!(test_stall && k + 1 == steps)) *flag = want;
        if (h_bal_ != nullptr) bal_p_[(size_t) ((k / G) * 2 + grp)] = ms_since(b);
        ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
        ms_pool += ms_since(b);
    }
    progress_at("verify window: waiting for the GPU to finish the window (flags A/B/M raised)", (int64_t) T);
    // #267: a window the GPU never finishes (a spin kernel that never sees its flag) holds the host here; the stall
    // watchdog then releases every verifier's GPU waits (release_live_verifiers) before it ends the engine, so no
    // spin kernel outlives the process - the case that left Windows GPUs "lost" until a power cycle.  The wait
    // itself stays a blocking sync: a cudaStreamQuery poll here cost IQ3_S ~3% decode (a core calling the driver
    // beside the expert workers).
    const cudaError_t launch_result = launch_o ? launch_o->finish() : cudaSuccess;
    if (launch_result != cudaSuccess) {
        err = std::string("verify: launch: ") + cudaGetErrorString(launch_result);
        return false;
    }
    trace_ev("SYNC", -1, -1, 0);
    const cudaError_t se = cudaStreamSynchronize(cs_);
    trace_ev("SYNCED", -1, -1, (int64_t) se);
    if (se != cudaSuccess) { err = std::string("verify: ") + cudaGetErrorString(se); return false; }
    if (ar_on() && h_plan_err_ != nullptr && *(volatile uint32_t*) h_plan_err_ != 0) {   // #871: refresh_ar saw the table whole
        *(volatile uint32_t*) h_plan_err_ = 0;
        err = "verify: the all-resident plan met an expert that is not in VRAM (the residency table changed during the window)";
        return false;
    }
    commit_pending_ = false;
    progress_at("verify window: waiting for the expert copies", (int64_t) T);
    if (copy_used_) {
        cudaStreamSynchronize(copy_);   // no host function of this window may raise flag B in the next one
        copy_used_ = false;
    }
    if (h_bal_ != nullptr && !all_resident_ && !bal_paused_) pcie_balance_window(steps, G);
    if (prof_on_ && G == 1) collect_profile();   // the window's GPU stage stamps
    // ---- a sampled or penalized request: the head's sampling again, host-side so its parameters are this call's
    // own (a captured kernel would replay the same draws forever).  Row t's draw is Philox(seed, pos0 + t): tied to
    // the POSITION it samples, not to how the text was cut into windows, so a seed replays the same text whatever
    // the drafts were. Exact: a rejected row's draw is discarded, and no kept decision depends on a reused draw.
    const bool sampled = !sampling_.greedy && sampling_.temperature > 0.0f;
    if (head_sampling_ && (sampled || hist_d_ != nullptr || sampling_.logit_bias != nullptr)) {
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) pos0;
        // STRATA_SPEC_PROB: the drafter's q lists for this window, judged by rejection sampling (spec_prob.hpp)
        bool judged = false;
        if (sampled && spec_q_ != nullptr && spec_nq_ > 0 && T > 1 && T <= kVerifyMaxT) {
            if (d_spec_ == nullptr &&
                cudaMalloc((void**) &d_spec_, (size_t) kVerifyMaxT * (1 + strata::core::kSpecQStride) * sizeof(int32_t)) !=
                    cudaSuccess) {
                (void) cudaGetLastError();
                d_spec_ = nullptr;
            }
            if (d_spec_ != nullptr) {
                const int nq = std::min(spec_nq_, T - 1);
                int32_t* d_q = d_spec_ + kVerifyMaxT;
                if (cudaMemcpyAsync(d_spec_, tokens + 1, (size_t) (T - 1) * sizeof(int32_t), cudaMemcpyHostToDevice, cs_) ==
                        cudaSuccess &&
                    cudaMemcpyAsync(d_q, spec_q_, (size_t) nq * strata::core::kSpecQStride * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, cs_) == cudaSuccess)
                    judged = sample_tokens_spec(head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, d_spec_, d_q, nq,
                                                m_out_, cs_);
            }
        }
        spec_q_ = nullptr;
        spec_nq_ = 0;
        if (judged) ++spec_windows;
        else sample_tokens(head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, m_out_, cs_);
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {   // m_out_ is the mapped h_out_: synced, it is readable
            err = "verify: the head sampling failed";
            return false;
        }
    }
    for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    note_nonfinite(T, nullptr);
    if (static const bool dbg = std::getenv("STRATA_DBG_NAN") != nullptr; dbg) {   // debug: the first non-finite head
        static bool reported = false;
        if (!reported) {
            std::vector<float> h((size_t) T * (size_t) n_vocab_);
            cudaMemcpy(h.data(), head_logits_, h.size() * 4, cudaMemcpyDeviceToHost);
            for (int t = 0; t < T && !reported; ++t) {
                int64_t bad = 0;
                for (int64_t v = 0; v < n_vocab_; ++v) bad += !std::isfinite(h[(size_t) t * n_vocab_ + v]);
                if (bad) {
                    reported = true;
                    std::fprintf(stderr, "strata dbg: verify window at position %lld, row %d: %lld of %lld logits non-finite "
                                         "(token out %d)\n", (long long) pos0, t, (long long) bad, (long long) n_vocab_, out[t]);
                }
            }
        }
    }
    VDBG("window done\n");
    ++windows;
    progress_at("decode");
    progress_beat();
    return true;
}

// set_pcie_balance: each stamped step's times into the model's fits (PcieModel::sample), solved once a window.
void Verifier::pcie_balance_window(int64_t steps, int G) {
    const volatile unsigned long long* st = h_bal_;
    PcieModel& M = *bal_;
    for (int64_t k = 0; k < steps; ++k) {
        const size_t i = (size_t) ((k / G) * 2 + k % G);
        if (!bal_ok_[i]) continue;
        const unsigned long long ta = st[4 * i], tb = st[4 * i + 1], tw = st[4 * i + 2];
        if (ta == 0 || tb < ta || tw < tb || tw - ta > 1000000000ull) continue;   // not stamped in this window
        M.sample(bal_nc_[i], bal_np_[i], bal_p_[i], (double) (tw - ta) * 1e-6);
    }
    M.solve();
}

void Verifier::set_plan_slot(int grp) {
    const int64_t cap = sink_.cap;
    int32_t* base = h_plan_ + (size_t) grp * (size_t) plan_i32_;
    const int64_t i32 = 4 + (cap + 1) + cap + cap;
    const int64_t ptr_off = (i32 + 1) & ~1ll;
    sink_.counts = base;
    sink_.start = base + 4;
    sink_.dst = sink_.start + cap + 1;
    sink_.tok = sink_.dst + cap;
    sink_.ptr = (unsigned long long*) (base + ptr_off);
    sink_.ptr2 = sink_.ptr + cap;
    sink_.start2 = base + ptr_off + 4 * cap;
    const int G = groups_[last_t_] > 0 ? groups_[last_t_] : 1;
    const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
    sink_.staging = (unsigned long long) (staging_ + (size_t) (grp * per) * strata::kernels::cpu::expert_layout().max_blob);
    sink_.staging_cap = per;
}

// Flag B only rises: a host function of an earlier layer may run after a later layer already raised it directly.
void Verifier::raise_flag(uint32_t* flag, uint32_t value) {
    volatile long* f = (volatile long*) flag;
    uint32_t cur = __atomic_load_n((uint32_t*) flag, __ATOMIC_SEQ_CST);
    while (cur < value && !__atomic_compare_exchange_n((uint32_t*) flag, &cur, value, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
}

// Plan v0.3 P6: the PCIe share by DMA.  The copy engine moves the blobs while the CPU computes its own share and the
// GPU its VRAM experts; a host function raises flag B when they have landed (the graph waits for it before the PCIe
// groups).  Staging is split between the two token groups of a split window.
void Verifier::fetch_dma(void* ctx, const uint8_t* const* src, int n, size_t bytes) {
    Verifier* v = (Verifier*) ctx;
    const uint32_t want = v->cur_layer_ + 1;
    if (n <= 0) { raise_flag(v->h_flagB_, want); return; }
    v->copy_used_ = true;
    uint8_t* stage = (uint8_t*) v->sink_.staging;                  // this group's half in a split window
    // STRATA_DMA_BATCH (midhatn's #807, F12): the group's independent uploads as one cudaMemcpyBatchAsync.  Mode 0 is
    // the loop the engine always ran, errors unchecked (it never stops the engine).  A batch mode whose uploads
    // cannot be submitted stops the engine: flag B must not release the consumers to expert weights that were never
    // copied.
    const int batch_mode = strata::core::dma_batch_mode();
    if (batch_mode == 0) {
        for (int i = 0; i < n; ++i) cudaMemcpyAsync(stage + (size_t) i * bytes, src[i], bytes, cudaMemcpyHostToDevice, v->copy_);
    } else {
        const auto copied = strata::core::copy_expert_blobs(stage, src, n, bytes, v->copy_, batch_mode);
        if (copied != cudaSuccess) {
            std::fprintf(stderr, "strata DMA: expert upload failed: %s\n", cudaGetErrorString(copied));
            std::fflush(stderr);
            v->release_gpu_waits(5000);
            std::abort();
        }
    }
    FlagSet& fs = v->flag_sets_[v->cur_layer_ % (sizeof v->flag_sets_ / sizeof v->flag_sets_[0])];
    fs.flag = v->h_flagB_;
    fs.value = want;
    cudaLaunchHostFunc(v->copy_, [](void* p) { FlagSet* s = (FlagSet*) p; raise_flag(s->flag, s->value); }, &fs);
}

void Verifier::publish_plan(void* ctx) {
    Verifier* v = (Verifier*) ctx;
    _mm_sfence();
    *(volatile uint32_t*) v->h_flagA_ = v->cur_layer_ + 1;
}

bool Verifier::window_logprobs(const int32_t* targets, int T, int64_t pos0, int32_t extra_id, std::FILE* out,
                               std::string& err) {
    const OnDevice on_device(device_);
    if (head_logits_ == nullptr || n_vocab_ <= 0 || T <= 0 || targets == nullptr || out == nullptr) {
        err = "window_logprobs: no head logits for this window";
        return false;
    }
    // run() synchronized cs_ before returning, so the head of this window is complete
    std::vector<float> h((size_t) T * (size_t) n_vocab_);
    if (cudaMemcpy(h.data(), head_logits_, h.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "window_logprobs: the head logits copy failed";
        return false;
    }
    for (int t = 0; t < T; ++t) {
        const float* row = h.data() + (size_t) t * (size_t) n_vocab_;
        const int32_t tgt = targets[t];
        if (tgt < 0 || (int64_t) tgt >= n_vocab_) continue;
        int64_t top = 0;
        for (int64_t v = 1; v < n_vocab_; ++v)
            if (row[v] > row[top]) top = v;
        const double maxv = row[top];
        const bool has_extra = extra_id >= 0 && (int64_t) extra_id < n_vocab_;
        double sum = 0.0, sum_without = 0.0;   // the second skips extra_id: no cancellation when it holds ~all mass
        for (int64_t v = 0; v < n_vocab_; ++v) {
            const double e = std::exp((double) row[v] - maxv);
            sum += e;
            if (v != (int64_t) extra_id) sum_without += e;
        }
        const double lse = maxv + std::log(sum);
        const double extra = has_extra ? (double) row[extra_id] - lse : NAN;
        const double without = has_extra && tgt != extra_id && sum_without > 0.0
                                   ? (double) row[tgt] - (maxv + std::log(sum_without)) : NAN;
        std::fprintf(out, "%lld\t%d\t%.9f\t%lld\t%.9f\t%d\t%.9f\t%.9f", (long long) (pos0 + t), (int) tgt,
                     (double) row[tgt] - lse, (long long) top, maxv - lse, (int) (top == (int64_t) tgt), extra,
                     without);
        // STRATA_LOGPOS_TOPK=K: the K most likely tokens and their log-probabilities too (`id:logprob`), for a
        // top-k comparison with another engine on the same tokens (docs/UNSLOTH_Q4.md)
        static const int topk = [] {
            const char* v = std::getenv("STRATA_LOGPOS_TOPK");
            return v != nullptr ? std::max(0, std::min(256, std::atoi(v))) : 0;
        }();
        if (topk > 0) {
            std::vector<int32_t> order((size_t) n_vocab_);
            for (int64_t v = 0; v < n_vocab_; ++v) order[(size_t) v] = (int32_t) v;
            std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                              [&](int32_t a, int32_t b) { return row[a] > row[b]; });
            for (int j = 0; j < topk; ++j)
                std::fprintf(out, "\t%d:%.6f", order[(size_t) j], (double) row[order[(size_t) j]] - lse);
        }
        std::fprintf(out, "\n");
    }
    std::fflush(out);
    return true;
}

namespace { bool g_commit_async = false; }
void Verifier::set_commit_async(bool on) { g_commit_async = on && std::getenv("STRATA_COMMIT_SYNC") == nullptr; }

bool Verifier::commit(int n_keep, std::string& err) {
    const OnDevice on_device(device_);
    if (n_keep < 1 || n_keep > last_t_) { err = "verify: commit count out of range"; return false; }
    const Clock::time_point t0 = Clock::now();
    h_commit_[0] = n_keep;
    h_commit_[1] = n_keep - 1;
    for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
    if (last_t_ == 1 && one_token_self_commit()) {
        // a one-token window has advanced the state itself (record_window): no commit graph
    } else {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        // STRATA_COMMIT_TIMING=1 (measurement only): the commit graph's GPU time, from an event pair around it (read at the
        // next commit, when the pair has long completed); printed every 300 commits
        static struct CommitTiming {
            bool on = false, pending = false;
            cudaEvent_t b = nullptr, e = nullptr;
            double ms = 0;
            int64_t n = 0;
            CommitTiming() { const char* v = std::getenv("STRATA_COMMIT_TIMING"); on = v != nullptr && v[0] == '1'; }
        } ct;
        if (ct.on && ct.b == nullptr &&
            (cudaEventCreate(&ct.b) != cudaSuccess || cudaEventCreate(&ct.e) != cudaSuccess)) { ct.on = false; (void) cudaGetLastError(); }
        if (ct.on && ct.pending) {
            float ms = 0;
            if (cudaEventSynchronize(ct.e) == cudaSuccess && cudaEventElapsedTime(&ms, ct.b, ct.e) == cudaSuccess) { ct.ms += ms; ++ct.n; }
            ct.pending = false;
            if (ct.n > 0 && ct.n % 300 == 0)
                std::fprintf(stderr, "strata verify: commit graph %.3f ms GPU on average over %lld commits\n", ct.ms / (double) ct.n,
                             (long long) ct.n);
        }
        if (ct.on) cudaEventRecord(ct.b, cs_);
        const cudaError_t le = cudaGraphLaunch(commit_exec_, cs_);
        if (ct.on) { cudaEventRecord(ct.e, cs_); ct.pending = true; }
        if (le != cudaSuccess) { err = std::string("verify: commit launch: ") + cudaGetErrorString(le); return false; }
        // set_commit_async: no wait here - the next window runs on the same stream after it, and the drafter (its own
        // stream) reads only this window's final rows and its own K/V. h_commit_ is next written after the next window's
        // results are read, i.e. after this graph has run.  Everything else waits on commit_done_ (wait_commit).
        if (!g_commit_async) {
            const cudaError_t se = cudaStreamSynchronize(cs_);
            if (se != cudaSuccess) { err = std::string("verify: commit: ") + cudaGetErrorString(se); return false; }
        } else {
            const cudaError_t re = cudaEventRecord(commit_done_, cs_);
            if (re != cudaSuccess) { err = std::string("verify: commit event: ") + cudaGetErrorString(re); return false; }
            (void) cudaStreamQuery(cs_);
            commit_pending_ = true;
        }
    }
    if (ple_stage())   // stages that share one session must advance it once
        for (int t = 0; t < n_keep; ++t) {
            ss_->ple_prev[0] = ss_->ple_prev[1];
            ss_->ple_prev[1] = last_tokens_[t];
        }
    ms_commit += ms_since(t0);
    return true;
}

bool Verifier::wait_commit(std::string& err) {
    if (commit_pending_) {
        const OnDevice on_device(device_);
        commit_pending_ = false;
        const cudaError_t se = cudaEventSynchronize(commit_done_);
        if (se != cudaSuccess) { err = std::string("verify: commit: ") + cudaGetErrorString(se); return false; }
    }
    return true;
}

bool Verifier::instantiate_evicting(cudaGraphExec_t& ex, cudaGraph_t graph, const char* what, std::string& err) {
    const cudaError_t ie = cudaGraphInstantiate(&ex, graph, 0);
    if (ie == cudaErrorMemoryAllocation) (void) cudaGetLastError();
    if (ie != cudaSuccess) {
        err = std::string(what) + cudaGetErrorString(ie);
        return false;
    }
    return true;
}

// #879: STRATA_NAN_GUARD=0 turns the guard off (no kernel is recorded; the graphs are captured once, so it is read once)
bool Verifier::nan_guard_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_NAN_GUARD");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

namespace {
// STRATA_NAN_INJECT=<n>[,<n>...] (debug): the windows (counted from 1 over the process) the guard reports as non-finite
const std::vector<long long>& nan_inject_list() {
    static const std::vector<long long> list = [] {
        std::vector<long long> v;
        if (const char* e = std::getenv("STRATA_NAN_INJECT"))
            for (const char* q = e; *q != 0;) {
                char* end = nullptr;
                const long long n = std::strtoll(q, &end, 10);
                if (end == q) break;
                if (n > 0) v.push_back(n);
                q = *end == ',' ? end + 1 : end;
            }
        return v;
    }();
    return list;
}
std::atomic<long long> g_nf_windows{0};
}  // namespace

void Verifier::note_nonfinite(int T, const int64_t* pos_b) {
    nf_row_ = -1;
    nf_pos_ = -1;
    nf_mask_ = 0;
    if (!nan_guard_on() || h_nf_ == nullptr) return;
    const volatile uint32_t* f = h_nf_;
    for (int t = 0; t < T && t < strata::kernels::kVerifyMaxT; ++t)
        for (int b = 0; b < strata::kernels::kNonfiniteBlocks; ++b)
            if (f[t * strata::kernels::kNonfiniteBlocks + b] != 0) { nf_mask_ |= 1u << t; break; }
    const std::vector<long long>& inject = nan_inject_list();
    if (!inject.empty()) {
        const long long k = g_nf_windows.fetch_add(1) + 1;
        if (nf_mask_ == 0 && std::find(inject.begin(), inject.end(), k) != inject.end()) {
            nf_mask_ = 1u;
            std::fprintf(stderr, "strata verify: STRATA_NAN_INJECT: window %lld reported as non-finite\n", k);
        }
    }
    for (int t = 0; t < T && nf_row_ < 0; ++t)
        if ((nf_mask_ >> t) & 1u) nf_row_ = t;
    if (nf_row_ >= 0) nf_pos_ = pos_b != nullptr ? pos_b[nf_row_] : last_pos0_ + nf_row_;
}

bool Verifier::copy_logits(int t, float* host) const {
    if (head_logits_ == nullptr || host == nullptr || t < 0 || n_vocab_ <= 0) return false;
    return cudaMemcpy(host, head_logits_ + (size_t) t * (size_t) n_vocab_, (size_t) n_vocab_ * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
}

}  // namespace strata::core
