// src/core/pinned.cu - P2.S1: the pinned host arena and the parallel expert load.
#include "strata/core/pinned.hpp"
#include "strata/platform/memory.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <climits>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <thread>

// Loader fix: `fseek`/`ftell` are 32-bit on Windows by default (and the pack is 42.9 GB), and the 64-bit
// spelling is not the same on the two platforms the engine builds for.
#define STRATA_FSEEK64(f, o) fseeko((f), (off_t) (o), SEEK_SET)

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <linux/mman.h>
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << 26)
#endif

namespace strata::core {

namespace {

// bytes in free blocks of 2 MiB or more (/proc/buddyinfo, order >= 9): what MADV_HUGEPAGE can get without compaction
uint64_t free_huge_bytes() {
    uint64_t total = 0;
    if (std::FILE* f = std::fopen("/proc/buddyinfo", "r")) {
        char line[512];
        while (std::fgets(line, sizeof line, f) != nullptr) {
            const char* z = std::strstr(line, "zone");
            if (z == nullptr) continue;
            const char* q = z + 4;
            while (*q == ' ') ++q;
            while (*q != '\0' && *q != ' ') ++q;   // the zone's name
            unsigned long long c[11] = {0};
            int n = 0;
            for (; n < 11; ++n) {
                char* e = nullptr;
                const unsigned long long v = std::strtoull(q, &e, 10);
                if (e == q) break;
                c[n] = v;
                q = e;
            }
            for (int o = 9; o < n; ++o) total += c[o] * (4096ull << o);
        }
        std::fclose(f);
    }
    return total;
}

// A 2 MB-aligned reservation.  Large pages first, then the largest alignment the OS will give us for free.
void* reserve(uint64_t bytes, PageBacking& got, std::string& note) {
    // STRATA_NO_LARGEPAGES=1 is the same-run A/B switch the Windows branch documents; honor it
    // here too, so the large-page path can be compared without changing the pool or rebooting.
    if (std::getenv("STRATA_NO_LARGEPAGES") != nullptr) {
        note = "large pages skipped (STRATA_NO_LARGEPAGES); using 4 KB pages";
    } else {
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
        if (p != MAP_FAILED) {
            got = PageBacking::LargePages;
            note = "hugetlb 2 MB pages";
            return p;
        }
        // MAP_HUGETLB is all-or-nothing: a pool smaller than the mapping fails exactly like an absent
        // one, and the old message guessed "no hugetlb pool configured" either way. Name the shortfall:
        // how many 2 MiB pages the mapping needs against what vm.nr_hugepages actually holds.
        const unsigned long long need = ((unsigned long long) bytes + (1ull << 21) - 1) / (1ull << 21);
        unsigned long long pool = 0;
        bool have_pool = false;
        if (std::FILE* f = std::fopen("/proc/sys/vm/nr_hugepages", "r")) {
            have_pool = std::fscanf(f, "%llu", &pool) == 1;
            std::fclose(f);
        }
        note = "MAP_HUGETLB unavailable (needed " + std::to_string(need) + " 2 MiB pages, vm.nr_hugepages=" +
               (have_pool ? std::to_string(pool) : std::string("?")) + "); using 4 KB pages";
    }
    // No hugetlb mapping: a 2 MiB-aligned anonymous mapping with MADV_HUGEPAGE, so transparent huge pages back
    // it where THP is "madvise" (the common distro default). The CPU expert pool streams whole experts out of
    // this arena; with 4 KB pages every 3 MB expert costs ~750 TLB misses.
    constexpr uint64_t kAlign = 2ull << 20;
    const uint64_t padded = bytes + kAlign;
    void* raw = mmap(nullptr, padded, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    got = PageBacking::NormalPages;
    if (raw == MAP_FAILED) return nullptr;
    const uintptr_t start = (uintptr_t) raw;
    const uintptr_t aligned = (start + kAlign - 1) & ~(uintptr_t) (kAlign - 1);
    if (aligned > start) munmap(raw, aligned - start);
    const uintptr_t end = aligned + bytes, raw_end = start + padded;
    if (raw_end > end) munmap((void*) end, raw_end - end);
    void* p = (void*) aligned;
    // #771: with THP "always" the kernel already backs this mapping with huge pages where it can; the extra
    // MADV_HUGEPAGE only adds direct reclaim and compaction on every fault (defrag=madvise), which on a fragmented
    // machine stretched a 25 s start to 432 s.  So it is not asked for there.  STRATA_NO_ARENA_THP=1 skips it
    // anywhere (the request alone; STRATA_NO_LARGEPAGES also skips the hugetlb try).
    std::string thp_mode;
    if (std::FILE* f = std::fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r")) {
        char line[128] = {0};
        if (std::fgets(line, sizeof line, f) != nullptr) {
            const char* open = std::strchr(line, '[');
            const char* close = open ? std::strchr(open, ']') : nullptr;
            if (open && close) thp_mode.assign(open + 1, close);
        }
        std::fclose(f);
    }
    const bool skip_thp_request = std::getenv("STRATA_NO_ARENA_THP") != nullptr || thp_mode == "always";
    if (std::getenv("STRATA_NO_LARGEPAGES") == nullptr && skip_thp_request) {
        const std::string four_k = "; using 4 KB pages";
        if (note.size() >= four_k.size() && note.compare(note.size() - four_k.size(), four_k.size(), four_k) == 0)
            note.resize(note.size() - four_k.size());
        note += std::getenv("STRATA_NO_ARENA_THP") != nullptr
                    ? "; transparent huge pages skipped (STRATA_NO_ARENA_THP)"
                    : "; transparent huge pages are on for every mapping (THP always): MADV_HUGEPAGE not requested";
    } else if (std::getenv("STRATA_NO_LARGEPAGES") == nullptr && std::getenv("STRATA_ARENA_THP") == nullptr &&
               free_huge_bytes() < bytes / 2) {
        // #2126: THP "madvise" + defrag "madvise" makes every fault of a MADV_HUGEPAGE mapping reclaim and compact
        // until it finds a 2 MiB block.  When less than half of the arena could be backed by blocks that are free
        // now (a 62 GB box holding 47 GB of anonymous memory: 148K failed compactions, 100% system time, the load
        // never ended), that is minutes of kernel work per GiB for a TLB gain.  Said, and STRATA_ARENA_THP=1 asks for it anyway.
        const std::string four_k = "; using 4 KB pages";
        if (note.size() >= four_k.size() && note.compare(note.size() - four_k.size(), four_k.size(), four_k) == 0)
            note.resize(note.size() - four_k.size());
        note += "; transparent huge pages NOT requested: only " + std::to_string(free_huge_bytes() >> 30) +
                " GiB of free 2 MiB blocks for a " + std::to_string(bytes >> 30) +
                " GiB arena, compaction would stall the load (STRATA_ARENA_THP=1 requests them anyway) (#2126)";
        std::fprintf(stderr, "strata: expert arena: %s\n", note.c_str());   // said now: the load that follows can take minutes
    } else if (std::getenv("STRATA_NO_LARGEPAGES") == nullptr && madvise(p, bytes, MADV_HUGEPAGE) == 0) {
        const std::string four_k = "; using 4 KB pages";
        if (note.size() >= four_k.size() && note.compare(note.size() - four_k.size(), four_k.size(), four_k) == 0)
            note.resize(note.size() - four_k.size());
        note += "; transparent huge pages requested (MADV_HUGEPAGE)";
    }
    return p;
}

void release(void* p, uint64_t bytes) {
    if (!p) return;
    munmap(p, bytes);
}

}  // namespace

uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed) {
    uint64_t h = seed;
    for (uint64_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

namespace {
bool clear_error() { (void) cudaGetLastError(); return true; }
}  // namespace

int arena_pin_cap_gib() {
    const char* e = std::getenv("STRATA_ARENA_PIN_GIB");
    if (e == nullptr || *e == '\0') return -1;
    if (std::string(e) == "auto") return -2;   // #243: the Windows shared-memory budget sets the sliced pin's cap
    const int v = std::atoi(e);
    return v < 0 ? -1 : v;
}

namespace {
}  // namespace

namespace {
std::vector<uint64_t> uniform_bounds(uint64_t bytes, uint64_t slice) {
    std::vector<uint64_t> b;
    if (slice == 0) return b;
    for (uint64_t off = 0; off + slice <= bytes; off += slice) b.push_back(off);
    if (!b.empty()) b.push_back(b.back() + slice);
    return b;
}
}  // namespace

PinnedArena::PinnedArena(uint64_t bytes, uint64_t slice) : PinnedArena(bytes, uniform_bounds(bytes, slice)) {
    if (slice_bytes) slice_bytes = slice;   // sliced registration: record the uniform size
}

PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds,
                         uint64_t max_pinned_bytes) : capacity(bytes) {
    if (bytes == 0) return;
    base = reserve(bytes, backing, note);
    if (base != nullptr && mapping_base == nullptr) {
        mapping_base = base;
        mapping_bytes = bytes;
    }

    // Register with CUDA BEFORE any page is touched: cudaHostRegister pins what is resident now, and a region
    // that has already been faulted in page by page is far more expensive to register and may fail outright.
    if (base) {
        // #243: STRATA_ARENA_PIN_GIB=N caps the registration from the start where the caller set no cap
        const int env_gib = arena_pin_cap_gib();
        uint64_t cap = max_pinned_bytes;
        std::string cap_why = "by the engine (multi-GPU under WDDM, or remote experts)";
        if (cap == 0 && env_gib > 0) {
            cap = (uint64_t) env_gib << 30;
            cap_why = "by STRATA_ARENA_PIN_GIB";
        }
        const bool capped = cap > 0 && cap < bytes && bounds.size() >= 2;
        const cudaError_t e = capped ? cudaSuccess :
            cudaHostRegister(base, (size_t) bytes, cudaHostRegisterPortable | cudaHostRegisterMapped);
        if (!capped && e == cudaSuccess) {
            note = "cudaHostRegister PORTABLE ok; " + note;
            registered_bytes = bytes;
        } else if (bounds.size() >= 2 && (capped || clear_error())) {
            // Plan v0.3 P5: the whole range is refused, so pin it slice by slice from the start.  The rest stays
            // resident through the working-set lock below.  (P6: slices may differ in size, one per layer.)
            slice_bytes = 1;   // sliced; the uniform constructor records the size
            bool limited = capped;
            uint64_t limit = cap;
            std::string limit_why;
            for (size_t i = 0; i + 1 < bounds.size(); ++i) {
                const uint64_t off = bounds[i], n = bounds[i + 1] - bounds[i];
                if (limited && (off > limit || n > limit - off)) break;
                if (cudaHostRegister((uint8_t*) base + off, (size_t) n, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
                    (void) cudaGetLastError();
                    break;
                }
                slice_starts.push_back(off);
                registered_bytes = off + n;
                ++registered_slices;
            }
            char gib[32];
            std::snprintf(gib, sizeof gib, "%.1f", (double) limit / (double) (1ull << 30));
            note = (capped ? "cudaHostRegister limited to " + std::to_string(cap >> 30) + " GiB " + cap_why + "; " :
                             "cudaHostRegister of the whole arena FAILED (" + std::string(cudaGetErrorString(e)) + "); " +
                             (limited ? "slices capped at " + std::string(gib) + " GiB (" + limit_why +
                                        "; STRATA_ARENA_PIN_GIB=auto; N sets a cap; #243); " : std::string())) +
                   std::to_string(registered_slices) + " slices pinned (" + std::to_string(registered_bytes >> 30) +
                   " GiB); " + note;
            if (registered_bytes < bytes) {
                const char* env = std::getenv("STRATA_ARENA_LOCK");
                if (backing == PageBacking::LargePages) {   // #779: large pages cannot be paged out: nothing to lock
                    note = "large pages are resident without a lock; " + note;
                } else if (env == nullptr || std::string(env) != "0") {
                    const strata::platform::LockResult lr =
                        strata::platform::lock_resident((uint8_t*) base + registered_bytes, bytes - registered_bytes);
                    locked_bytes = lr.locked_bytes;
                    note = lr.note + "; " + note;
                }
            }
        } else {
            note = std::string("cudaHostRegister FAILED (") + cudaGetErrorString(e) +
                   ") - the arena is NOT pinned, so copies will be slow; " + note;
            // **CONSUME THE ERROR, OR IT LIES ABOUT SOMETHING ELSE LATER.**
            //
            // `cudaGetLastError()` returns the last error and CLEARS it; until something reads it, the error
            // state is sticky.  This failure is caught and handled right here - the arena is simply not pinned -
            // but leaving it set meant the next `cudaGetLastError()` in the engine, which is `gr_read`'s launch
            // check, reported "out of memory" for kernels that allocate nothing.  That cost a round: the arena
            // was written off as not fitting the machine when in fact the only thing wrong was a stale error
            // from this line.
            //
            // It is the same trap `gr.cu` warns about for ASYNC faults, in the other direction: a synchronous
            // failure is sticky too, and it lies about where it happened just as convincingly.
            (void) cudaGetLastError();
            // Plan v0.3 P0.1: keep it RESIDENT instead. Unpinned, Windows trims the arena under memory pressure
            // and the CPU pool's rate then depends on the OS; locking it through the working set needs no
            // special privilege. STRATA_ARENA_LOCK=0 is the A/B arm.
            const char* env = std::getenv("STRATA_ARENA_LOCK");
            if (backing == PageBacking::LargePages) {       // #779: large pages cannot be paged out: nothing to lock
                note = "large pages are resident without a lock; " + note;
            } else if (env == nullptr || std::string(env) != "0") {
                const strata::platform::LockResult lr = strata::platform::lock_resident(base, bytes);
                locked_bytes = lr.locked_bytes;
                note = lr.note + "; " + note;
            } else {
                note = "arena lock disabled (STRATA_ARENA_LOCK=0); " + note;
            }
        }
    }
}

PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds, Deferred)
    : capacity(bytes), bounds_(bounds) {
    if (bytes == 0 || bounds.size() < 2) return;
    base = reserve(bytes, backing, note);
    if (base != nullptr && mapping_base == nullptr) {
        mapping_base = base;
        mapping_bytes = bytes;
    }
    slice_bytes = 1;   // sliced: one registration per layer
}

void PinnedArena::register_slices(std::atomic<int>& ready) {
    const int n = (int) bounds_.size() - 1;
    int i = 0;
    for (; base != nullptr && i < n; ++i) {
        const uint64_t off = bounds_[(size_t) i], len = bounds_[(size_t) i + 1] - off;
        if (cudaHostRegister((uint8_t*) base + off, (size_t) len, cudaHostRegisterPortable | cudaHostRegisterMapped) !=
            cudaSuccess) {
            (void) cudaGetLastError();
            break;
        }
        slice_starts.push_back(off);
        registered_bytes = off + len;
        ++registered_slices;
        ready.store(i + 1, std::memory_order_release);
    }
    note = "cudaHostRegister per layer, pipelined with the load: " + std::to_string(registered_slices) + " of " +
           std::to_string(n) + " slices pinned; " + note;
    if (i < n && base != nullptr) {   // the rest stays resident through the working-set lock, as the sliced fallback does
        const char* env = std::getenv("STRATA_ARENA_LOCK");
        if (backing == PageBacking::LargePages) {           // #779: large pages cannot be paged out: nothing to lock
            note = "large pages are resident without a lock; " + note;
        } else if (env == nullptr || std::string(env) != "0") {
            const strata::platform::LockResult lr =
                strata::platform::lock_resident((uint8_t*) base + registered_bytes, capacity - registered_bytes);
            locked_bytes = lr.locked_bytes;
            note = lr.note + "; " + note;
        }
    }
    ready.store(INT_MAX, std::memory_order_release);   // every slice done (or given up on)
}

PinnedArena::~PinnedArena() {
    if (base) {
        if (locked_bytes) strata::platform::unlock_resident((uint8_t*) base + (slice_bytes ? registered_bytes : 0), locked_bytes);
        if (slice_bytes) {
            for (uint64_t off : slice_starts) cudaHostUnregister((uint8_t*) base + off);
        } else {
            cudaHostUnregister(base);
        }
        release(mapping_base ? mapping_base : base, mapping_bytes ? mapping_bytes : capacity);
        base = nullptr;
        mapping_base = nullptr;
        mapping_bytes = 0;
    }
}

LoadStats load_experts(const std::string& path, uint8_t* dst, uint64_t blob_bytes, uint64_t blobs_per_layer,
                       uint64_t layers, int threads, uint64_t chunk) {
    std::vector<uint64_t> off((size_t) layers), n((size_t) layers, blobs_per_layer * blob_bytes);
    for (uint64_t L = 0; L < layers; ++L) off[(size_t) L] = L * blobs_per_layer * blob_bytes;
    return load_experts_ranges(path, dst, off, n, threads, chunk);
}

LoadStats load_experts_direct(const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
                              const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk,
                              const std::atomic<int>* ready) {
    LoadStats st;
    st.ok = false;
    (void) path; (void) dst; (void) layer_off; (void) layer_bytes; (void) threads; (void) chunk; (void) ready;
    return st;
}

bool experts_unbuffered(const std::vector<std::string>& files, uint64_t arena_bytes, std::string& why,
                        bool cache_counts, uint64_t read_bytes) {
    const char* env = std::getenv("STRATA_UNBUFFERED_LOAD");
    if (env != nullptr && env[0] != '\0') {
        why = std::string("STRATA_UNBUFFERED_LOAD=") + env;
        return env[0] != '0';
    }
    (void) files; (void) arena_bytes; (void) cache_counts; (void) read_bytes;
    why = "buffered (not Windows)";
    return false;
}

LoadStats load_experts_ranges(const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
                              const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk,
                              const std::atomic<int>* ready) {
    return load_experts_ranges(path, dst, layer_off, layer_bytes, threads, chunk, ready, nullptr);
}

LoadStats load_experts_ranges(const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
                              const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk,
                              const std::atomic<int>* ready, const PadMap* pad) {
    LoadStats st;
    const uint64_t layers = (uint64_t) layer_off.size();
    st.layers = layers;
    st.bytes = 0;
    for (uint64_t b : layer_bytes) st.bytes += b;
    if (threads < 1) threads = 1;

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<uint64_t> layer_hash((size_t) layers, 1469598103934665603ull);
    std::atomic<uint64_t> next_layer{0};
    std::atomic<uint64_t> read_ns_sum{0};   // summed over the threads: see LoadStats::read_seconds
    std::atomic<uint64_t> copy_ns_sum{0};
    std::mutex err_mu;
    std::string err;

    auto worker = [&]() {
        std::vector<uint8_t> buf((size_t) chunk);
        // One handle per thread, seeked once per layer: a shared handle would need a lock around the seek and
        // would serialise the very thing the threads are here to parallelise.  `fread` on a `FILE*` rather than
        // `std::ifstream`: see the header - MSVC's `basic_filebuf::xsgetn` splits any request larger than
        // `_INTERNAL_BUFSIZ - 1` into 4095-byte freads, which turned one 8 MiB chunk into ~2048 4 KiB reads.
        // `fread` sees a request bigger than the stream buffer and passes it to `_read()`/
        // `ReadFile()` unchanged, so the chunk size reaches the disk.  Buffered, not `FILE_FLAG_NO_BUFFERING`:
        // the cache should still hold what it can.
        FILE* f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) {
            std::lock_guard<std::mutex> g(err_mu);
            err = "cannot open " + path;
            return;
        }
        // A `FILE*` has no destructor that closes it, and this function has early returns below (open, seek and
        // short-read failures), so the guard is what keeps the closing correct on every path.
        struct Closer {
            FILE* f;
            ~Closer() { if (f != nullptr) std::fclose(f); }
        } closer{f};
        uint64_t read_ns = 0, copy_ns = 0;
        for (;;) {
            const uint64_t L = next_layer.fetch_add(1);
            if (L >= layers) break;
            if (ready != nullptr)      // the slice must be registered before its pages are touched
                while (ready->load(std::memory_order_acquire) <= (int) L + 1) std::this_thread::yield();
            const uint64_t off = layer_off[(size_t) L];
            uint64_t remaining = layer_bytes[(size_t) L];
            uint64_t pos = 0;
            uint64_t h = 1469598103934665603ull;
            // 64-bit seek: the pack is 42.9 GB, so the 32-bit `fseek` would wrap past 4 GiB
            if (STRATA_FSEEK64(f, off) != 0) {
                std::lock_guard<std::mutex> g(err_mu);
                err = "seek to " + std::to_string(off) + " B failed in layer " + std::to_string(L);
                return;
            }
            while (remaining > 0) {
                const uint64_t n = remaining < chunk ? remaining : chunk;
                const auto t_read = std::chrono::steady_clock::now();
                const size_t got = std::fread(buf.data(), 1, (size_t) n, f);
                read_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - t_read).count();
                // A short read is EOF or an I/O error, never a silent zero fill: say WHERE and HOW SHORT, and
                // keep going no further - the caller turns this into a refused load, not a wrong answer.
                if (got != (size_t) n) {
                    std::lock_guard<std::mutex> g(err_mu);
                    err = "short read in layer " + std::to_string(L) + ": got " + std::to_string(got) + " of "
                          + std::to_string(n) + " B at offset " + std::to_string(off + pos)
                          + (std::ferror(f) != 0 ? " (ferror set)" : "");
                    return;
                }
                const auto t_copy = std::chrono::steady_clock::now();
                if (pad == nullptr) {
                    std::memcpy(dst + off + pos, buf.data(), (size_t) n);
                } else {   // the padded layout: a chunk may span blobs, each lands at its own page-aligned start
                    const uint64_t B = pad->blob[(size_t) L], S = pad->stride[(size_t) L];
                    for (uint64_t q = 0; q < n;) {
                        const uint64_t p = pos + q, bi = p / B, in = p % B;
                        const uint64_t m = std::min<uint64_t>(n - q, B - in);
                        std::memcpy(dst + pad->dst_off[(size_t) L] + bi * S + in, buf.data() + q, (size_t) m);
                        q += m;
                    }
                }
                h = fnv1a64(buf.data(), n, h);
                copy_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - t_copy).count();
                pos += n;
                remaining -= n;
            }
            layer_hash[(size_t) L] = h;
        }
        read_ns_sum.fetch_add(read_ns);
        copy_ns_sum.fetch_add(copy_ns);
    };

    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();

    if (!err.empty()) {
        std::fprintf(stderr, "load_experts: %s\n", err.c_str());
        st.seconds = -1.0;
        st.ok = false;
        st.error = err;
        return st;
    }
    st.read_seconds = (double) read_ns_sum.load() / 1e9;
    st.copy_seconds = (double) copy_ns_sum.load() / 1e9;
    st.layer_checksums = std::move(layer_hash);
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

StreamStats stream_bandwidth(const uint8_t* src, uint64_t bytes, uint64_t chunk, int iters) {
    StreamStats st;
    st.bytes = bytes * (uint64_t) iters;
    st.chunk = chunk;
    uint8_t* dst = nullptr;
    cudaStream_t s{};
    if (cudaMalloc(&dst, (size_t) chunk) != cudaSuccess) {
        std::fprintf(stderr, "stream_bandwidth: cudaMalloc failed for %llu B\n", (unsigned long long) chunk);
        st.seconds = -1.0;
        return st;
    }
    cudaStreamCreate(&s);

    // one untimed pass so the first transfer's page-fault and setup cost is not in the measurement
    for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
        cudaMemcpyAsync(dst, src + off, (size_t) chunk, cudaMemcpyHostToDevice, s);
    }
    cudaStreamSynchronize(s);

    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
            cudaMemcpyAsync(dst, src + off, (size_t) chunk, cudaMemcpyHostToDevice, s);
        }
    }
    cudaStreamSynchronize(s);
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    cudaStreamDestroy(s);
    cudaFree(dst);
    return st;
}

}  // namespace strata::core
