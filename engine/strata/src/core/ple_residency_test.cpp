#include "strata/core/ple_residency.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
constexpr uint64_t GiB = 1ull << 30;
}

int main() {
    const uint64_t table = 26ull * GiB + 870ull * (1ull << 20);   // IQ3_S's table, 26.8 GiB
    const uint64_t arena = 51ull * GiB + 100ull * (1ull << 20);   // IQ3_S's 51.1 GiB of experts
    // the owner PC, 2026-10-26: 45.2 GiB available when the table is opened, 63 GiB total.  The 0.2.7 first policy
    // (table + 8 GiB) admitted it and the expert load then failed with error 1453.  Not admitted now.
    check(!ple_resident_admit(45ull * GiB + (1ull << 28), table, arena, 8 * GiB), "64 GB PC with the IQ3_S arena: not admitted");
    check(!ple_resident_admit(40ull * GiB, table, arena, 8 * GiB), "40 GiB: not admitted");
    // STRATA_DEFAULTS=026: the policy does not run (v0.2.6 had none); an explicit STRATA_PLE_AUTO_RAM wins either way
    check(ple_auto_ram_allowed(nullptr, false), "policy on by default");
    check(!ple_auto_ram_allowed(nullptr, true), "policy off under STRATA_DEFAULTS=026");
    check(ple_auto_ram_allowed("1", true), "STRATA_PLE_AUTO_RAM=1 wins over 026");
    check(!ple_auto_ram_allowed("0", false), "STRATA_PLE_AUTO_RAM=0 is the old off switch");
    check(!ple_auto_ram_allowed("0", true), "STRATA_PLE_AUTO_RAM=0 under 026");
    // a PC that holds the table, the arena and 8 GiB
    check(ple_resident_admit(table + arena + 8 * GiB, table, arena, 8 * GiB), "exactly table + arena + headroom is admitted");
    check(!ple_resident_admit(table + arena + 8 * GiB - 1, table, arena, 8 * GiB), "one byte below is not");
    check(ple_resident_admit(240ull * GiB, table, arena, 8 * GiB), "a 256 GB box: admitted");
    // a smaller model's arena on a mid PC (Q2_0: 34 GiB arena): 26.8 + 34 + 8 = 68.8
    check(ple_resident_admit(80ull * GiB, table, 34ull * GiB, 8 * GiB), "Q2_0 on 96 GB: admitted");
    check(!ple_resident_admit(60ull * GiB, table, 34ull * GiB, 8 * GiB), "Q2_0 with 60 GiB available: not admitted");
    // unknowns fail closed
    check(!ple_resident_admit({}, table, arena, 8 * GiB), "unknown telemetry");
    check(!ple_resident_admit(240ull * GiB, 0, arena, 8 * GiB), "unknown table size");
    check(!ple_resident_admit(240ull * GiB, table, 0, 8 * GiB), "unknown arena size");
    check(!ple_resident_admit(std::numeric_limits<uint64_t>::max(), table, std::numeric_limits<uint64_t>::max(), 8 * GiB),
          "sizes cannot overflow");
    check(!ple_resident_admit(0, table, arena, 0), "zero available");
    // #4846 (owner 11-01): RAM still short at the arena after the wait and the give-back: the --mmap-experts source, never a refusal,
    // never registering anyway.  A hand-set --ple-io ram or a shared arena file is respected (warn only).
    {
        const uint64_t need = arena + (1ull << 20);   // the arena and one blob
        const uint64_t none = std::numeric_limits<uint64_t>::max();   // no cgroup limit
        check(arena_ram_verdict(need, need, none, true) == ArenaRamVerdict::Load, "RAM exactly there: load");
        check(arena_ram_verdict(need, 240ull * GiB, none, true) == ArenaRamVerdict::Load, "plenty: load");
        check(arena_ram_verdict(need, need - 1, none, true) == ArenaRamVerdict::FallBack, "one byte short: file source");
        check(arena_ram_verdict(need, 20ull * GiB, none, true) == ArenaRamVerdict::FallBack, "20 GiB of 51: file source");
        check(arena_ram_verdict(need, 20ull * GiB, none, false) == ArenaRamVerdict::Warn, "20 GiB of 51, fixed by hand: warn only");
        check(arena_ram_verdict(need, 240ull * GiB, 30ull * GiB, true) == ArenaRamVerdict::FallBack, "a 30 GiB container limit: file source, not a refusal");
        check(arena_ram_verdict(need, 240ull * GiB, 30ull * GiB, false) == ArenaRamVerdict::Refuse, "a 30 GiB container limit, fixed by hand: as before");
        check(arena_ram_verdict(need, 240ull * GiB, need, true) == ArenaRamVerdict::Load, "a limit of exactly the arena: load");
        check(arena_ram_verdict(0, 0, 0, true) == ArenaRamVerdict::Load, "an unpriced arena loads");
        check(arena_ram_fallback_allowed(nullptr, false, false), "fallback on by default");
        check(!arena_ram_fallback_allowed(nullptr, true, false), "a hand-set --ple-io ram is kept");
        check(!arena_ram_fallback_allowed(nullptr, false, true), "a shared arena file is kept");
        check(!arena_ram_fallback_allowed("0", false, false), "STRATA_ARENA_RAM_FALLBACK=0 is the old behaviour");
        check(arena_ram_fallback_allowed("1", false, false), "STRATA_ARENA_RAM_FALLBACK=1");
        check(arena_ram_fallback_allowed("", false, false), "an empty value is unset");
        check(!arena_ram_fallback_allowed("1", true, false), "the user's --ple-io ram wins over the env value");
    }
    {
        const uint64_t need = 46ull * GiB + 860ull * (1ull << 20) + 2660000ull;   // IQ3_S: 46.84 GiB of experts and the spare blob
        const uint64_t none = std::numeric_limits<uint64_t>::max();
        const uint64_t win = fallback_safety_headroom(true), lin = fallback_safety_headroom(false);
        check(win == 8 * GiB && lin == 4 * GiB, "safety headroom: 8 GiB Windows, 4 GiB elsewhere");
        check(arena_need_with_keep(need, 0) == need && arena_need_with_keep(need, 8 * GiB) == need + 8 * GiB, "need plus W");
        check(arena_need_with_keep(none - 5, 100) == none, "need plus W cannot overflow");
        // W = 0 is the behaviour before the setting: the arena exactly when arena_ram_verdict says Load, the copy leaves the safety headroom
        const double frees[] = {10.0, 24.0, 30.0, 36.0, 40.0, 44.6, 46.5, 48.0, 48.9, 49.1, 50.0, 54.9, 60.0, 100.0};
        for (const double f : frees) {
            const uint64_t avail = (uint64_t) (f * (double) GiB);
            for (const uint64_t safety : {win, lin}) {
                const uint64_t headroom = fallback_headroom(std::nullopt, 0, safety);
                check(headroom == safety, "W=0: the safety headroom");
                check(fallback_budget(avail, headroom) == (avail > safety + GiB ? avail - safety : 0), "W=0: the budget is what is available less the safety headroom");
                // the environment variable wins over both W and the safety headroom, even when smaller
                check(fallback_headroom(std::optional<uint64_t>(2 * GiB), 8 * GiB, safety) == 2 * GiB, "STRATA_RESIDENT_HEADROOM_GIB still wins");
            }
        }
        // the owner's PC (63.1 GB, 12 GB card, IQ3_S): 48.9 GiB free at the start
        const uint64_t f489 = (uint64_t) (48.9 * (double) GiB), f446 = (uint64_t) (44.6 * (double) GiB);
        check(arena_ram_verdict(arena_need_with_keep(need, 0), f489, none, true) == ArenaRamVerdict::Load, "48.9 GiB free, W=0: the arena");
        check(arena_ram_verdict(arena_need_with_keep(need, 0), f446, none, true) != ArenaRamVerdict::Load &&
                  fallback_budget(f446, fallback_headroom(std::nullopt, 0, win)) > 36 * GiB,
              "44.6 GiB free, W=0: the fallback with a 36.6 GiB copy");
    }
    // A HARD memory limit just above the arena (R5070, found on l620: a 49 GiB cgroup limit, the 46.84 GiB arena chosen, the engine killed by the
    // cgroup OOM killer at the start).  With the fallback allowed the arena needs its headroom under the limit too.
    {
        const uint64_t need = 46ull * GiB + 860ull * (1ull << 20) + 2660000ull;   // IQ3_S: 46.84 GiB of experts and the spare blob
        const uint64_t head4 = 4 * GiB, head8 = 8 * GiB;
        auto verdict = [&](uint64_t limit, uint64_t keep, uint64_t headroom, bool fallback) {
            return arena_ram_verdict(arena_hard_limit_need(need, keep, headroom, fallback), ~uint64_t{0}, limit, fallback);
        };
        check(arena_hard_limit_need(need, 0, head4, true) == need + head4, "fallback allowed: the arena needs its headroom under the limit");
        check(arena_hard_limit_need(need, 8 * GiB, head4, true) == need + 8 * GiB, "the RAM to keep counts when it is larger than the headroom");
        check(arena_hard_limit_need(need, 0, head4, false) == need, "fallback not allowed: the comparison stays as it was");
        check(arena_hard_limit_need(need, 8 * GiB, head4, false) == need + 8 * GiB, "fallback not allowed: the RAM to keep is still asked");
        check(verdict(49ull * GiB, 0, head4, true) == ArenaRamVerdict::FallBack, "49 GiB limit, 4 GiB headroom: the copy, not an arena that the OOM killer ends");
        check(verdict(49ull * GiB, 0, head8, true) == ArenaRamVerdict::FallBack, "49 GiB limit, 8 GiB headroom (Windows): the copy");
        check(verdict(51ull * GiB, 0, head4, true) == ArenaRamVerdict::Load, "51 GiB limit, 4 GiB headroom: the arena");
        check(verdict(need + head4, 0, head4, true) == ArenaRamVerdict::Load, "a limit of exactly arena + headroom: the arena");
        check(verdict(need + head4 - 1, 0, head4, true) == ArenaRamVerdict::FallBack, "one byte under: the copy");
        check(verdict(need - 1, 0, head4, true) == ArenaRamVerdict::FallBack, "a limit under the arena: the copy, as before");
        check(verdict(49ull * GiB, 0, head4, false) == ArenaRamVerdict::Load, "fallback not allowed, 49 GiB limit: as before (no new refusal)");
        check(verdict(45ull * GiB, 0, head4, false) == ArenaRamVerdict::Refuse, "fallback not allowed, limit under the arena: refused as before");
        check(verdict(60ull * GiB, 8 * GiB, head4, true) == ArenaRamVerdict::Load, "60 GiB limit, 8 GiB to keep: the arena");
        check(verdict(54ull * GiB, 8 * GiB, head4, true) == ArenaRamVerdict::FallBack, "54 GiB limit, 8 GiB to keep: the copy");
        check(verdict(std::numeric_limits<uint64_t>::max(), 0, head4, true) == ArenaRamVerdict::Load, "no limit: the arena");
    }
    std::printf("ple_residency_test: %d checks passed\n", checks);
    return 0;
}
