// src/program/draft_kv_plan_test.cpp - which path computes the draft layer's prompt K/V (header-only, CPU).
#include "strata/program/draft_kv_plan.hpp"

#include <cstdio>

namespace dkv = strata::program::draft_kv;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-72s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    std::printf("draft_kv_plan_test\n");
    {
        const dkv::Plan p = dkv::plan(true);
        check(p.try_batched, "a draft layer: the batched pass is tried");
    }
    {
        const dkv::Plan p = dkv::plan(false);
        check(!p.try_batched, "no draft layer: nothing tried");
    }

    std::printf(g_fail == 0 ? "draft_kv_plan_test: all ok\n" : "draft_kv_plan_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
