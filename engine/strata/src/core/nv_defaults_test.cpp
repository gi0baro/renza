// src/core/nv_defaults_test.cpp - the NVIDIA sm_86 default table (no GPU needed).
#include "strata/core/nv_defaults.hpp"

#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <string>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static void unset(const char* k) {
    unsetenv(k);
}
static void put(const char* k, const char* v) {
    setenv(k, v, 1);
}

static const std::string* find(const std::vector<std::pair<std::string, std::string>>& t, const char* k) {
    for (const auto& kv : t)
        if (kv.first == k) return &kv.second;
    return nullptr;
}

int main() {
    using namespace strata::core;
    unset("STRATA_SM86_PERF");
    CHECK(nv_default_env(12, 0, false).empty());
    CHECK(nv_default_env(10, 0, false).empty());
    CHECK(nv_default_env(8, 9, false).empty());
    CHECK(nv_default_env(7, 5, false).empty());
    CHECK(nv_default_env(6, 0, false).empty());
    CHECK(nv_default_env(12, 1, false).empty());
    const auto a = nv_default_env(8, 6, false), b = nv_default_env(8, 6, true);
    CHECK(!a.empty() && a == b);               // STRATA_DEFAULTS=026 does not touch the 2026-10 sm_86 table
    CHECK(!find(a, "STRATA_PF_FUSED"));

    const std::pair<const char*, const char*> expected[] = {
        {"STRATA_ADAPT_NOWAIT", "1"}, {"STRATA_ADAPT_LAG", "2"}, {"STRATA_CPU_IQ_ALL_EXACT1", "1"},
        {"STRATA_SIDE_MIXER", "2"}, {"STRATA_ATTN_DAILY", "1"}, {"STRATA_GR_UP75", "1"},
        {"STRATA_GU_SWIGLU", "1"}, {"STRATA_QSA_ROWS", "1"}, {"STRATA_B6_MMVQ_ROWS", "1"},
    };
    CHECK(a.size() == std::size(expected));
    for (const auto& kv : expected) {
        CHECK(find(a, kv.first) && *find(a, kv.first) == kv.second);
        unset(kv.first);
    }
    const auto set = apply_nv_defaults(8, 6);
    CHECK(set.size() == std::size(expected));
    for (size_t i = 0; i < std::size(expected); ++i) {
        CHECK(i < set.size() && set[i] == expected[i].first);
        const char* value = std::getenv(expected[i].first);
        CHECK(value && std::string(value) == expected[i].second);
        unset(expected[i].first);
    }
    put("STRATA_SIDE_MIXER", "0");
    put("STRATA_DEFAULTS", "026");
    CHECK(nv_default_env(8, 6) == a);
    CHECK(apply_nv_defaults(8, 6).size() == std::size(expected) - 1);
    CHECK(std::string(std::getenv("STRATA_SIDE_MIXER")) == "0");
    CHECK(apply_nv_defaults(8, 6).empty());
    unset("STRATA_DEFAULTS");
    for (const auto& kv : expected) unset(kv.first);
    put("STRATA_SM86_PERF", "0");
    CHECK(nv_default_env(8, 6).empty());
    CHECK(apply_nv_defaults(8, 6).empty());
    for (const auto& kv : expected) CHECK(std::getenv(kv.first) == nullptr);
    unset("STRATA_SM86_PERF");
    if (fails == 0) std::printf("nv_defaults_test: ok\n");
    return fails == 0 ? 0 : 1;
}
