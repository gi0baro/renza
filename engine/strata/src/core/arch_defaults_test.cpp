// src/core/arch_defaults_test.cpp - the per-architecture default table (no GPU needed).
#include "strata/core/arch_defaults.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static void unset(const char* k) {
    unsetenv(k);
}
static void put(const char* k, const char* v) {
    setenv(k, v, 1);
}

int main() {
    using namespace strata::core;
    // QUAL-GDN on Linux CUDA
    CHECK(gdn_default_env_cuda(true).size() == 2);
    CHECK(gdn_default_env_cuda(false).empty());
    {
        unset("STRATA_GDN_TC"); put("STRATA_GDN_CONVL2", "0");
        CHECK(apply_gdn_defaults_cuda(true).size() == 1);
        CHECK(std::string(std::getenv("STRATA_GDN_TC")) == "1" && std::string(std::getenv("STRATA_GDN_CONVL2")) == "0");
        unset("STRATA_GDN_TC"); unset("STRATA_GDN_CONVL2");
        CHECK(apply_gdn_defaults_cuda(false).empty() && std::getenv("STRATA_GDN_TC") == nullptr);
    }
    // Both Linux CUDA defaults must be installed when neither has a user override.
    {
        unset("STRATA_GDN_TC"); unset("STRATA_GDN_CONVL2");
        const auto set = apply_gdn_defaults_cuda(true);
        CHECK((set == std::vector<std::string>{"STRATA_GDN_TC", "STRATA_GDN_CONVL2"}));
        CHECK(std::string(std::getenv("STRATA_GDN_TC")) == "1" && std::string(std::getenv("STRATA_GDN_CONVL2")) == "2");
        unset("STRATA_GDN_TC"); unset("STRATA_GDN_CONVL2");
    }
    return fails ? 1 : 0;
}
