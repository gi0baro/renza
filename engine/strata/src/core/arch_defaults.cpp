// src/core/arch_defaults.cpp - see include/strata/core/arch_defaults.hpp.
#include "strata/core/arch_defaults.hpp"

#include <cstdlib>

namespace strata::core {

static void set_default(const std::pair<std::string, std::string>& kv) {
    setenv(kv.first.c_str(), kv.second.c_str(), 0);
}

std::vector<std::pair<std::string, std::string>> gdn_default_env_cuda(bool linux_cuda) {
    if (!linux_cuda) return {};
    return {{"STRATA_GDN_TC", "1"}, {"STRATA_GDN_CONVL2", "2"}};
}

std::vector<std::string> apply_gdn_defaults_cuda(bool linux_cuda) {
    std::vector<std::string> set;
    for (const auto& kv : gdn_default_env_cuda(linux_cuda)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;   // the user's setting wins, =0 included
        set_default(kv);
        set.push_back(kv.first);
    }
    return set;
}

}  // namespace strata::core
