// include/strata/artifact/model_family.hpp - which model family a GGUF holds, from its general.architecture.
//
// No dependencies: the GGUF reader (the architecture guard) and the core geometry both include this.
#pragma once

#include <string>

namespace strata {

enum class ModelFamily {
    Unknown = 0,
    Qwen4Exp,    ///< Qwen3.8-Flash-Next and its fine-tunes / pruned variants (Swift 1.5, the Coder)
};

inline ModelFamily model_family_from_arch(const std::string& arch) {
    if (arch == "qwen4exp") return ModelFamily::Qwen4Exp;
    return ModelFamily::Unknown;
}

/// The GGUF architecture string (also the metadata key prefix: "<arch>.block_count").
inline const char* model_family_arch(ModelFamily f) {
    switch (f) {
    case ModelFamily::Qwen4Exp:
        return "qwen4exp";
    default:
        return "";
    }
}

}  // namespace strata
