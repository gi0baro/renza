// include/strata/core/device.hpp - P2.S1: the device-code check.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// "" when this build has device code for the current device, else CUDA's error: a build for other GPUs would
/// otherwise fail at its first kernel launch, with nothing that names the cause.
std::string device_code_error();

}  // namespace strata::core
