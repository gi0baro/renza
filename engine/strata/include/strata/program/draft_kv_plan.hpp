// include/strata/program/draft_kv_plan.hpp - which code computes the draft layer's prompt K/V for a chunk.
//
// The prompt path's batched pass (E-9, `Prefill::draft_kv`) tries first and the drafter's own per-8-row pass
// (`MtpDrafter::prefill`) takes what it declines.
//
// Pure (no GPU, no environment read): engine.cpp passes `use_mtp`.
#pragma once

namespace strata::program::draft_kv {

struct Plan {
    bool try_batched = false;   ///< call Prefill::draft_kv for this chunk
};

/// `use_mtp`: the engine runs a draft layer.
inline Plan plan(bool use_mtp) {
    Plan p;
    if (!use_mtp) return p;
    p.try_batched = true;
    return p;
}

}  // namespace strata::program::draft_kv
