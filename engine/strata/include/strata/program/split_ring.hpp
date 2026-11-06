#pragma once

#include <cstdint>

// The lend slots, kept apart from the engine so they can be tested on the CPU.
//
//  split_lend_slots: how many tail slots of a cache `need` bytes take - the scan's own arithmetic, used for the
//     RAM-copy region and for the borrow itself so the two cannot disagree.
namespace strata::program {

/// The number of tail slots of a cache of `slots` slots whose bytes cover `need`.  `offsets` is the cache's per-slot
/// byte-offset prefix sum (slots + 1 entries, `bytes` = offsets[slots]) or nullptr when every slot is `max_blob` bytes.
inline int64_t split_lend_slots(int64_t slots, int64_t bytes, const uint64_t* offsets, uint64_t max_blob, uint64_t need) {
    if (offsets != nullptr) {
        int64_t k = 0;
        while (k < slots && (uint64_t) (bytes - (int64_t) offsets[slots - k]) < need) ++k;
        return k;
    }
    return max_blob == 0 ? 0 : (int64_t) ((need + max_blob - 1) / max_blob);
}

}  // namespace strata::program
