// strata/program/engine.hpp - Strata's serve program as a library: Engine::open loads the model the way
// `strata --serve` starts up, a Session runs its requests (sync a prompt, decode one window at a time, finish).
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::program {

// The serve program's exit statuses, which open's errors carry.
inline constexpr int kExitOk = 0;
inline constexpr int kExitFailure = 1;   // a load failure
inline constexpr int kExitUsage = 2;     // bad or missing options

// Session results.  decode returns a token count >= 1 where the others return kOk.
inline constexpr int kOk = 0;
inline constexpr int kRefused = -1;       // a precondition failed: nothing changed
inline constexpr int kInterrupted = -2;   // sync stopped by its cancel callback
inline constexpr int kDead = -3;          // the engine cannot continue: every later call returns kDead
inline constexpr int kInvalidated = -4;   // non-finite logits: the request ended and the next sync reads from 0

// decode's token: the first call after a sync passes it and reads the last prompt token.
inline constexpr int32_t kNoToken = -1;

// <|image_pad|> (qwen4exp.ple.image_token_id): one per image row in a prompt; the PLE hash reads it for image cells.
inline constexpr int64_t kImagePad = 248056;

struct OpenError {
    int status = kExitOk;
    std::string message;   // the line start-up wrote to stderr, without its newline
};

// One image of a prompt: its rows go to the prompt positions [start, start + token_count), all <|image_pad|>.
// The M-RoPE positions come from the merged patch grid (grid_width x grid_height == token_count); `key` is what
// the conversation cache matches the picture on, with its grid.  The rows (token_count x embd_dim floats,
// row-major) need only live for the sync call.
struct ImageSpan {
    int64_t start = 0;
    int64_t token_count = 0;
    int64_t grid_width = 0;
    int64_t grid_height = 0;
    const float* rows = nullptr;
    uint64_t key = 0;
};

// Strata's request sampling keys, with their protocol defaults (greedy, no filters, no penalties).
struct Sampling {
    float temperature = 0.0f;   // <= 0: greedy
    int top_k = 20;             // 0 or more than 64: the sampler keeps 64
    float top_p = 1.0f;
    float min_p = 0.0f;         // clamped to [0, 1]
    int penalty_last_n = 0;     // 0: no penalties
    float penalty_repeat = 1.0f;
    float penalty_freq = 0.0f;
    float penalty_present = 0.0f;
    uint64_t seed = 0;          // 0: a seed from the clock, drawn when the sampling is applied
    bool operator==(const Sampling&) const = default;
};

// A prompt chunk reached the session state: `done` of `total` prompt tokens are in it; the reading started at
// `from`, `elapsed_ms` after the request began.
struct SyncProgress {
    int64_t done = 0;
    int64_t total = 0;
    int64_t from = 0;
    double elapsed_ms = 0.0;
};
using ProgressFn = std::function<void(const SyncProgress&)>;
using CancelFn = std::function<bool()>;

// The request's counters, as the protocol's DONE line reports them (valid after finish).
struct RequestStats {
    int64_t generated = 0;
    int64_t prompt_tokens = 0;
    double prompt_ms = 0.0;
    double decode_ms = 0.0;
    int64_t draft_accepted = 0;
    int64_t draft_offered = 0;
    int64_t reused = 0;
    int64_t expert_hits = 0;
    int64_t expert_lookups = 0;
    int64_t ram_reads = 0;
    int64_t file_reads = 0;
    double file_read_mb = 0.0;
    int64_t prompt_read = 0;
    int64_t offloaded = 0;
    bool chain_counts = false;   // --lookup-chain: the five counts below are reported
    int64_t chain_accepted = 0;
    int64_t chain_offered = 0;
    int64_t suffix_accepted = 0;
    int64_t suffix_offered = 0;
    int64_t windows = 0;
    int64_t chunk = 0;
    int64_t experts_streamed = 0;
};

class Session;

// One engine per process, for the life of the process: a second open is refused, even after close.  Engines are
// heap objects that never move.
class Engine {
public:
    // Parses argv (Strata's own options, argv[0] the program), runs start-up and the serve set-up.  On failure
    // returns nullptr with error.status kExitUsage or kExitFailure; nullptr with kExitOk means --version or --help
    // was printed.
    static std::unique_ptr<Engine> open(int argc, char** argv, OpenError& error);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // The engine's one session, for prompts plus generation of up to ctx tokens (1 <= ctx <= max_context).
    // nullptr with `error` set while another session exists, for a ctx out of range, or on a dead engine.
    std::unique_ptr<Session> create_session(int64_t ctx, std::string& error);
    // Finishes an open request, lands the adaptive tier's round in flight and writes --expert-profile-save's
    // profile, then destroys what open built; a live session is detached (its calls return kDead).  On a dead
    // engine it makes no CUDA call and frees only host memory (the rest is left to the process exit).  Returns
    // kOk, or kDead when the engine was or went dead (dead_message() says why).  Afterwards only dead() and
    // dead_message() are valid.  Also run by the destructor.
    int close();

    bool dead() const;
    const std::string& dead_message() const;
    // The fields of the protocol's INFO line, measured at the end of open.
    const std::string& info() const;

    int64_t spec() const;
    int64_t prefill_chunk() const;
    int64_t max_context() const;
    int64_t vocab_size() const;
    int64_t embd_dim() const;
    int64_t layer_count() const;
    bool has_vision() const;
    const std::vector<int64_t>& eos_ids() const;

private:
    friend class Session;
    struct State;
    Engine();
    std::unique_ptr<State> state_;
    Session* session_ = nullptr;
    bool closed_dead_ = false;
    std::string closed_message_;
};

// The engine's conversation: Strata's live state, its checkpoints and the request in flight.  One thread at a time
// with its engine.  The callbacks of sync run inside it.
class Session {
public:
    // Finishes an open request and drops the conversation state.
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    // Brings the state to ids[0, n) (n < ctx) and the image rows, resuming from the live state or the longest
    // checkpoint that is a whole prefix (tokens and images), and stops before the first decode window: the last
    // prompt token stays unread.  Finishes an open request first.  `progress` is called once with done = reused()
    // before anything is read, then after each chunk; `cancel` is polled between chunks and windows.  kOk;
    // kInterrupted (held() tells what the state holds); kRefused (bad ids, images or n; nothing changed);
    // kInvalidated; kDead.  After kInterrupted or kInvalidated the request is open until finish, decode refuses.
    int sync(const int32_t* ids, int64_t n, const ImageSpan* images, size_t image_count, const ProgressFn& progress,
             const CancelFn& cancel);
    // The prompt tokens the last sync resumed from instead of reading (the protocol's RESUME / REUSED).
    int64_t reused() const;
    // After a sync: the prompt tokens the state holds exactly (all but the last after kOk), or -1 when it was
    // stopped with a chunk in flight.
    int64_t held() const;
    // One verify window: feeds `token` (kNoToken on the first call after a sync: the last prompt token), samples,
    // and writes n >= 1 tokens to out[0, n), n <= min(budget, cap); each follows the one before, out[0..n-1) are fed,
    // out[n - 1] is not (the next call's token), and an end-of-generation token can only be out[n - 1].  A token that
    // is not the engine's own last sample drops the drafts (that window holds the fed token alone).  Returns n, or
    // kRefused, kInvalidated (the request stays open until finish), kDead.
    int decode(int32_t token, const Sampling& sampling, int budget, int32_t* out, int cap);
    // Ends the open request: the live state for the next sync, the per-request stderr summary, stats().
    // Idempotent; kOk or kDead.
    int finish();
    const RequestStats& stats() const;
    // Drops the live state and every checkpoint (after finishing an open request): the next sync reads from 0.
    void invalidate();
    int64_t ctx() const;
    // Why the last call was refused, interrupted, invalidated or died.
    const std::string& error() const;

private:
    friend class Engine;
    Session(Engine& engine, int64_t ctx);
    Engine* engine_;
    int64_t ctx_;
};

// The key Strata gives a picture whose request names none: the hash of its rows.
uint64_t image_rows_key(const float* rows, size_t floats);

}  // namespace strata::program
