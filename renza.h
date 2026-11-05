#ifndef RENZA_H
#define RENZA_H

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The session contract between the server (C) and the engine (C++/CUDA behind
 * this C ABI).  ds4.h pruned to what the server needs and Strata's engine can
 * provide; docs/work/renza-h-mapping.md maps every function to its Strata
 * mechanism.
 *
 * renza_engine is the loaded model; renza_session is one mutable inference
 * timeline over it.  The server hands the session whole token prefixes
 * (sync) and steps generation one decode window at a time; the engine
 * decides how much of its state it can reuse.
 *
 * Pointer arguments must not be NULL unless a function says otherwise.
 *
 * Status codes: functions returning int return RENZA_OK (0) or one of the
 * negative codes below, except where a count is documented.  On error, err
 * (when non-NULL and errlen > 0) holds a NUL-terminated message.
 * RENZA_ENGINE_DEAD means the engine hit a failure it cannot recover from
 * (a sticky CUDA error, a failed restore, a failed window): the call that
 * hit it and every later call on that engine or its sessions fail with it;
 * the only remaining operations are the queries, the free functions and
 * renza_engine_close.  It covers the failures the engine returns; a CUDA
 * fault inside Strata's library code (kernels, the prompt path, the CPU
 * expert pool's stall check) still ends the process.
 *
 * Threading: one thread at a time per engine, its sessions included (the
 * server's inference lock).  Callbacks run on the calling thread, inside the
 * call that invokes them; a cancel callback usually reads a flag another
 * thread sets.  The pure helpers (tokens, think modes, log) are thread-safe. */

enum {
    RENZA_OK = 0,
    RENZA_ERROR = -1,
    /* sync stopped because the cancel callback returned true. */
    RENZA_INTERRUPTED = -2,
    RENZA_ENGINE_DEAD = -3,
};

/* decode's token argument: sample from the state sync left, feed nothing. */
#define RENZA_TOKEN_NONE (-1)

#define RENZA_VISION_FINGERPRINT_BYTES 32

typedef struct renza_engine renza_engine;
typedef struct renza_session renza_session;

typedef struct {
    int *v;
    int len;
    int cap;
} renza_tokens;

typedef enum {
    RENZA_MODEL_QWEN4 = 1,  /* Qwen3.8-Flash-Next (GGUF architecture qwen4exp) */
} renza_model_family;

/* Same values as ds4_think_mode. */
typedef enum {
    RENZA_THINK_NONE,
    RENZA_THINK_HIGH,
    RENZA_THINK_MAX,
    RENZA_THINK_LOW,
    RENZA_THINK_MEDIUM,
} renza_think_mode;
/* An explicit effort level N in [0, 100] is the mode RENZA_THINK_LEVEL_BASE + N. */
#define RENZA_THINK_LEVEL_BASE 1000

typedef enum {
    RENZA_LOG_DEFAULT,
    RENZA_LOG_PREFILL,
    RENZA_LOG_GENERATION,
    RENZA_LOG_KVCACHE,
    RENZA_LOG_TOOL,
    RENZA_LOG_WARNING,
    RENZA_LOG_TIMING,
    RENZA_LOG_OK,
    RENZA_LOG_ERROR,
} renza_log_type;

/* Numeric fields left at RENZA_OPTION_DEFAULT and NULL strings are not passed
 * to Strata, whose own default applies.  RENZA_OPTION_AUTO is Strata's
 * "auto" for expert_cache and prefill_chunk. */
#define RENZA_OPTION_DEFAULT (-1)
#define RENZA_OPTION_AUTO (-2)

typedef struct {
    const char *pack_dir;        /* --pack (required) */
    const char *native_gguf;     /* --native: first shard (required; also the tokenizer's source) */
    const char *ple_gguf;        /* --ple-gguf */
    const char *expert_profile;  /* --expert-profile */
    int expert_cache;            /* --expert-cache: slots, or AUTO */
    int prefill_chunk;           /* --prefill: tokens, or AUTO */
    int spec;                    /* --spec: the most tokens one decode window holds */
    const char *mtp_dir;         /* --mtp: NULL runs without the MTP drafter */
    double spec_min_p;           /* --spec-min-p */
    int max_context;             /* --max-context */
    const char *kv;              /* --kv: "fp16" or "int8" */
    int kv_resident;             /* --kv-resident */
    const char *mmproj;          /* vision encoder model; non-NULL also enables --vision */
    int vram_reserve_mib;        /* --vram-reserve-mib */
    double pcie_frac;            /* --pcie-frac */
    int pool_workers;            /* --pool-workers */
    int prompt_cache;            /* --prompt-cache: checkpoints kept per conversation */
    /* Further Strata flags, argv style, appended after the fields above (step 1
     * only: the pinned baseline's --adapt-swaps 0 and the like). */
    const char *const *extra_args;
    int n_extra_args;
} renza_engine_options;

/* Sets every numeric field to RENZA_OPTION_DEFAULT and every pointer to NULL. */
void renza_engine_options_init(renza_engine_options *opt);

/* Loads the model and allocates everything that outlives a request (weights,
 * expert cache and pool, drafter, prompt path, verifier).  On RENZA_OK *out
 * is the engine; otherwise *out is NULL.  Errors: RENZA_ERROR for missing or
 * inconsistent options and for any load failure (the message says which).
 * One engine per process: once an open has reached the engine (past the
 * check of the options above), every later open fails, even after
 * renza_engine_close. */
int renza_engine_open(renza_engine **out, const renza_engine_options *opt,
                      char *err, size_t errlen);
/* Frees the engine.  Its sessions must have been freed.  NULL is a no-op. */
void renza_engine_close(renza_engine *e);

/* Queries: fixed after open, valid on a dead engine. */
renza_model_family renza_engine_model_family(const renza_engine *e);
/* Stable id of the model variant, a cache-compatibility key (ds4's variant
 * numbering: 5 is Qwen3.8 Flash Next). */
int renza_engine_model_id(const renza_engine *e);
/* Display name of the loaded model; owned by the engine. */
const char *renza_engine_model_name(const renza_engine *e);
int renza_engine_vocab_size(const renza_engine *e);
/* Width of one vision embedding row (the model's n_embd). */
int renza_engine_embd_dim(const renza_engine *e);
int renza_engine_layer_count(const renza_engine *e);
/* The resolved prompt chunk in tokens (after "auto"); for logging. */
int renza_engine_prefill_chunk(const renza_engine *e);
/* The most tokens one renza_session_decode can return; for logging. */
int renza_engine_spec(const renza_engine *e);
/* True when the engine takes images (mmproj given, or --vision among
 * extra_args): sessions accept image rows.  renza_engine_vision_encode_memory
 * needs the in-process encoder, opened only from the mmproj (RENZA_ERROR with
 * --vision alone). */
bool renza_engine_has_vision(const renza_engine *e);

/* Pure helpers.  Allocation failure exits the process, as in ds4. */
void renza_tokens_push(renza_tokens *tv, int token);
/* Frees the storage and zeroes the vector. */
void renza_tokens_free(renza_tokens *tv);
/* Replaces dst's contents with src's. */
void renza_tokens_copy(renza_tokens *dst, const renza_tokens *src);
bool renza_tokens_starts_with(const renza_tokens *tokens, const renza_tokens *prefix);

/* Tokenizes rendered chat text (special tokens such as <|im_start|> are
 * parsed as such) and APPENDS the ids to out.  len is in bytes; the text may
 * contain NULs. */
void renza_engine_tokenize_chat(const renza_engine *e, const char *text, size_t len,
                                renza_tokens *out);
/* The bytes of one token; the caller frees the result.  *len (when non-NULL)
 * excludes the trailing NUL.  An id outside the vocabulary gives "". */
char *renza_engine_token_text(const renza_engine *e, int token, size_t *len);
int renza_engine_token_eos(const renza_engine *e);
int renza_engine_token_user(const renza_engine *e);
int renza_engine_token_assistant(const renza_engine *e);
/* True when generation stops at token under mode: end of turn or of text,
 * and <think> / </think> when mode has thinking off. */
bool renza_engine_token_is_stop(const renza_engine *e, int token, renza_think_mode mode);

/* Pure helpers. */
bool renza_think_mode_enabled(renza_think_mode mode);
/* The explicit level N of RENZA_THINK_LEVEL_BASE + N, else -1. */
int renza_think_mode_level(renza_think_mode mode);
/* Parses a decimal level in [0, 100] into *out; false (and *out untouched)
 * otherwise. */
bool renza_think_mode_parse_level(const char *text, renza_think_mode *out);
/* "none", "high", "max", "low", "medium", or the level's digits.  The level
 * string is thread-local and valid until the next call on the thread. */
const char *renza_think_mode_name(renza_think_mode mode);
/* The smallest context in which RENZA_THINK_MAX is kept. */
uint32_t renza_think_max_min_context(void);
/* The mode actually used for a session of ctx_size tokens: MAX becomes HIGH
 * below renza_think_max_min_context(). */
renza_think_mode renza_think_mode_for_context(renza_think_mode mode, int ctx_size,
                                              renza_model_family family);
/* The reasoning-effort system instruction the family's chat template adds
 * for mode, or NULL when it adds none.  Static storage. */
const char *renza_think_effort_text(renza_think_mode mode, renza_model_family family);

/* One encoded image.  rows is token_count x embd_dim floats, row-major,
 * heap memory owned by the embedding and released with free() by
 * renza_vision_embedding_free (so a copy made with malloc is also an
 * embedding).  grid_width x grid_height == token_count: the merged patch
 * grid, which gives the rows their M-RoPE positions.  fingerprint
 * identifies the image for the session's prefix checks; key is the 64-bit
 * identity the engine's conversation cache matches on. */
typedef struct {
    float *rows;
    uint32_t token_count;
    uint32_t grid_width;
    uint32_t grid_height;
    uint8_t fingerprint[RENZA_VISION_FINGERPRINT_BYTES];
    uint64_t key;
} renza_vision_embedding;

/* An image placed in a prompt: its rows are the prompt positions
 * [token_start, token_start + embedding.token_count). */
typedef struct {
    uint32_t token_start;
    renza_vision_embedding embedding;
} renza_vision_span;

/* Decodes and encodes an image file held in memory (PNG or JPEG only).
 * Precondition: renza_engine_has_vision(e).  On RENZA_OK *out is a new
 * embedding the caller owns; otherwise *out is zeroed.  Errors: RENZA_ERROR
 * (no encoder, undecodable image), RENZA_ENGINE_DEAD. */
int renza_engine_vision_encode_memory(renza_engine *e, const uint8_t *encoded,
                                      size_t encoded_len, renza_vision_embedding *out,
                                      char *err, size_t errlen);
/* Appends the image's tokens to prompt (vision start, token_count image
 * pads, vision end) and MOVES the embedding into *span (token_start set to
 * the first pad), zeroing *embedding.  Precondition: has_vision.  On error
 * prompt, *span and *embedding are unchanged. */
int renza_prompt_append_vision(const renza_engine *e, renza_tokens *prompt,
                               renza_vision_span *span, renza_vision_embedding *embedding,
                               char *err, size_t errlen);
/* Frees rows and zeroes the embedding.  NULL is a no-op. */
void renza_vision_embedding_free(renza_vision_embedding *embedding);

/* Called inside sync after each prompt chunk reaches the session state:
 * done is the number of prompt tokens the state now holds, total the prompt
 * length; the first call comes before anything is read, with done the
 * tokens resumed from (renza_session_reused).  Only the session's read-only
 * queries may be called from it; they reflect the state at done. */
typedef void (*renza_progress_fn)(void *ud, int done, int total);
/* Polled inside sync between prompt chunks and windows; true stops the sync
 * at the next boundary where the state is consistent. */
typedef bool (*renza_cancel_fn)(void *ud);

/* Sampling for one decode call.  A zeroed struct is greedy with no filters
 * and no penalties. */
typedef struct {
    float temperature;      /* <= 0: greedy */
    int top_k;              /* 0: off (Strata still keeps at most 64 candidates) */
    float top_p;            /* <= 0 or >= 1: off */
    float min_p;            /* 0: off */
    int penalty_last_n;     /* 0: no penalties */
    float penalty_repeat;   /* 0 or 1: off */
    float penalty_freq;
    float penalty_present;
    /* Draws are a function of (seed, position), so a replay from the same
     * prefix with the same seed draws the same tokens.  0: the engine draws a
     * seed from the clock where it applies the sampling, at the first decode
     * after a sync and at a decode whose sampling differs from the previous
     * call's: one seed per request, not per call. */
    uint64_t seed;
} renza_sampling;

/* Creates the engine's session for up to ctx_size tokens (prompt plus
 * generation).  Strata holds one live session per engine.  On RENZA_OK the
 * session is empty: pos 0, checkpoint_valid false.  Errors: RENZA_ERROR
 * (ctx_size outside [1, max_context], a session already exists),
 * RENZA_ENGINE_DEAD. */
int renza_session_create(renza_session **out, renza_engine *e, int ctx_size,
                         char *err, size_t errlen);
/* Frees a session made by renza_session_create.  NULL is a no-op. */
void renza_session_free(renza_session *s);

/* Brings the session to exactly prompt.  The engine resumes from the longest
 * state it holds whose tokens (and images) are a whole prefix of prompt (the
 * live state, else one of its checkpoints) and reads the rest; how much it
 * resumed from is renza_session_reused afterwards.  sync(p1) followed by
 * sync(p2), p2 extending p1, ends with the same tokens as sync(p2), not
 * always the same numbers: the engine reads the tail through decode windows
 * or batched depending on where it resumes.  With Strata's prompt cache off
 * (--prompt-cache 0) the engine keeps no state from one sync to the next:
 * every sync reads from 0.
 * Preconditions: 1 <= prompt->len < ctx; every id in the vocabulary.
 * RENZA_OK: tokens == prompt, pos == prompt->len, checkpoint_valid, and the
 * next decode may pass RENZA_TOKEN_NONE.  RENZA_INTERRUPTED: tokens is the
 * prefix of prompt the state holds (possibly empty), the session keeps only
 * the images wholly inside that prefix, and checkpoint_valid is true only
 * when the engine keeps that prefix as its state (it stopped between two
 * chunks, prompt cache on), so that a sync extending it continues from
 * there; decode needs another sync first.
 * RENZA_ERROR: a refused request (precondition) leaves the session as it
 * was; a failure while reading invalidates it.  RENZA_ENGINE_DEAD. */
int renza_session_sync(renza_session *s, const renza_tokens *prompt,
                       char *err, size_t errlen);
/* sync for a prompt with images.  images (ascending token_start, inside
 * prompt) supply the rows of their positions; their rows need only live for
 * the call.  The session keeps each span's identity (no rows) for the vision
 * queries below. */
int renza_session_sync_multimodal(renza_session *s, const renza_tokens *prompt,
                                  const renza_vision_span *images, size_t image_count,
                                  char *err, size_t errlen);
/* The prompt tokens the last successful or interrupted sync resumed from
 * instead of reading (Strata's RESUME n); 0 before any sync. */
int renza_session_reused(const renza_session *s);

/* Runs one decode window.  token >= 0 is fed first (appended to tokens);
 * RENZA_TOKEN_NONE is allowed only right after a successful sync and samples
 * the token that follows the synced prompt.  Returns n >= 1 sampled tokens in
 * out[0..n), n <= min(budget, cap): out[0] follows token (or the prompt),
 * each out[i] follows out[i-1].  out[0..n-1) are fed by the call (appended
 * to tokens); out[n-1] is not: the server passes it as the next call's
 * token, or stops.  A token of the engine's end-of-generation set ends the
 * run, so it can only be out[n-1].  One call returns more than one token
 * only when the engine drafts (MTP, suffix drafts; renza_engine_spec bounds
 * it); the cancel callback is not polled.  Preconditions: checkpoint_valid;
 * budget >= 1, cap >= 1; room in ctx: with q the position the call reads
 * first (pos for a fed token, pos - 1 for RENZA_TOKEN_NONE, the prompt's last
 * token), the call is refused when q + 2 > ctx, and when
 * q + renza_engine_spec > ctx the window holds the fed token alone (n is 1).
 * A token >= 0 right after a sync: the engine reads the prompt's last token
 * in the first window, so the call syncs prompt + token and decodes from
 * there; same tokens, but it reads again what the engine did not keep (all
 * of it with the prompt cache off), with no progress or cancel callback.
 * Errors (negative): RENZA_ERROR (a refused call leaves the session as it
 * was; a failure invalidates it), RENZA_ENGINE_DEAD. */
int renza_session_decode(renza_session *s, int token, const renza_sampling *sampling,
                         int budget, int *out, int cap, char *err, size_t errlen);

/* Truncates tokens to pos (no-op when pos >= renza_session_pos(s)); a
 * negative pos is 0.
 * The engine cannot restore its state to an earlier position in place, so
 * checkpoint_valid becomes false: the next sync of a prompt starting with
 * the kept tokens rebuilds the state (from the engine's nearest checkpoint).
 * Images not wholly below pos are dropped. */
void renza_session_rewind(renza_session *s, int pos);
/* Empties the session: tokens empty, pos 0, no images, checkpoint_valid
 * false. */
void renza_session_invalidate(renza_session *s);

/* Read-only queries. */
int renza_session_pos(const renza_session *s);
int renza_session_ctx(const renza_session *s);
/* The exact token history the state corresponds to (what sync was given plus
 * what decode fed); pos == tokens->len.  Owned by the session and changed by
 * the mutating calls. */
const renza_tokens *renza_session_tokens(const renza_session *s);
/* The length of the common prefix of tokens and prompt; 0 when
 * checkpoint_valid is false. */
int renza_session_common_prefix(const renza_session *s, const renza_tokens *prompt);
/* True when the state matches tokens and can be continued by decode or
 * extended by sync without a rebuild. */
bool renza_session_checkpoint_valid(const renza_session *s);

/* One callback each; NULL clears it.  They stay set until changed. */
void renza_session_set_progress(renza_session *s, renza_progress_fn fn, void *ud);
void renza_session_set_cancel(renza_session *s, renza_cancel_fn fn, void *ud);

/* Vision identity, pure on the session's spans; the caller checks the token
 * prefix separately.  A session whose checkpoint is not valid never matches.
 *
 * Prefix: every session image equals images[i] in token_start, token_count
 * and fingerprint, and the first extra request image (if any) starts at or
 * after pos. */
bool renza_session_vision_prefix_matches(const renza_session *s,
                                         const renza_vision_span *images, size_t image_count);
/* Prefix, and the same number of images. */
bool renza_session_vision_state_matches(const renza_session *s,
                                        const renza_vision_span *images, size_t image_count);
/* Every session image equals images[i] in token_count and fingerprint,
 * ignoring positions; extra request images allowed. */
bool renza_session_vision_fingerprint_prefix_matches(const renza_session *s,
                                                     const renza_vision_span *images,
                                                     size_t image_count);
/* When images has the session's count and every token_count and fingerprint
 * matches, copies the session's token_start into each span and returns true;
 * otherwise leaves images unchanged.  Says nothing about the text history. */
bool renza_session_rebase_vision_state(const renza_session *s,
                                       renza_vision_span *images, size_t image_count);
/* True while the session holds image-conditioned state or a sync with images
 * is in progress. */
bool renza_session_has_vision_state(const renza_session *s);

/* A session shell holding tokens as a valid checkpoint (ctx 0, no engine),
 * for the server's routing tests: the queries, rewind, invalidate and the
 * vision checks work on it; sync and decode fail.  Free it with
 * renza_session_free_test_checkpoint. */
renza_session *renza_session_new_test_checkpoint(const int *tokens, int n);
void renza_session_free_test_checkpoint(renza_session *s);
/* Replaces the session's image identities with copies of images'
 * (token_start, token_count, grid, fingerprint, key; no rows).  images must
 * be in ascending token_start order, as for sync_multimodal. */
void renza_session_set_test_images(renza_session *s, const renza_vision_span *images,
                                   size_t image_count);
/* The server's canonical-rewrite rule, pure: true unless 0 <= common ==
 * live_len <= canonical_len. */
bool renza_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common);

/* Writes the formatted message to fp, coloured by type when fp is a
 * terminal. */
void renza_log(FILE *fp, renza_log_type type, const char *fmt, ...);
void renza_vlog(FILE *fp, renza_log_type type, const char *fmt, va_list ap);
bool renza_log_is_tty(FILE *fp);

#ifdef __cplusplus
}
#endif

#endif
