/* The ds4.h symbols server/ds4_server.c and server/ds4_kvstore.c link, as
 * thin wrappers over renza.h, so the server stays ds4's code.
 *
 * A NULL ds4_engine answers as ds4.c does with no model loaded (its default
 * shape is DeepSeek V4 Flash, which decides the syntax of NULL-engine request
 * parsing in the server tests).  Functions renza.h dropped fail, return
 * false, or report "not matched", as the server's no-backend paths expect. */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ds4.h"
#include "ds4_distributed.h"
#include "ds4_gpu_args.h"
#include "ds4_tp.h"
#include "renza.h"

/* The think modes and log types are passed through by value. */
typedef char shim_think_values_match[
    (int)DS4_THINK_NONE == (int)RENZA_THINK_NONE && (int)DS4_THINK_HIGH == (int)RENZA_THINK_HIGH &&
    (int)DS4_THINK_MAX == (int)RENZA_THINK_MAX && (int)DS4_THINK_LOW == (int)RENZA_THINK_LOW &&
    (int)DS4_THINK_MEDIUM == (int)RENZA_THINK_MEDIUM &&
    DS4_THINK_LEVEL_BASE == RENZA_THINK_LEVEL_BASE ? 1 : -1];
typedef char shim_log_values_match[
    (int)DS4_LOG_DEFAULT == (int)RENZA_LOG_DEFAULT && (int)DS4_LOG_ERROR == (int)RENZA_LOG_ERROR ? 1 : -1];

static const char SHIM_DROPPED[] = "not supported by the renza engine";

static const char *const NULL_ENGINE_MODEL_NAME = "DeepSeek V4 Flash";
enum {
    NULL_ENGINE_MODEL_ID = 0,
    NULL_ENGINE_N_LAYER = 43,
    NULL_ENGINE_N_EMBD = 4096,
};

static const char DS4_REASONING_EFFORT_MAX_PREFIX[] =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

struct ds4_engine {
    renza_engine *engine;
};

/* tokens_view mirrors the renza session's history (it shares its storage);
 * every wrapper that changes the history refreshes it.
 *
 * ds4 samples and feeds in separate calls; renza_session_decode feeds a token
 * and samples the ones after it.  held is the last token a decode sampled and
 * did not feed: ds4_session_sample returns it, and the server passes it back
 * to eval or eval_speculative, which feed it.  It is not in the history, as
 * ds4's sampled-but-unfed token is not. */
struct ds4_session {
    renza_session *session;
    ds4_tokens tokens_view;
    ds4_session_progress_fn progress;
    void *progress_ud;
    bool has_held;
    int held;
    /* The last sample or eval_speculative's, for ds4_session_eval, which
     * takes none. */
    renza_sampling sampling;
};

static void shim_die_oom(void) {
    fprintf(stderr, "ds4: out of memory\n");
    exit(1);
}

static void *shim_xcalloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p) shim_die_oom();
    return p;
}

static void shim_set_err(char *err, size_t errlen, const char *msg) {
    if (err && errlen) snprintf(err, errlen, "%s", msg);
}

static renza_tokens tokens_to_renza(const ds4_tokens *tv) {
    renza_tokens r = {tv->v, tv->len, tv->cap};
    return r;
}

static void tokens_from_renza(ds4_tokens *tv, renza_tokens r) {
    tv->v = r.v;
    tv->len = r.len;
    tv->cap = r.cap;
}

static const ds4_tokens *refresh_tokens_view(ds4_session *s) {
    tokens_from_renza(&s->tokens_view, *renza_session_tokens(s->session));
    return &s->tokens_view;
}

/* ds4 embeddings carry no cache key; the fingerprint's first bytes stand in. */
static uint64_t fingerprint_key(const uint8_t *fingerprint) {
    uint64_t key = 0;
    for (int i = 0; i < 8; i++) key |= (uint64_t)fingerprint[i] << (8 * i);
    return key;
}

static renza_vision_embedding embedding_to_renza(const ds4_vision_embedding *d) {
    renza_vision_embedding r = {0};
    r.rows = d->data;
    r.token_count = d->token_count;
    r.grid_width = d->grid_width;
    r.grid_height = d->grid_height;
    memcpy(r.fingerprint, d->fingerprint, sizeof(r.fingerprint));
    r.key = fingerprint_key(d->fingerprint);
    return r;
}

static ds4_vision_embedding embedding_from_renza(const renza_vision_embedding *r) {
    ds4_vision_embedding d = {0};
    d.data = r->rows;
    d.token_count = r->token_count;
    d.grid_width = r->grid_width;
    d.grid_height = r->grid_height;
    memcpy(d.fingerprint, r->fingerprint, sizeof(d.fingerprint));
    return d;
}

/* NULL for no images, as renza.h's vision checks expect. */
static renza_vision_span *spans_to_renza(const ds4_vision_span *images, size_t count) {
    if (!images || count == 0) return NULL;
    renza_vision_span *spans = shim_xcalloc(count, sizeof(*spans));
    for (size_t i = 0; i < count; i++) {
        spans[i].token_start = images[i].token_start;
        spans[i].embedding = embedding_to_renza(&images[i].embedding);
    }
    return spans;
}

void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    renza_vlog(fp, (renza_log_type)type, fmt, ap);
    va_end(ap);
}

const char *ds4_backend_name(ds4_backend backend) {
    switch (backend) {
    case DS4_BACKEND_METAL: return "metal";
    case DS4_BACKEND_CUDA:  return "cuda";
    case DS4_BACKEND_CPU:   return "cpu";
    }
    return "unknown";
}

void ds4_tokens_push(ds4_tokens *tv, int token) {
    renza_tokens r = tokens_to_renza(tv);
    renza_tokens_push(&r, token);
    tokens_from_renza(tv, r);
}

void ds4_tokens_free(ds4_tokens *tv) {
    renza_tokens r = tokens_to_renza(tv);
    renza_tokens_free(&r);
    tokens_from_renza(tv, r);
}

void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src) {
    renza_tokens r = tokens_to_renza(dst);
    const renza_tokens from = tokens_to_renza(src);
    renza_tokens_copy(&r, &from);
    tokens_from_renza(dst, r);
}

bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix) {
    const renza_tokens t = tokens_to_renza(tokens);
    const renza_tokens p = tokens_to_renza(prefix);
    return renza_tokens_starts_with(&t, &p);
}

bool ds4_think_mode_enabled(ds4_think_mode mode) {
    return renza_think_mode_enabled((renza_think_mode)mode);
}

int ds4_think_mode_level(ds4_think_mode mode) {
    return renza_think_mode_level((renza_think_mode)mode);
}

bool ds4_think_mode_parse_level(const char *text, ds4_think_mode *out) {
    renza_think_mode mode;
    if (!out || !renza_think_mode_parse_level(text, &mode)) return false;
    *out = (ds4_think_mode)mode;
    return true;
}

const char *ds4_think_mode_name(ds4_think_mode mode) {
    return renza_think_mode_name((renza_think_mode)mode);
}

uint32_t ds4_think_max_min_context(void) {
    return renza_think_max_min_context();
}

ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size) {
    return (ds4_think_mode)renza_think_mode_for_context((renza_think_mode)mode, ctx_size,
                                                        RENZA_MODEL_QWEN4);
}

const char *ds4_qwen4_reasoning_effort_text(ds4_think_mode mode) {
    return renza_think_effort_text((renza_think_mode)mode, RENZA_MODEL_QWEN4);
}

const char *ds4_think_max_prefix(void) {
    return DS4_REASONING_EFFORT_MAX_PREFIX;
}

const char *ds4_glm_reasoning_effort_text(ds4_think_mode mode) {
    switch (mode) {
    case DS4_THINK_LOW:
    case DS4_THINK_MEDIUM:
    case DS4_THINK_HIGH: return "Reasoning Effort: High";
    case DS4_THINK_MAX:  return "Reasoning Effort: Max";
    case DS4_THINK_NONE: return NULL;
    }
    return NULL;
}

const char *ds4_deepseek41_reasoning_effort_text(ds4_think_mode mode) {
    int level = ds4_think_mode_level(mode);
    if (mode == DS4_THINK_HIGH) level = 75;
    if (mode == DS4_THINK_MAX) level = 100;
    if (level <= 0) return NULL;
    static __thread char text[128];
    snprintf(text, sizeof(text), "Reasoning Effort: %d (range 1-100, the higher the value, the more thorough the reasoning)\n\n", level);
    return text;
}

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt) {
    if (out) *out = NULL;
    if (!out || !opt) {
        fprintf(stderr, "ds4: missing engine options or output\n");
        return 1;
    }
    renza_engine_options ro;
    renza_engine_options_init(&ro);
    ro.native_gguf = opt->model_path;
    ro.mtp_dir = opt->mtp_path;
    ro.mmproj = opt->vision_path;
    ro.expert_profile = opt->expert_profile_path;
    ro.pack_dir = opt->pack_dir;
    ro.extra_args = opt->extra_args;
    ro.n_extra_args = opt->n_extra_args;
    if (opt->prefill_chunk > 0) ro.prefill_chunk = (int)opt->prefill_chunk;
    if (opt->context_size > 0) ro.max_context = opt->context_size;

    char err[256] = {0};
    renza_engine *engine = NULL;
    if (renza_engine_open(&engine, &ro, err, sizeof(err)) != RENZA_OK) {
        fprintf(stderr, "ds4: %s\n", err);
        return 1;
    }
    ds4_engine *e = shim_xcalloc(1, sizeof(*e));
    e->engine = engine;
    *out = e;
    return 0;
}

int ds4_engine_create_with_gpu_config(ds4_engine **out,
                                       const ds4_engine_options *opt,
                                       const struct ds4_gpu_config *gpu_cfg) {
    (void)gpu_cfg;
    return ds4_engine_open(out, opt);
}

void ds4_engine_close(ds4_engine *e) {
    if (!e) return;
    renza_engine_close(e->engine);
    free(e);
}

int ds4_engine_model_id(ds4_engine *e) {
    return e ? renza_engine_model_id(e->engine) : NULL_ENGINE_MODEL_ID;
}

const char *ds4_engine_model_name(ds4_engine *e) {
    return e ? renza_engine_model_name(e->engine) : NULL_ENGINE_MODEL_NAME;
}

bool ds4_engine_is_qwen4(ds4_engine *e) {
    return e && renza_engine_model_family(e->engine) == RENZA_MODEL_QWEN4;
}

bool ds4_engine_is_glm_dsa(ds4_engine *e) {
    (void)e;
    return false;
}

bool ds4_engine_is_glm53(ds4_engine *e) {
    (void)e;
    return false;
}

bool ds4_engine_is_deepseek41(ds4_engine *e) {
    (void)e;
    return false;
}

int ds4_engine_layer_count(ds4_engine *e) {
    return e ? renza_engine_layer_count(e->engine) : NULL_ENGINE_N_LAYER;
}

int ds4_engine_embd_dim(ds4_engine *e) {
    return e ? renza_engine_embd_dim(e->engine) : NULL_ENGINE_N_EMBD;
}

int ds4_engine_vocab_size(ds4_engine *e) {
    return e ? renza_engine_vocab_size(e->engine) : 0;
}

uint64_t ds4_engine_model_bytes(ds4_engine *e) {
    (void)e;
    return 0;
}

uint32_t ds4_engine_prefill_chunk(ds4_engine *e) {
    return e ? (uint32_t)renza_engine_prefill_chunk(e->engine) : 0;
}

int ds4_engine_routed_quant_bits(ds4_engine *e) {
    (void)e;
    return 0;
}

bool ds4_engine_has_vision(ds4_engine *e) {
    return e && renza_engine_has_vision(e->engine);
}

/* Above 1 the server takes its eval_speculative path, which returns a whole
 * decode window. */
int ds4_engine_mtp_draft_tokens(ds4_engine *e) {
    return e ? renza_engine_spec(e->engine) : 0;
}

bool ds4_engine_mtp_exact_sampling(ds4_engine *e) {
    (void)e;
    return false;
}

/* The DeepSeek V4 Flash branch of ds4.c's schedule: every layer fires ATTN then FFN. */
void ds4_engine_tp_gate_schedule(ds4_engine *e,
                                 uint32_t *start,
                                 uint32_t *step,
                                 uint32_t *per_token,
                                 uint64_t mask[DS4_TP_GATE_MASK_WORDS]) {
    memset(mask, 0, sizeof(uint64_t) * DS4_TP_GATE_MASK_WORDS);
    *start = 0;
    *step = 1;
    *per_token = (uint32_t)ds4_engine_layer_count(e) * DS4_TP_GATES_PER_LAYER;
}

int ds4_engine_tp_bind(ds4_engine *e, struct ds4_tp *tp, char *err, size_t errlen) {
    (void)e;
    (void)tp;
    shim_set_err(err, errlen, SHIM_DROPPED);
    return 0;
}

ds4_context_memory ds4_context_memory_estimate_with_prefill_mode(
        ds4_backend backend,
        int ctx_size,
        uint32_t prefill_chunk,
        bool ssd_streaming) {
    (void)backend;
    (void)ctx_size;
    (void)prefill_chunk;
    (void)ssd_streaming;
    ds4_context_memory m = {0};
    return m;
}

void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out) {
    if (!e) return;
    renza_tokens r = tokens_to_renza(out);
    renza_engine_tokenize_chat(e->engine, text, text ? strlen(text) : 0, &r);
    tokens_from_renza(out, r);
}

/* Same result as ds4.c for a token outside the vocabulary. */
char *ds4_token_text(ds4_engine *e, int token, size_t *len) {
    if (e) return renza_engine_token_text(e->engine, token, len);
    if (len) *len = 0;
    char *out = shim_xcalloc(1, 1);
    return out;
}

int ds4_token_eos(ds4_engine *e) {
    return e ? renza_engine_token_eos(e->engine) : -1;
}

int ds4_token_user(ds4_engine *e) {
    return e ? renza_engine_token_user(e->engine) : -1;
}

int ds4_token_assistant(ds4_engine *e) {
    return e ? renza_engine_token_assistant(e->engine) : -1;
}

bool ds4_token_is_stop_for_think_mode(ds4_engine *e, int token, ds4_think_mode mode) {
    return e && renza_engine_token_is_stop(e->engine, token, (renza_think_mode)mode);
}

int ds4_engine_vision_encode_memory(ds4_engine *e,
                                    const uint8_t *encoded,
                                    size_t encoded_len,
                                    ds4_vision_embedding *out,
                                    char *error,
                                    size_t error_cap) {
    if (!e || !out) {
        shim_set_err(error, error_cap, "invalid vision input");
        return 0;
    }
    renza_vision_embedding r;
    if (renza_engine_vision_encode_memory(e->engine, encoded, encoded_len, &r,
                                          error, error_cap) != RENZA_OK) return 0;
    *out = embedding_from_renza(&r);
    return 1;
}

int ds4_prompt_append_vision(ds4_engine *e,
                             ds4_tokens *tokens,
                             ds4_vision_span *span,
                             ds4_vision_embedding *embedding,
                             char *error,
                             size_t error_cap) {
    if (!e || !tokens || !span || !embedding) {
        shim_set_err(error, error_cap, "invalid vision prompt input");
        return 0;
    }
    renza_tokens prompt = tokens_to_renza(tokens);
    renza_vision_embedding moved = embedding_to_renza(embedding);
    renza_vision_span placed;
    if (renza_prompt_append_vision(e->engine, &prompt, &placed, &moved,
                                   error, error_cap) != RENZA_OK) return 0;
    tokens_from_renza(tokens, prompt);
    memset(span, 0, sizeof(*span));
    span->token_start = placed.token_start;
    span->embedding = embedding_from_renza(&placed.embedding);
    memset(embedding, 0, sizeof(*embedding));
    return 1;
}

void ds4_vision_embedding_free(ds4_vision_embedding *embedding) {
    if (!embedding) return;
    renza_vision_embedding r = embedding_to_renza(embedding);
    renza_vision_embedding_free(&r);
    memset(embedding, 0, sizeof(*embedding));
}

static ds4_session *wrap_session(renza_session *session) {
    ds4_session *s = shim_xcalloc(1, sizeof(*s));
    s->session = session;
    refresh_tokens_view(s);
    return s;
}

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size) {
    if (out) *out = NULL;
    char err[256] = "no engine";
    renza_session *session = NULL;
    if (!out || !e ||
        renza_session_create(&session, e->engine, ctx_size, err, sizeof(err)) != RENZA_OK) {
        fprintf(stderr, "ds4: %s\n", err);
        return 1;
    }
    *out = wrap_session(session);
    return 0;
}

void ds4_session_free(ds4_session *s) {
    if (!s) return;
    renza_session_free(s->session);
    free(s);
}

ds4_session *ds4_session_new_test_checkpoint(const int *tokens, int n) {
    return wrap_session(renza_session_new_test_checkpoint(tokens, n));
}

void ds4_session_free_test_checkpoint(ds4_session *s) {
    if (!s) return;
    renza_session_free_test_checkpoint(s->session);
    free(s);
}

void ds4_session_set_test_images(ds4_session *s,
                                 const ds4_vision_span *images, size_t n) {
    if (!s) return;
    renza_vision_span *spans = spans_to_renza(images, n);
    renza_session_set_test_images(s->session, spans, spans ? n : 0);
    free(spans);
}

static void forward_progress(void *ud, int done, int total) {
    ds4_session *s = ud;
    s->progress(s->progress_ud, "prefill_chunk", done, total);
}

void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->progress = fn;
    s->progress_ud = ud;
    renza_session_set_progress(s->session, fn ? forward_progress : NULL, s);
}

/* renza reports chunk progress only; the server's keepalive runs on it. */
void ds4_session_set_display_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
    (void)s;
    (void)fn;
    (void)ud;
}

void ds4_session_set_cancel(ds4_session *s, ds4_session_cancel_fn fn, void *ud) {
    if (!s) return;
    renza_session_set_cancel(s->session, fn, ud);
}

int ds4_session_pos(ds4_session *s) {
    return renza_session_pos(s->session);
}

int ds4_session_ctx(ds4_session *s) {
    return renza_session_ctx(s->session);
}

const ds4_tokens *ds4_session_tokens(ds4_session *s) {
    return s ? refresh_tokens_view(s) : NULL;
}

bool ds4_session_checkpoint_valid(const ds4_session *s) {
    return s && renza_session_checkpoint_valid(s->session);
}

int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt) {
    const renza_tokens p = tokens_to_renza(prompt);
    return renza_session_common_prefix(s->session, &p);
}

void ds4_session_invalidate(ds4_session *s) {
    if (!s) return;
    s->has_held = false;
    renza_session_invalidate(s->session);
    refresh_tokens_view(s);
}

void ds4_session_rewind(ds4_session *s, int pos) {
    if (!s) return;
    s->has_held = false;
    renza_session_rewind(s->session, pos);
    refresh_tokens_view(s);
}

bool ds4_session_rewind_speculative(ds4_session *s, int pos) {
    (void)s;
    (void)pos;
    return false;
}

bool ds4_session_vision_prefix_matches(const ds4_session *s,
                                       const ds4_vision_span *images,
                                       size_t image_count) {
    if (!s) return false;
    renza_vision_span *spans = spans_to_renza(images, image_count);
    const bool ok = renza_session_vision_prefix_matches(s->session, spans, image_count);
    free(spans);
    return ok;
}

bool ds4_session_vision_state_matches(const ds4_session *s,
                                      const ds4_vision_span *images,
                                      size_t image_count) {
    if (!s) return false;
    renza_vision_span *spans = spans_to_renza(images, image_count);
    const bool ok = renza_session_vision_state_matches(s->session, spans, image_count);
    free(spans);
    return ok;
}

bool ds4_session_vision_fingerprint_prefix_matches(const ds4_session *s,
                                                   const ds4_vision_span *images,
                                                   size_t image_count) {
    if (!s) return false;
    renza_vision_span *spans = spans_to_renza(images, image_count);
    const bool ok = renza_session_vision_fingerprint_prefix_matches(s->session, spans, image_count);
    free(spans);
    return ok;
}

bool ds4_session_rebase_vision_state(const ds4_session *s,
                                     ds4_vision_span *images, size_t image_count) {
    if (!s) return false;
    renza_vision_span *spans = spans_to_renza(images, image_count);
    const bool ok = renza_session_rebase_vision_state(s->session, spans, image_count);
    for (size_t i = 0; ok && i < image_count; i++) images[i].token_start = spans[i].token_start;
    free(spans);
    return ok;
}

bool ds4_session_has_vision_state(const ds4_session *s) {
    return s && renza_session_has_vision_state(s->session);
}

bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common) {
    return renza_session_rewrite_requires_rebuild(live_len, canonical_len, common);
}

ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen) {
    (void)s;
    (void)prompt;
    (void)common;
    shim_set_err(err, errlen, SHIM_DROPPED);
    return DS4_SESSION_REWRITE_ERROR;
}

static int sync_status_to_ds4(int rc) {
    if (rc == RENZA_OK) return 0;
    return rc == RENZA_INTERRUPTED ? DS4_SESSION_SYNC_INTERRUPTED : 1;
}

int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    return ds4_session_sync_multimodal(s, prompt, NULL, 0, err, errlen);
}

int ds4_session_sync_multimodal(ds4_session *s,
                                const ds4_tokens *prompt,
                                const ds4_vision_span *images,
                                size_t image_count,
                                char *err,
                                size_t errlen) {
    if (!s || !prompt) {
        shim_set_err(err, errlen, "missing session or prompt");
        return 1;
    }
    s->has_held = false;
    const renza_tokens p = tokens_to_renza(prompt);
    renza_vision_span *spans = spans_to_renza(images, image_count);
    const int rc = renza_session_sync_multimodal(s->session, &p, spans, spans ? image_count : 0,
                                                 err, errlen);
    free(spans);
    refresh_tokens_view(s);
    return sync_status_to_ds4(rc);
}

/* The server never advances *rng (ds4's sampler did), so the seed is the
 * request's for every window. */
static renza_sampling sampling_of(float temperature, int top_k, float top_p, float min_p,
                                  const uint64_t *rng) {
    renza_sampling sampling = {0};
    sampling.temperature = temperature;
    sampling.top_k = top_k;
    sampling.top_p = top_p;
    sampling.min_p = min_p;
    sampling.seed = rng ? *rng : 0;
    return sampling;
}

/* Feeds token (RENZA_TOKEN_NONE: nothing) and holds the window's last
 * sampled token.  Returns the window's n, or -1.
 *
 * renza samples after every token it feeds, so it refuses one at the last
 * context position (pos ctx - 1), which ds4 feeds.  There nothing can follow:
 * the token is accepted unfed and the window is the token alone. */
static int decode_window(ds4_session *s, int token, int budget, int *out, int cap,
                         char *err, size_t errlen) {
    s->has_held = false;
    if (token >= 0 && renza_session_checkpoint_valid(s->session) &&
        renza_session_pos(s->session) == renza_session_ctx(s->session) - 1) return 1;
    const int n = renza_session_decode(s->session, token, &s->sampling, budget, out, cap,
                                       err, errlen);
    refresh_tokens_view(s);
    if (n < 1) return -1;
    s->held = out[n - 1];
    s->has_held = true;
    return n;
}

int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen) {
    if (!s) {
        shim_set_err(err, errlen, "missing session");
        return 1;
    }
    int next;
    return decode_window(s, token, 1, &next, 1, err, errlen) < 0 ? 1 : 0;
}

/* Right after a sync nothing is held: the window feeds nothing and samples
 * the token after the prompt. */
int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng) {
    if (!s) return -1;
    s->sampling = sampling_of(temperature, top_k, top_p, min_p, rng);
    if (s->has_held) return s->held;
    int token;
    char err[256] = {0};
    if (decode_window(s, RENZA_TOKEN_NONE, 1, &token, 1, err, sizeof(err)) < 0) {
        fprintf(stderr, "ds4: %s\n", err);
        return -1;
    }
    return token;
}

int ds4_session_argmax_ignoring_eos(ds4_session *s, ds4_think_mode think_mode) {
    (void)s;
    (void)think_mode;
    return -1;
}

/* accepted = [first_token, the window's tokens but the last], all fed, at
 * most max_tokens of them as in ds4 (whose count includes first_token); the
 * window's last token is held.  The engine ends windows at its own
 * end-of-generation set, so eos_token is not needed. */
int ds4_session_eval_speculative(ds4_session *s, int first_token,
                                 int max_tokens, int eos_token,
                                 float temperature, int top_k,
                                 float top_p, float min_p, uint64_t *rng,
                                 int *accepted, int accepted_cap,
                                 char *err, size_t errlen) {
    (void)eos_token;
    if (!s || !accepted || max_tokens <= 0 || accepted_cap <= 0) return 0;
    s->sampling = sampling_of(temperature, top_k, top_p, min_p, rng);
    accepted[0] = first_token;
    if (accepted_cap == 1) return ds4_session_eval(s, first_token, err, errlen) ? -1 : 1;
    return decode_window(s, first_token, max_tokens, accepted + 1, accepted_cap - 1,
                         err, errlen);
}

int ds4_session_eval_speculative_argmax_ignoring_eos(
        ds4_session *s, int first_token, int max_tokens, int eos_token,
        ds4_think_mode think_mode,
        int *accepted, int accepted_cap, char *err, size_t errlen) {
    (void)s;
    (void)first_token;
    (void)max_tokens;
    (void)eos_token;
    (void)think_mode;
    (void)accepted;
    (void)accepted_cap;
    shim_set_err(err, errlen, SHIM_DROPPED);
    return -1;
}

int ds4_sessions_eval_batch(ds4_decode_item *items, int count,
                            char *err, size_t errlen) {
    (void)items;
    (void)count;
    shim_set_err(err, errlen, SHIM_DROPPED);
    return 1;
}

int ds4_sessions_eval_batch_speculative_argmax(ds4_decode_item *items, int count,
                                               int (*accepted)[2], int *n_accepted,
                                               char *err, size_t errlen) {
    (void)items;
    (void)count;
    (void)accepted;
    (void)n_accepted;
    shim_set_err(err, errlen, SHIM_DROPPED);
    return 1;
}

int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out,
                              char *err, size_t errlen) {
    (void)s;
    if (out) memset(out, 0, sizeof(*out));
    shim_set_err(err, errlen, SHIM_DROPPED);
    return 1;
}

int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen) {
    (void)payload;
    (void)fp;
    shim_set_err(err, errlen, SHIM_DROPPED);
    return 1;
}

void ds4_session_payload_file_free(ds4_session_payload_file *payload) {
    if (!payload) return;
    if (payload->path) {
        unlink(payload->path);
        free(payload->path);
    }
    memset(payload, 0, sizeof(*payload));
}

int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) {
    (void)s;
    (void)fp;
    (void)payload_bytes;
    shim_set_err(err, errlen, SHIM_DROPPED);
    return 1;
}

int ds4_gpu_args_probe_auto_cuda(const int      *device_filter,
                                  int             filter_len,
                                  ds4_gpu_config *out,
                                  size_t          safety_margin_bytes,
                                  char           *errbuf,
                                  size_t          errbuflen) {
    (void)device_filter;
    (void)filter_len;
    (void)out;
    (void)safety_margin_bytes;
    shim_set_err(errbuf, errbuflen, SHIM_DROPPED);
    return 1;
}

/* Distributed and TP options are not recognised, so the server rejects them as
 * unknown arguments; with none set, the option checks pass as in ds4. */
ds4_dist_cli_parse_result ds4_dist_parse_cli_arg(
        const char *arg,
        int *index,
        int argc,
        char **argv,
        ds4_dist_options *opt,
        char *err,
        size_t errlen) {
    (void)arg;
    (void)index;
    (void)argc;
    (void)argv;
    (void)opt;
    (void)err;
    (void)errlen;
    return DS4_DIST_CLI_NOT_MATCHED;
}

int ds4_dist_prepare_engine_options(
        const ds4_dist_options *opt,
        ds4_engine_options *engine,
        char *err,
        size_t errlen) {
    (void)opt;
    (void)engine;
    (void)err;
    (void)errlen;
    return 0;
}

int ds4_dist_run(ds4_engine *engine, const ds4_dist_options *opt, const ds4_dist_generation_options *gen) {
    (void)engine;
    (void)opt;
    (void)gen;
    fprintf(stderr, "ds4: distributed inference %s\n", SHIM_DROPPED);
    return 1;
}

int ds4_tp_parse_cli_arg(
        const char *arg,
        int *index,
        int argc,
        char **argv,
        ds4_tp_options *opt,
        char *err,
        size_t errlen) {
    (void)arg;
    (void)index;
    (void)argc;
    (void)argv;
    (void)opt;
    (void)err;
    (void)errlen;
    return DS4_TP_CLI_NOT_MATCHED;
}

int ds4_tp_adopt_distributed_options(
        ds4_tp_options *tp,
        ds4_distributed_options *dist,
        char *err,
        size_t errlen) {
    (void)tp;
    (void)dist;
    (void)err;
    (void)errlen;
    return 1;
}

int ds4_tp_validate_engine_options(
        const ds4_engine_options *opt,
        char *err,
        size_t errlen) {
    (void)opt;
    (void)err;
    (void)errlen;
    return 1;
}

int ds4_tp_create(
        ds4_tp **out,
        const ds4_tp_options *opt,
        const ds4_tp_identity *id,
        char *err,
        size_t errlen) {
    (void)opt;
    (void)id;
    if (out) *out = NULL;
    shim_set_err(err, errlen, SHIM_DROPPED);
    return 0;
}

void ds4_tp_free(ds4_tp *tp) {
    (void)tp;
}

int ds4_tp_send_stop(ds4_tp *tp) {
    (void)tp;
    return 0;
}
