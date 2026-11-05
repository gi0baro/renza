/* renza.h's pure helpers, the text side and the session bookkeeping, shared
 * by the stub engine and the Strata adapter (engine/renza_common.h). */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "renza_common.h"
#include "tokenizer.h"

static const char MODEL_ARCH[] = "qwen4exp";

/* ds4's variant id for Qwen3.8 Flash Next (DS4_VARIANT_QWEN4_EXP). */
enum { MODEL_ID_QWEN4 = 5 };

/* DeepSeek recommends Think Max only with at least a 384K-token context window.
 * Below that size we keep ordinary thinking to avoid injecting a prompt that
 * asks for a reasoning budget the allocated context is not meant to hold. */
#define THINK_MAX_MIN_CONTEXT 393216u

static const char *const QWEN4_REASONING_XHIGH =
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer.";
static const char *const QWEN4_REASONING_LOW =
    "Reasoning effort is set to low. Keep your thinking brief and focused, "
    "moving directly to the conclusion without unnecessary elaboration.";

struct renza_engine_text {
    renza_model model;
    renza_vocab *vocab;
    char *model_name;
};

static void die_oom(void) {
    fprintf(stderr, "renza: out of memory\n");
    exit(1);
}

void *renza_xcalloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p) die_oom();
    return p;
}

void renza_set_err(char *err, size_t errlen, const char *msg) {
    if (err && errlen) snprintf(err, errlen, "%s", msg);
}

static const char *log_color_code(renza_log_type type) {
    switch (type) {
    case RENZA_LOG_PREFILL:
    case RENZA_LOG_TIMING:
        return "\x1b[36m";
    case RENZA_LOG_GENERATION:
    case RENZA_LOG_OK:
        return "\x1b[32m";
    case RENZA_LOG_KVCACHE:
        return "\x1b[33m";
    case RENZA_LOG_TOOL:
        return "\x1b[90m";
    case RENZA_LOG_WARNING:
        return "\x1b[38;5;208m";
    case RENZA_LOG_ERROR:
        return "\x1b[31m";
    default:
        return "";
    }
}

bool renza_log_is_tty(FILE *fp) {
    int fd = fileno(fp);
    return fd >= 0 && isatty(fd) != 0;
}

void renza_vlog(FILE *fp, renza_log_type type, const char *fmt, va_list ap) {
    const bool colorize = type != RENZA_LOG_DEFAULT && renza_log_is_tty(fp);
    if (colorize) fputs(log_color_code(type), fp);
    vfprintf(fp, fmt, ap);
    if (colorize) fputs("\x1b[0m", fp);
}

void renza_log(FILE *fp, renza_log_type type, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    renza_vlog(fp, type, fmt, ap);
    va_end(ap);
}

void renza_tokens_copy(renza_tokens *dst, const renza_tokens *src) {
    dst->len = 0;
    for (int i = 0; i < src->len; i++) renza_tokens_push(dst, src->v[i]);
}

bool renza_tokens_starts_with(const renza_tokens *tokens, const renza_tokens *prefix) {
    if (prefix->len > tokens->len) return false;
    for (int i = 0; i < prefix->len; i++) {
        if (tokens->v[i] != prefix->v[i]) return false;
    }
    return true;
}

bool renza_think_mode_enabled(renza_think_mode mode) {
    return mode == RENZA_THINK_LOW || mode == RENZA_THINK_MEDIUM || mode == RENZA_THINK_HIGH ||
           mode == RENZA_THINK_MAX || renza_think_mode_level(mode) > 0;
}

int renza_think_mode_level(renza_think_mode mode) {
    if ((unsigned)mode < RENZA_THINK_LEVEL_BASE ||
        (unsigned)mode > RENZA_THINK_LEVEL_BASE + 100u) return -1;
    return (int)mode - RENZA_THINK_LEVEL_BASE;
}

bool renza_think_mode_parse_level(const char *text, renza_think_mode *out) {
    if (!text || !out || !text[0]) return false;
    unsigned level = 0;
    for (const char *p = text; *p; p++) {
        if (*p < '0' || *p > '9' || level > 100) return false;
        level = level * 10u + (unsigned)(*p - '0');
    }
    if (level > 100) return false;
    *out = (renza_think_mode)(RENZA_THINK_LEVEL_BASE + level);
    return true;
}

const char *renza_think_mode_name(renza_think_mode mode) {
    const int level = renza_think_mode_level(mode);
    if (level >= 0) {
        static __thread char name[12];
        snprintf(name, sizeof(name), "%d", level);
        return name;
    }
    switch (mode) {
    case RENZA_THINK_NONE: return "none";
    case RENZA_THINK_HIGH: return "high";
    case RENZA_THINK_MAX:  return "max";
    case RENZA_THINK_LOW:  return "low";
    case RENZA_THINK_MEDIUM: return "medium";
    }
    return "unknown";
}

uint32_t renza_think_max_min_context(void) {
    return THINK_MAX_MIN_CONTEXT;
}

/* ds4 keeps MAX at any size only for DeepSeek V4.1, which is not a renza family. */
renza_think_mode renza_think_mode_for_context(renza_think_mode mode, int ctx_size,
                                              renza_model_family family) {
    (void)family;
    if (mode == RENZA_THINK_MAX && (uint32_t)(ctx_size > 0 ? ctx_size : 0) < THINK_MAX_MIN_CONTEXT) {
        return RENZA_THINK_HIGH;
    }
    return mode;
}

/* The official template's default effort is xhigh; medium adds no text. */
const char *renza_think_effort_text(renza_think_mode mode, renza_model_family family) {
    if (family != RENZA_MODEL_QWEN4) return NULL;
    switch (mode) {
    case RENZA_THINK_HIGH:
    case RENZA_THINK_MAX:  return QWEN4_REASONING_XHIGH;
    case RENZA_THINK_LOW:  return QWEN4_REASONING_LOW;
    case RENZA_THINK_MEDIUM:
    case RENZA_THINK_NONE: return NULL;
    }
    return NULL;
}

void renza_engine_options_init(renza_engine_options *opt) {
    memset(opt, 0, sizeof(*opt));
    opt->expert_cache = RENZA_OPTION_DEFAULT;
    opt->prefill_chunk = RENZA_OPTION_DEFAULT;
    opt->spec = RENZA_OPTION_DEFAULT;
    opt->spec_min_p = RENZA_OPTION_DEFAULT;
    opt->max_context = RENZA_OPTION_DEFAULT;
    opt->kv_resident = RENZA_OPTION_DEFAULT;
    opt->vram_reserve_mib = RENZA_OPTION_DEFAULT;
    opt->pcie_frac = RENZA_OPTION_DEFAULT;
    opt->pool_workers = RENZA_OPTION_DEFAULT;
    opt->prompt_cache = RENZA_OPTION_DEFAULT;
}

static bool gguf_string_is(const renza_model *m, const char *key, const char *want) {
    renza_str s;
    return renza_model_get_string(m, key, &s) &&
           s.len == strlen(want) && memcmp(s.ptr, want, s.len) == 0;
}

static char *gguf_model_name(const renza_model *m) {
    renza_str s;
    if (!renza_model_get_string(m, "general.name", &s)) {
        s.ptr = MODEL_ARCH;
        s.len = sizeof(MODEL_ARCH) - 1;
    }
    char *name = renza_xcalloc(1, (size_t)s.len + 1);
    memcpy(name, s.ptr, (size_t)s.len);
    return name;
}

renza_engine_text *renza_engine_text_open(const char *native_gguf, char *err, size_t errlen) {
    renza_engine_text *t = renza_xcalloc(1, sizeof(*t));
    if (!renza_model_open(&t->model, native_gguf, err, errlen)) {
        free(t);
        return NULL;
    }
    if (!gguf_string_is(&t->model, "general.architecture", MODEL_ARCH)) {
        renza_set_err(err, errlen, "the native GGUF is not a qwen4exp model");
        renza_model_close(&t->model);
        free(t);
        return NULL;
    }
    t->vocab = renza_vocab_create(&t->model, err, errlen);
    if (!t->vocab) {
        renza_model_close(&t->model);
        free(t);
        return NULL;
    }
    t->model_name = gguf_model_name(&t->model);
    return t;
}

void renza_engine_text_close(renza_engine_text *t) {
    if (!t) return;
    renza_vocab_free(t->vocab);
    renza_model_close(&t->model);
    free(t->model_name);
    free(t);
}

int renza_engine_text_vocab_size(const renza_engine_text *t) {
    return renza_vocab_size(t->vocab);
}

int renza_engine_text_arch_u32(const renza_engine_text *t, const char *name) {
    char key[64];
    uint32_t v = 0;
    snprintf(key, sizeof(key), "%s.%s", MODEL_ARCH, name);
    return renza_model_get_u32(&t->model, key, &v) ? (int)v : 0;
}

renza_model_family renza_engine_model_family(const renza_engine *e) {
    (void)e;
    return RENZA_MODEL_QWEN4;
}

int renza_engine_model_id(const renza_engine *e) {
    (void)e;
    return MODEL_ID_QWEN4;
}

const char *renza_engine_model_name(const renza_engine *e) {
    return renza_engine_text_of(e)->model_name;
}

void renza_engine_tokenize_chat(const renza_engine *e, const char *text, size_t len,
                                renza_tokens *out) {
    renza_tokenize_rendered_chat(renza_engine_text_of(e)->vocab, text, len, out);
}

char *renza_engine_token_text(const renza_engine *e, int token, size_t *len) {
    return renza_token_text(renza_engine_text_of(e)->vocab, token, len);
}

int renza_engine_token_eos(const renza_engine *e) {
    return renza_token_eos(renza_engine_text_of(e)->vocab);
}

int renza_engine_token_user(const renza_engine *e) {
    return renza_token_user(renza_engine_text_of(e)->vocab);
}

int renza_engine_token_assistant(const renza_engine *e) {
    return renza_token_assistant(renza_engine_text_of(e)->vocab);
}

bool renza_engine_token_is_stop(const renza_engine *e, int token, renza_think_mode mode) {
    return renza_token_is_stop_for_thinking(renza_engine_text_of(e)->vocab, token,
                                            renza_think_mode_enabled(mode));
}

void renza_vision_embedding_free(renza_vision_embedding *embedding) {
    if (!embedding) return;
    free(embedding->rows);
    memset(embedding, 0, sizeof(*embedding));
}

renza_session *renza_session_book_new(void) {
    return renza_xcalloc(1, sizeof(renza_session));
}

static void session_clear_images(renza_session *s) {
    free(s->images);
    s->images = NULL;
    s->image_count = 0;
}

void renza_session_book_free(renza_session *s) {
    if (!s) return;
    free(s->tokens.v);
    session_clear_images(s);
    free(s);
}

static bool image_below(const renza_vision_span *image, int pos) {
    return (uint64_t)image->token_start + image->embedding.token_count <= (uint64_t)(pos > 0 ? pos : 0);
}

static void session_copy_images(renza_session *s, const renza_vision_span *images,
                                size_t image_count) {
    session_clear_images(s);
    if (image_count == 0 || !images) return;
    s->images = renza_xcalloc(image_count, sizeof(*s->images));
    for (size_t i = 0; i < image_count; i++) {
        s->images[i] = images[i];
        s->images[i].embedding.rows = NULL;
    }
    s->image_count = image_count;
}

void renza_session_book_assign(renza_session *s, const int *ids, int n,
                               const renza_vision_span *images, size_t image_count) {
    s->tokens.len = 0;
    for (int i = 0; i < n; i++) renza_tokens_push(&s->tokens, ids[i]);
    size_t kept = 0;
    while (kept < image_count && image_below(&images[kept], n)) kept++;
    session_copy_images(s, images, kept);
}

bool renza_session_book_rewind(renza_session *s, int pos) {
    if (!s) return false;
    if (pos < 0) pos = 0;
    if (pos >= s->tokens.len) return false;
    s->tokens.len = pos;
    s->checkpoint_valid = false;
    size_t kept = 0;
    while (kept < s->image_count && image_below(&s->images[kept], pos)) kept++;
    s->image_count = kept;
    return true;
}

void renza_session_book_invalidate(renza_session *s) {
    if (!s) return;
    s->checkpoint_valid = false;
    s->tokens.len = 0;
    session_clear_images(s);
}

int renza_session_reused(const renza_session *s) {
    return s->reused;
}

int renza_session_pos(const renza_session *s) {
    return s->tokens.len;
}

int renza_session_ctx(const renza_session *s) {
    return s->ctx;
}

const renza_tokens *renza_session_tokens(const renza_session *s) {
    return &s->tokens;
}

int renza_session_common_prefix(const renza_session *s, const renza_tokens *prompt) {
    if (!s->checkpoint_valid) return 0;
    const int n = s->tokens.len < prompt->len ? s->tokens.len : prompt->len;
    int i = 0;
    while (i < n && s->tokens.v[i] == prompt->v[i]) i++;
    return i;
}

bool renza_session_checkpoint_valid(const renza_session *s) {
    return s->checkpoint_valid;
}

void renza_session_set_progress(renza_session *s, renza_progress_fn fn, void *ud) {
    s->progress = fn;
    s->progress_ud = ud;
}

void renza_session_set_cancel(renza_session *s, renza_cancel_fn fn, void *ud) {
    s->cancel = fn;
    s->cancel_ud = ud;
}

static bool same_image(const renza_vision_span *a, const renza_vision_span *b) {
    return a->embedding.token_count == b->embedding.token_count &&
           memcmp(a->embedding.fingerprint, b->embedding.fingerprint,
                  sizeof(a->embedding.fingerprint)) == 0;
}

bool renza_session_vision_prefix_matches(const renza_session *s,
                                         const renza_vision_span *images, size_t image_count) {
    if (!s || !s->checkpoint_valid || (image_count != 0 && !images)) return false;
    if (s->image_count > image_count) return false;
    for (size_t i = 0; i < s->image_count; i++) {
        if (s->images[i].token_start != images[i].token_start ||
            !same_image(&s->images[i], &images[i])) return false;
    }
    if (s->image_count < image_count &&
        images[s->image_count].token_start < (uint32_t)s->tokens.len) return false;
    return true;
}

bool renza_session_vision_state_matches(const renza_session *s,
                                        const renza_vision_span *images, size_t image_count) {
    return s && s->image_count == image_count &&
           renza_session_vision_prefix_matches(s, images, image_count);
}

bool renza_session_vision_fingerprint_prefix_matches(const renza_session *s,
                                                     const renza_vision_span *images,
                                                     size_t image_count) {
    if (!s || !s->checkpoint_valid || (image_count != 0 && !images)) return false;
    if (s->image_count > image_count) return false;
    for (size_t i = 0; i < s->image_count; i++) {
        if (!same_image(&s->images[i], &images[i])) return false;
    }
    return true;
}

bool renza_session_rebase_vision_state(const renza_session *s,
                                       renza_vision_span *images, size_t image_count) {
    if (!s || !s->checkpoint_valid || (image_count != 0 && !images) ||
        image_count != s->image_count) return false;
    for (size_t i = 0; i < image_count; i++) {
        if (!same_image(&s->images[i], &images[i])) return false;
    }
    for (size_t i = 0; i < image_count; i++) images[i].token_start = s->images[i].token_start;
    return true;
}

bool renza_session_has_vision_state(const renza_session *s) {
    return s && (s->image_count != 0 || s->syncing_images);
}

renza_session *renza_session_new_test_checkpoint(const int *tokens, int n) {
    renza_session *s = renza_session_book_new();
    for (int i = 0; i < n; i++) renza_tokens_push(&s->tokens, tokens[i]);
    s->checkpoint_valid = true;
    return s;
}

void renza_session_free_test_checkpoint(renza_session *s) {
    renza_session_book_free(s);
}

void renza_session_set_test_images(renza_session *s, const renza_vision_span *images,
                                   size_t image_count) {
    if (!s) return;
    session_copy_images(s, images, image_count);
}

bool renza_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common) {
    if (live_len < 0 || canonical_len < 0 || common < 0) return true;
    if (common > live_len || common > canonical_len) return true;
    return common < live_len;
}
