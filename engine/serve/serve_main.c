/* `renza-engine --serve`: Strata's stdin/stdout protocol over renza.h (plus
 * engine/renza_strata.h), with the lines, texts and exit statuses of
 * `strata --serve`.  A test harness for the adapter (tests/baseline/engine_run.py).
 *
 * Requests arrive on stdin as token ids (GEN, GENI with image rows; STOP,
 * QUIT), tokens go back on stdout (the protocol lines INFO, READY, RESUME, PP,
 * REUSED, T, DONE, ERR).  There is no tokenizer here: the client owns it.
 *
 * The command line is Strata's: the flags renza_engine_options has a field
 * for fill it, the rest go to extra_args. */

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "renza.h"
#include "renza_strata.h"

enum {
    EXIT_DEAD = 1,
    ERR_BYTES = 4096,
    READ_CHUNK = 4096,
};

static const int32_t SVE_MAGIC = 0x31455653;  /* 'SVE1' */
/* <|image_pad|>: one per image row in a prompt. */
static const int IMAGE_PAD = 248056;
static const uint64_t FNV_OFFSET = 1469598103934665603ull;
static const uint64_t FNV_PRIME = 1099511628211ull;

/* stdin is read on its own thread, so a STOP line reaches a request that is
 * still running: the flag is checked between prompt chunks and between
 * decode calls.  The reader is detached and may sit in read(2) past QUIT, so
 * what it touches is static. */
typedef struct line_node {
    struct line_node *next;
    char *text;
} line_node;

static struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    line_node *head;
    line_node *tail;
    bool eof;
    int stop_req;
} in = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, NULL, NULL, false, 0};

static void die_oom(void) {
    fprintf(stderr, "renza-engine: out of memory\n");
    exit(EXIT_DEAD);
}

static void *xrealloc(void *p, size_t size) {
    p = realloc(p, size);
    if (!p) die_oom();
    return p;
}

static bool stop_requested(void) {
    return __atomic_load_n(&in.stop_req, __ATOMIC_SEQ_CST) != 0;
}

static void set_stop(int value) {
    __atomic_store_n(&in.stop_req, value, __ATOMIC_SEQ_CST);
}

static void push_line(char *text) {
    line_node *node = xrealloc(NULL, sizeof(*node));
    node->next = NULL;
    node->text = text;
    pthread_mutex_lock(&in.mu);
    if (in.tail) in.tail->next = node;
    else in.head = node;
    in.tail = node;
    pthread_cond_signal(&in.cv);
    pthread_mutex_unlock(&in.mu);
}

/* read(2) on the descriptor, not stdio: exit() flushes every stdio stream and
 * would wait for a lock a blocked reader holds. */
static void *reader_main(void *arg) {
    (void)arg;
    char *buf = NULL;
    size_t len = 0, cap = 0, scanned = 0;
    bool at_eof = false;
    for (;;) {
        char *nl = scanned < len ? memchr(buf + scanned, '\n', len - scanned) : NULL;
        if (!nl && !at_eof) {
            if (cap - len < READ_CHUNK) {
                cap = cap ? cap * 2 : 2 * READ_CHUNK;
                buf = xrealloc(buf, cap);
            }
            const ssize_t n = read(0, buf + len, READ_CHUNK);
            if (n < 0 && errno == EINTR) continue;
            scanned = len;
            if (n <= 0) at_eof = true;
            else len += (size_t)n;
            continue;
        }
        if (!nl && len == 0) break;
        const size_t line_len = nl ? (size_t)(nl - buf) : len;
        char *text = xrealloc(NULL, line_len + 1);
        memcpy(text, buf, line_len);
        text[line_len] = '\0';
        const size_t used = nl ? line_len + 1 : len;
        memmove(buf, buf + used, len - used);
        len -= used;
        scanned = 0;
        if (line_len > 0 && text[line_len - 1] == '\r') text[line_len - 1] = '\0';
        if (strcmp(text, "STOP") == 0) {
            set_stop(1);
            free(text);
            continue;
        }
        push_line(text);
    }
    free(buf);
    pthread_mutex_lock(&in.mu);
    in.eof = true;
    pthread_cond_signal(&in.cv);
    pthread_mutex_unlock(&in.mu);
    return NULL;
}

static void start_reader(void) {
    pthread_t thread;
    if (pthread_create(&thread, NULL, reader_main, NULL) != 0) {
        fprintf(stderr, "renza-engine: cannot start the stdin reader\n");
        exit(EXIT_DEAD);
    }
    pthread_detach(thread);
}

/* The next line (the caller frees it), or NULL at the end of input. */
static char *next_line(void) {
    pthread_mutex_lock(&in.mu);
    while (!in.head && !in.eof) pthread_cond_wait(&in.cv, &in.mu);
    line_node *node = in.head;
    if (node) {
        in.head = node->next;
        if (!in.head) in.tail = NULL;
    }
    pthread_mutex_unlock(&in.mu);
    if (!node) return NULL;
    char *text = node->text;
    free(node);
    return text;
}

/* A dead engine: the process ends as Strata's did on a fatal error, without
 * the destructors (a faulted CUDA context can hang them). */
static void exit_dead(const char *message) {
    printf("ERR %s\n", message);
    fflush(stdout);
    fflush(stderr);
    _Exit(EXIT_DEAD);
}

typedef enum { FIELD_TEXT, FIELD_COUNT, FIELD_COUNT_AUTO, FIELD_REAL } field_kind;

static const struct {
    const char *flag;
    field_kind kind;
    size_t offset;
} FIELDS[] = {
    {"--pack", FIELD_TEXT, offsetof(renza_engine_options, pack_dir)},
    {"--native", FIELD_TEXT, offsetof(renza_engine_options, native_gguf)},
    {"--ple-gguf", FIELD_TEXT, offsetof(renza_engine_options, ple_gguf)},
    {"--expert-profile", FIELD_TEXT, offsetof(renza_engine_options, expert_profile)},
    {"--expert-cache", FIELD_COUNT_AUTO, offsetof(renza_engine_options, expert_cache)},
    {"--prefill", FIELD_COUNT_AUTO, offsetof(renza_engine_options, prefill_chunk)},
    {"--spec", FIELD_COUNT, offsetof(renza_engine_options, spec)},
    {"--mtp", FIELD_TEXT, offsetof(renza_engine_options, mtp_dir)},
    {"--spec-min-p", FIELD_REAL, offsetof(renza_engine_options, spec_min_p)},
    {"--max-context", FIELD_COUNT, offsetof(renza_engine_options, max_context)},
    {"--kv", FIELD_TEXT, offsetof(renza_engine_options, kv)},
    {"--kv-resident", FIELD_COUNT, offsetof(renza_engine_options, kv_resident)},
    {"--vram-reserve-mib", FIELD_COUNT, offsetof(renza_engine_options, vram_reserve_mib)},
    {"--pcie-frac", FIELD_REAL, offsetof(renza_engine_options, pcie_frac)},
    {"--pool-workers", FIELD_COUNT, offsetof(renza_engine_options, pool_workers)},
    {"--prompt-cache", FIELD_COUNT, offsetof(renza_engine_options, prompt_cache)},
};
enum { N_FIELDS = sizeof(FIELDS) / sizeof(FIELDS[0]) };

static bool is_sentinel(double v) {
    return v == RENZA_OPTION_DEFAULT || v == RENZA_OPTION_AUTO;
}

/* Sets the field from value when it is exactly representable, so the adapter
 * hands Strata the value its own parse would have read. */
static bool set_field(renza_engine_options *opt, int f, const char *value) {
    char *field = (char *)opt + FIELDS[f].offset;
    char *end = NULL;
    if (FIELDS[f].kind == FIELD_TEXT) {
        *(const char **)field = value;
        return true;
    }
    if (FIELDS[f].kind == FIELD_COUNT_AUTO && strcmp(value, "auto") == 0) {
        *(int *)field = RENZA_OPTION_AUTO;
        return true;
    }
    errno = 0;
    if (FIELDS[f].kind == FIELD_REAL) {
        const double v = strtod(value, &end);
        if (end == value || *end || errno || !isfinite(v) || is_sentinel(v)) return false;
        *(double *)field = v;
        return true;
    }
    const long v = strtol(value, &end, 10);
    if (end == value || *end || errno || v < INT_MIN || v > INT_MAX || is_sentinel((double)v)) return false;
    *(int *)field = (int)v;
    return true;
}

static int field_of(const char *arg) {
    for (int f = 0; f < N_FIELDS; f++) {
        if (strcmp(arg, FIELDS[f].flag) == 0) return f;
    }
    return -1;
}

static bool is_version(const char *arg) {
    return strcmp(arg, "--version") == 0 || strcmp(arg, "-V") == 0;
}

/* extra has room for argc + 2 * N_FIELDS entries.  Only the last occurrence
 * of a field flag counts, as in Strata's parser; a value the field cannot
 * hold goes to extra_args with its flag, last, where it wins the same way. */
static void options_from_args(int argc, char **argv, renza_engine_options *opt,
                              const char **extra, int *n_extra) {
    *n_extra = 0;
    /* Strata answers --version as one of the first two arguments, alone. */
    for (int i = 1; i < argc && i < 3; i++) {
        if (is_version(argv[i])) {
            extra[(*n_extra)++] = argv[i];
            return;
        }
    }
    const char *last[N_FIELDS] = {0};
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--serve") == 0) continue;
        const int f = field_of(argv[i]);
        if (f < 0 || i + 1 >= argc) {
            extra[(*n_extra)++] = argv[i];
            continue;
        }
        last[f] = argv[++i];
    }
    for (int f = 0; f < N_FIELDS; f++) {
        if (!last[f] || set_field(opt, f, last[f])) continue;
        extra[(*n_extra)++] = FIELDS[f].flag;
        extra[(*n_extra)++] = last[f];
    }
}

typedef struct {
    bool geni;
    long long max_new;
    renza_sampling sampling;
    /* img_keys=<hex>,<hex>,... (GENI): each picture's key from the server, in
     * prompt order; absent, the embeddings' hash. */
    uint64_t *img_keys;
    size_t n_img_keys;
    char *emb_path;
    renza_tokens ids;
} request;

static void request_init(request *r) {
    memset(r, 0, sizeof(*r));
    r->sampling.top_k = 20;
    r->sampling.top_p = 1.0f;
    r->sampling.penalty_repeat = 1.0f;
}

static void request_free(request *r) {
    free(r->img_keys);
    free(r->emb_path);
    renza_tokens_free(&r->ids);
}

static bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
}

/* One id of a GEN list: an int32 in decimal with an optional '-', nothing else. */
static bool parse_id(const char *s, size_t n, int *out) {
    size_t i = 0;
    const bool negative = n > 0 && s[0] == '-';
    if (negative) i++;
    if (i == n) return false;
    int64_t v = 0;
    for (; i < n; i++) {
        if (s[i] < '0' || s[i] > '9') return false;
        v = v * 10 + (s[i] - '0');
        if (v > (int64_t)INT32_MAX + 1) return false;
    }
    if (negative) v = -v;
    if (v < INT32_MIN || v > INT32_MAX) return false;
    *out = (int)v;
    return true;
}

/* Ids separated by commas or white space; the error text, or NULL. */
static const char *parse_id_list(const char *s, renza_tokens *out) {
    out->len = 0;
    for (;;) {
        while (*s && (is_space(*s) || *s == ',')) s++;
        if (!*s) break;
        const char *start = s;
        while (*s && !is_space(*s) && *s != ',') s++;
        int id = 0;
        if (!parse_id(start, (size_t)(s - start), &id) || id < 0) {
            out->len = 0;
            return "invalid token id: expected an integer in [0, 2147483647]";
        }
        renza_tokens_push(out, id);
    }
    if (out->len == 0) return "token list was empty";
    return NULL;
}

static void push_key(request *r, uint64_t key) {
    r->img_keys = xrealloc(r->img_keys, (r->n_img_keys + 1) * sizeof(*r->img_keys));
    r->img_keys[r->n_img_keys++] = key;
}

static void parse_img_keys(request *r, const char *q) {
    r->n_img_keys = 0;
    while (*q != '\0') {
        char *qe = NULL;
        const unsigned long long v = strtoull(q, &qe, 16);
        if (qe == q) {
            r->n_img_keys = 0;
            break;
        }
        push_key(r, (uint64_t)v);
        q = *qe == ',' ? qe + 1 : qe;
        if (*qe != ',' && *qe != '\0') {
            r->n_img_keys = 0;
            break;
        }
    }
}

static void parse_key(request *r, const char *key, const char *value) {
    const float fv = strtof(value, NULL);
    renza_sampling *sp = &r->sampling;
    if (strcmp(key, "temperature") == 0) sp->temperature = fv;
    else if (strcmp(key, "top_p") == 0) sp->top_p = fv;
    else if (strcmp(key, "top_k") == 0) sp->top_k = atoi(value);
    else if (strcmp(key, "min_p") == 0) sp->min_p = fv;
    else if (strcmp(key, "penalty_last_n") == 0) sp->penalty_last_n = atoi(value);
    else if (strcmp(key, "penalty_repeat") == 0) sp->penalty_repeat = fv;
    else if (strcmp(key, "penalty_freq") == 0) sp->penalty_freq = fv;
    else if (strcmp(key, "penalty_present") == 0) sp->penalty_present = fv;
    else if (strcmp(key, "seed") == 0) sp->seed = strtoull(value, NULL, 10);
    else if (strcmp(key, "img_keys") == 0) parse_img_keys(r, value);
}

/* GEN <max_new> [key=value ...] <id,id,...> or GENI <max_new> [key=value ...]
 * <file> <id,id,...>.  Absent keys: greedy, no penalties.  Strata's ckpt,
 * pcie_frac, spec_min_p and aux_cpus have no session field: skipped like any
 * unknown key.  False with the ERR text in msg. */
static bool parse_request(const char *line, request *r, char *msg, size_t msg_len) {
    r->geni = strncmp(line, "GENI ", 5) == 0;
    if (!r->geni && strncmp(line, "GEN ", 4) != 0) {
        snprintf(msg, msg_len, "expected: GEN <max_new> <id,id,...> or GENI <max_new> <file> <id,id,...>");
        return false;
    }
    char *endp = NULL;
    r->max_new = strtoll(line + (r->geni ? 5 : 4), &endp, 10);
    for (;;) {
        while (*endp == ' ') ++endp;
        char *start = endp;
        while (*endp != '\0' && *endp != ' ') ++endp;
        if (endp == start) break;
        const size_t len = (size_t)(endp - start);
        char *tok = xrealloc(NULL, len + 1);
        memcpy(tok, start, len);
        tok[len] = '\0';
        char *eq = strchr(tok, '=');
        if (!eq) {
            free(tok);
            endp = start;
            break;
        }
        *eq = '\0';
        parse_key(r, tok, eq + 1);
        free(tok);
    }
    if (r->geni) {
        while (*endp == ' ') ++endp;
        char *gap = strchr(endp, ' ');
        if (gap) {
            const size_t len = (size_t)(gap - endp);
            r->emb_path = xrealloc(NULL, len + 1);
            memcpy(r->emb_path, endp, len);
            r->emb_path[len] = '\0';
            endp = gap;
        }
    }
    const bool no_path = r->geni && (!r->emb_path || !r->emb_path[0]);
    const char *list_err = NULL;
    if (r->max_new < 1 || no_path || (list_err = parse_id_list(endp, &r->ids)) != NULL) {
        snprintf(msg, msg_len, "bad request: %s", list_err ? list_err : "max_new");
        return false;
    }
    return true;
}

static uint64_t fnv1a(const void *data, size_t n) {
    const uint8_t *p = data;
    uint64_t h = FNV_OFFSET;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= FNV_PRIME;
    }
    return h;
}

typedef struct {
    int64_t n, nx, ny;
    size_t off;
} image_record;

typedef struct {
    float *rows;
    renza_vision_span *spans;
    size_t n_spans;
} images;

static void images_free(images *im) {
    free(im->rows);
    free(im->spans);
}

static bool read_records(const char *path, int64_t n_embd, images *im, image_record **records,
                         size_t *n_records, char *msg, size_t msg_len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        snprintf(msg, msg_len, "cannot open %s", path);
        return false;
    }
    size_t n_rows = 0;
    bool ok = true;
    for (;;) {
        int32_t hdr[5];
        const size_t got = fread(hdr, sizeof(int32_t), 5, f);
        if (got == 0) break;
        if (got != 5 || hdr[0] != SVE_MAGIC || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 ||
            (int64_t)hdr[2] * hdr[3] != hdr[1] || hdr[4] != (int32_t)n_embd) {
            snprintf(msg, msg_len, "bad embeddings file (expected strata-vision records of width %lld)",
                     (long long)n_embd);
            ok = false;
            break;
        }
        const size_t off = n_rows, cnt = (size_t)hdr[1] * (size_t)hdr[4];
        float *rows = realloc(im->rows, (off + cnt) * sizeof(float));
        if (!rows || fread(rows + off, sizeof(float), cnt, f) != cnt) {
            if (rows) im->rows = rows;
            snprintf(msg, msg_len, "short embeddings file");
            ok = false;
            break;
        }
        im->rows = rows;
        n_rows = off + cnt;
        *records = xrealloc(*records, (*n_records + 1) * sizeof(**records));
        (*records)[(*n_records)++] = (image_record){hdr[1], hdr[2], hdr[3], off};
    }
    fclose(f);
    return ok;
}

/* The GENI embeddings file: one or more strata-vision records (int32 'SVE1',
 * n, nx, ny, n_embd, then n x n_embd floats) in prompt order, one per run of
 * <|image_pad|> tokens.  Fills one span per image; false with the ERR text. */
static bool read_images(const request *r, int64_t n_embd, images *im, char *msg, size_t msg_len) {
    image_record *recs = NULL;
    size_t n_recs = 0;
    bool ok = read_records(r->emb_path, n_embd, im, &recs, &n_recs, msg, msg_len);
    const int n = r->ids.len;
    int i = 0;
    size_t k = 0;
    while (ok && i < n) {
        if (r->ids.v[i] != IMAGE_PAD) {
            ++i;
            continue;
        }
        if (k >= n_recs) {
            snprintf(msg, msg_len, "the prompt has more images than the embeddings file");
            ok = false;
            break;
        }
        const image_record *rec = &recs[k++];
        for (int64_t j = 0; ok && j < rec->n; ++j) {
            if (i + j >= n || r->ids.v[i + j] != IMAGE_PAD) {
                snprintf(msg, msg_len, "image %zu has %lld rows but fewer <|image_pad|> tokens", k,
                         (long long)rec->n);
                ok = false;
            }
        }
        if (!ok) break;
        const uint64_t key = r->n_img_keys == n_recs
                                 ? r->img_keys[k - 1]
                                 : fnv1a(im->rows + rec->off, (size_t)rec->n * (size_t)n_embd * sizeof(float));
        im->spans = xrealloc(im->spans, (im->n_spans + 1) * sizeof(*im->spans));
        renza_vision_span *span = &im->spans[im->n_spans++];
        memset(span, 0, sizeof(*span));
        span->token_start = (uint32_t)i;
        span->embedding.token_count = (uint32_t)rec->n;
        span->embedding.grid_width = (uint32_t)rec->nx;
        span->embedding.grid_height = (uint32_t)rec->ny;
        span->embedding.key = key;
        i += (int)rec->n;
    }
    if (ok && k != n_recs) {
        snprintf(msg, msg_len, "the embeddings file has more images than the prompt");
        ok = false;
    }
    if (ok && n > 0 && r->ids.v[n - 1] == IMAGE_PAD) {
        snprintf(msg, msg_len, "the prompt cannot end in an image");
        ok = false;
    }
    for (size_t s = 0; ok && s < im->n_spans; ++s) im->spans[s].embedding.rows = im->rows + recs[s].off;
    free(recs);
    return ok;
}

/* DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused> [hits]
 *      [lookups] [RAM blobs] [file blobs] [file MB] [prompt tokens read] [offloaded]
 *      [lookup-chain accepted] [lookup-chain offered] [suffix accepted] [suffix offered] [windows]
 *      (only with --lookup-chain) chunk=<tokens> experts_streamed=<blobs> */
static void print_done(const renza_strata_request_stats *st, const char *finish) {
    char chain_txt[128] = "";
    if (st->chain_counts)
        snprintf(chain_txt, sizeof(chain_txt), " %lld %lld %lld %lld %lld", (long long)st->chain_accepted,
                 (long long)st->chain_offered, (long long)st->suffix_accepted, (long long)st->suffix_offered,
                 (long long)st->windows);
    printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld %lld %lld %.1f %lld %lld%s chunk=%lld "
           "experts_streamed=%lld\n",
           (long long)st->generated, (long long)st->prompt_tokens, st->prompt_ms, st->decode_ms, finish,
           (long long)st->draft_accepted, (long long)st->draft_offered, (long long)st->reused,
           (long long)st->expert_hits, (long long)st->expert_lookups, (long long)st->ram_reads,
           (long long)st->file_reads, st->file_read_mb, (long long)st->prompt_read, (long long)st->offloaded,
           chain_txt, (long long)st->chunk, (long long)st->experts_streamed);
    fflush(stdout);
}

/* RESUME is the first progress call (before reading: this many prompt tokens
 * are reused); PP the later ones. */
typedef struct {
    bool resumed;
    int from;
    struct timespec t0;
} progress_state;

static double elapsed_ms(const struct timespec *t0) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)(now.tv_sec - t0->tv_sec) * 1000.0 + (double)(now.tv_nsec - t0->tv_nsec) / 1e6;
}

static void on_progress(void *ud, int done, int total) {
    progress_state *ps = ud;
    if (!ps->resumed) {
        ps->resumed = true;
        ps->from = done;
        printf("RESUME %lld\n", (long long)done);
        fflush(stdout);
        return;
    }
    /* PP <position reached> <prompt tokens> <ms> <fresh tokens/s> */
    const double ms = elapsed_ms(&ps->t0);
    printf("PP %lld %lld %.0f %.1f\n", (long long)done, (long long)total, ms,
           ms > 0.0 ? 1000.0 * (double)(done - ps->from) / ms : 0.0);
    fflush(stdout);
}

static bool on_cancel(void *ud) {
    (void)ud;
    return stop_requested();
}

typedef struct {
    renza_engine *engine;
    renza_session *session;
    progress_state progress;
    int *out;
    int cap;
    long long max_context;
} client;

static void finish_request(client *c, const char *finish) {
    char err[ERR_BYTES];
    renza_strata_request_stats st;
    if (renza_strata_session_finish(c->session, &st, err, sizeof(err)) == RENZA_ENGINE_DEAD) exit_dead(err);
    print_done(&st, finish);
}

/* The decode loop: eos -> stop, then STOP -> cancel, then max_new -> length. */
static void generate(client *c, const request *r) {
    char err[ERR_BYTES];
    const char *finish = "length";
    int token = RENZA_TOKEN_NONE;
    long long produced = 0;
    while (produced < r->max_new) {
        const int got = renza_session_decode(c->session, token, &r->sampling, (int)(r->max_new - produced),
                                             c->out, c->cap, err, sizeof(err));
        if (got == RENZA_ENGINE_DEAD) exit_dead(err);
        if (got < 0 && !renza_session_checkpoint_valid(c->session)) {
            finish = "nonfinite";
            break;
        }
        if (got < 0) {
            char finish_err[ERR_BYTES];
            renza_strata_request_stats st;
            if (renza_strata_session_finish(c->session, &st, finish_err, sizeof(finish_err)) == RENZA_ENGINE_DEAD)
                exit_dead(finish_err);
            printf("ERR %s\n", err);
            return;
        }
        for (int i = 0; i < got; ++i) printf("T %d\n", c->out[i]);
        fflush(stdout);
        produced += got;
        token = c->out[got - 1];
        if (renza_strata_token_is_eos(c->engine, token)) {
            finish = "stop";
            break;
        }
        if (stop_requested()) {
            finish = "cancel";
            break;
        }
    }
    finish_request(c, finish);
}

static void serve_request(client *c, const char *line) {
    char msg[ERR_BYTES];
    request r;
    images im = {0};
    request_init(&r);
    if (!parse_request(line, &r, msg, sizeof(msg))) {
        printf("ERR %s\n", msg);
        goto done;
    }
    const long long n = r.ids.len;
    if (r.geni && !renza_engine_has_vision(c->engine)) {
        printf("ERR this engine was started without --vision\n");
        goto done;
    }
    if (r.geni && !read_images(&r, renza_engine_embd_dim(c->engine), &im, msg, sizeof(msg))) {
        printf("ERR %s\n", msg);
        fflush(stdout);
        goto done;
    }
    if (n + r.max_new + 8 > c->max_context) {
        printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%lld)\n", n, r.max_new,
               c->max_context);
        goto done;
    }
    const int vocab = renza_engine_vocab_size(c->engine);
    for (int i = 0; i < r.ids.len; i++) {
        if (r.ids.v[i] < 0 || r.ids.v[i] >= vocab) {
            printf("ERR a token id is outside the vocabulary\n");
            goto done;
        }
    }
    c->progress.resumed = false;
    clock_gettime(CLOCK_MONOTONIC, &c->progress.t0);
    const int synced = renza_session_sync_multimodal(c->session, &r.ids, im.spans, im.n_spans, msg, sizeof(msg));
    if (synced == RENZA_ENGINE_DEAD) exit_dead(msg);
    if (synced == RENZA_ERROR) {
        /* a request that began reading ended on non-finite logits; a refused one opened none */
        char err[ERR_BYTES];
        renza_strata_request_stats st;
        const int ended = renza_strata_session_finish(c->session, &st, err, sizeof(err));
        if (ended == RENZA_ENGINE_DEAD) exit_dead(err);
        if (ended != RENZA_OK) {
            printf("ERR %s\n", msg);
            goto done;
        }
        printf("REUSED %d\n", renza_session_reused(c->session));
        fflush(stdout);
        print_done(&st, "nonfinite");
        goto done;
    }
    printf("REUSED %d\n", renza_session_reused(c->session));  /* the prompt is read; the first window comes next */
    fflush(stdout);
    if (synced == RENZA_INTERRUPTED) finish_request(c, "cancel");
    else generate(c, &r);
done:
    images_free(&im);
    request_free(&r);
}

int main(int argc, char **argv) {
    /* Unbuffered: the output before a crash is the interesting part. */
    setvbuf(stdout, NULL, _IONBF, 0);
    renza_engine_options opt;
    renza_engine_options_init(&opt);
    const char **extra = xrealloc(NULL, ((size_t)argc + 2 * N_FIELDS) * sizeof(*extra));
    int n_extra = 0;
    options_from_args(argc, argv, &opt, extra, &n_extra);
    opt.extra_args = extra;
    opt.n_extra_args = n_extra;
    char err[ERR_BYTES];
    client c = {0};
    if (renza_engine_open(&c.engine, &opt, err, sizeof(err)) != RENZA_OK) {
        bool printed = false;
        const int status = renza_strata_open_status(&printed);
        if (!printed) fprintf(stderr, "%s\n", err);
        return status;
    }
    start_reader();
    c.max_context = renza_strata_engine_max_context(c.engine);
    /* what the server's Monitor tab shows (servers before 0.1.8 skip unknown lines until READY) */
    printf("INFO %s\n", renza_strata_engine_info(c.engine));
    printf("READY %lld stop\n", c.max_context);  /* "stop": this engine honours STOP */
    fflush(stdout);
    if (renza_session_create(&c.session, c.engine, (int)c.max_context, err, sizeof(err)) != RENZA_OK)
        exit_dead(err);
    renza_session_set_progress(c.session, on_progress, &c.progress);
    renza_session_set_cancel(c.session, on_cancel, NULL);
    c.cap = renza_engine_spec(c.engine) > 1 ? renza_engine_spec(c.engine) : 1;
    c.out = xrealloc(NULL, (size_t)c.cap * sizeof(*c.out));
    for (char *line; (line = next_line()) != NULL; free(line)) {
        if (strcmp(line, "QUIT") == 0) {
            free(line);
            break;
        }
        set_stop(0);  /* a STOP that arrived between requests is stale */
        serve_request(&c, line);
    }
    renza_session_free(c.session);
    if (renza_strata_engine_close(c.engine, err, sizeof(err)) != RENZA_OK) exit_dead(err);
    free(c.out);
    free(extra);
    return 0;
}
