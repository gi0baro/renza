/* text/'s GGUF reader and tokenizer, directly and through renza.h's stub
 * engine, on GGUF files written here: a tiny vocabulary and broken copies of
 * it. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "renza.h"
#include "tokenizer.h"

static int failures;

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                  \
        }                                                                \
    } while (0)

enum {
    GGUF_MAGIC = 0x46554747u,
    GGUF_VERSION = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_VERSION_OFFSET = 4,
    GGUF_N_KV_OFFSET = 16,
};

/* The vocabulary: printable ASCII as single-byte tokens (id = byte -
 * FIRST_PRINTABLE), then EXTRA_TOKENS.  "\xc4\xa0" is the byte-level encoding
 * of a space. */
static const char *const EXTRA_TOKENS[] = {
    "\xc4\xa0", "he", "ll", "hell", "hello", "\xc4\xa0hello",
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>",
};
enum { FIRST_PRINTABLE = 33, LAST_PRINTABLE = 126 };
enum {
    N_PRINTABLE = LAST_PRINTABLE - FIRST_PRINTABLE + 1,
    N_TOKENS = N_PRINTABLE + sizeof(EXTRA_TOKENS) / sizeof(EXTRA_TOKENS[0]),
    TOK_HELLO = N_PRINTABLE + 4,
    TOK_SPACE_HELLO,
    TOK_IM_START,
    TOK_IM_END,
    TOK_ENDOFTEXT,
    TOK_THINK,
};
static const char *const MERGES[] = {"h e", "l l", "he ll", "hell o", "\xc4\xa0 hello"};
static const char MODEL_NAME[] = "tiny";

static int printable_id(char c) {
    return c - FIRST_PRINTABLE;
}

typedef struct {
    uint8_t *v;
    size_t len;
    size_t cap;
} bytes;

static void put(bytes *b, const void *p, size_t n) {
    if (b->len + n > b->cap) {
        b->cap = (b->len + n) * 2;
        b->v = realloc(b->v, b->cap);
        if (!b->v) abort();
    }
    memcpy(b->v + b->len, p, n);
    b->len += n;
}

static void put_u32(bytes *b, uint32_t v) {
    put(b, &v, sizeof(v));
}

static void put_u64(bytes *b, uint64_t v) {
    put(b, &v, sizeof(v));
}

static void put_str(bytes *b, const char *s) {
    put_u64(b, strlen(s));
    put(b, s, strlen(s));
}

typedef struct {
    bytes b;
    uint64_t n_kv;
} gguf;

static void gguf_begin(gguf *g) {
    memset(g, 0, sizeof(*g));
    put_u32(&g->b, GGUF_MAGIC);
    put_u32(&g->b, GGUF_VERSION);
    put_u64(&g->b, 0);
    put_u64(&g->b, 0);
}

static void kv_key(gguf *g, const char *key, uint32_t type) {
    put_str(&g->b, key);
    put_u32(&g->b, type);
    g->n_kv++;
}

static void kv_string(gguf *g, const char *key, const char *value) {
    kv_key(g, key, GGUF_TYPE_STRING);
    put_str(&g->b, value);
}

static void kv_u32(gguf *g, const char *key, uint32_t value) {
    kv_key(g, key, GGUF_TYPE_UINT32);
    put_u32(&g->b, value);
}

static void kv_strings(gguf *g, const char *key, const char *const *values, size_t n) {
    kv_key(g, key, GGUF_TYPE_ARRAY);
    put_u32(&g->b, GGUF_TYPE_STRING);
    put_u64(&g->b, n);
    for (size_t i = 0; i < n; i++) put_str(&g->b, values[i]);
}

static bytes gguf_end(gguf *g) {
    memcpy(g->b.v + GGUF_N_KV_OFFSET, &g->n_kv, sizeof(g->n_kv));
    return g->b;
}

/* The tiny vocabulary's GGUF; with_tokens false leaves out the token table,
 * omit (when non-NULL) one token. */
static bytes vocab_gguf(const char *arch, bool with_tokens, const char *omit) {
    static char printable[N_PRINTABLE][2];
    const char *tokens[N_TOKENS];
    size_t n = 0;
    for (int i = 0; i < N_PRINTABLE; i++) {
        printable[i][0] = (char)(FIRST_PRINTABLE + i);
        tokens[n++] = printable[i];
    }
    for (int i = 0; i < N_TOKENS - N_PRINTABLE; i++) {
        if (!omit || strcmp(EXTRA_TOKENS[i], omit)) tokens[n++] = EXTRA_TOKENS[i];
    }

    gguf g;
    gguf_begin(&g);
    kv_string(&g, "general.architecture", arch);
    kv_string(&g, "general.name", MODEL_NAME);
    kv_u32(&g, "qwen4exp.block_count", 2);
    if (with_tokens) kv_strings(&g, "tokenizer.ggml.tokens", tokens, n);
    kv_strings(&g, "tokenizer.ggml.merges", MERGES, sizeof(MERGES) / sizeof(MERGES[0]));
    return gguf_end(&g);
}

static char gguf_path[] = "/tmp/renza-text-test-XXXXXX";

static void write_gguf(const uint8_t *data, size_t len) {
    FILE *fp = fopen(gguf_path, "wb");
    if (!fp || (len && fwrite(data, 1, len, fp) != len) || fclose(fp)) {
        perror(gguf_path);
        exit(1);
    }
}

static int open_engine(renza_engine **e, char *err, size_t errlen) {
    renza_engine_options opt;
    renza_engine_options_init(&opt);
    opt.native_gguf = gguf_path;
    err[0] = '\0';
    return renza_engine_open(e, &opt, err, errlen);
}

static void expect_open_error(const char *what, const uint8_t *data, size_t len,
                              const char *message) {
    write_gguf(data, len);
    renza_engine *e = NULL;
    char err[256];
    const int rc = open_engine(&e, err, sizeof(err));
    if (rc == RENZA_ERROR && strstr(err, message)) return;
    fprintf(stderr, "%s: rc %d, err \"%s\", want \"%s\"\n", what, rc, err, message);
    failures++;
    if (rc == RENZA_OK) renza_engine_close(e);
}

static void test_malformed(void) {
    const bytes valid = vocab_gguf("qwen4exp", true, NULL);
    bytes copy = {0};
    put(&copy, valid.v, valid.len);

    expect_open_error("zero-byte file", copy.v, 0, "too small");
    expect_open_error("truncated metadata", copy.v, valid.len / 2, "truncated GGUF file");

    copy.v[0] ^= 0xff;
    expect_open_error("bad magic", copy.v, copy.len, "not a GGUF file");
    copy.v[0] = valid.v[0];

    copy.v[GGUF_VERSION_OFFSET] = GGUF_VERSION - 1;
    expect_open_error("wrong version", copy.v, copy.len, "GGUF v3");
    copy.v[GGUF_VERSION_OFFSET] = valid.v[GGUF_VERSION_OFFSET];

    memset(copy.v + GGUF_N_KV_OFFSET, 0xff, sizeof(uint64_t));
    expect_open_error("metadata count", copy.v, copy.len, "count exceeds file size");

    bytes other = vocab_gguf("llama", true, NULL);
    expect_open_error("other architecture", other.v, other.len, "not a qwen4exp model");
    free(other.v);
    other = vocab_gguf("qwen4exp", false, NULL);
    expect_open_error("no token table", other.v, other.len, "token table is missing");
    free(other.v);
    other = vocab_gguf("qwen4exp", true, "<think>");
    expect_open_error("no <think>", other.v, other.len, "missing: <think>");
    free(other.v);

    free(copy.v);
    free(valid.v);
}

static bool tokens_are(const renza_tokens *t, const int *ids, int n) {
    if (t->len != n) return false;
    for (int i = 0; i < n; i++) {
        if (t->v[i] != ids[i]) return false;
    }
    return true;
}

static void expect_token_text(const renza_engine *e, int token, const char *want) {
    size_t len = SIZE_MAX;
    char *text = renza_engine_token_text(e, token, &len);
    CHECK(len == strlen(want) && strcmp(text, want) == 0);
    free(text);
}

static void test_engine_routing(void) {
    const bytes valid = vocab_gguf("qwen4exp", true, NULL);
    write_gguf(valid.v, valid.len);
    free(valid.v);
    renza_engine *e = NULL;
    char err[256];
    if (open_engine(&e, err, sizeof(err)) != RENZA_OK) {
        fprintf(stderr, "open: %s\n", err);
        failures++;
        return;
    }
    CHECK(renza_engine_vocab_size(e) == N_TOKENS);
    CHECK(strcmp(renza_engine_model_name(e), MODEL_NAME) == 0);
    CHECK(renza_engine_layer_count(e) == 2);

    static const char chat[] = "<|im_start|>hello hello<|im_end|>";
    renza_tokens out = {0};
    renza_tokens_push(&out, TOK_ENDOFTEXT);
    renza_engine_tokenize_chat(e, chat, strlen(chat), &out);
    const int want[] = {TOK_ENDOFTEXT, TOK_IM_START, TOK_HELLO, TOK_SPACE_HELLO, TOK_IM_END};
    CHECK(tokens_are(&out, want, 5));
    renza_tokens_free(&out);

    expect_token_text(e, TOK_SPACE_HELLO, " hello");
    expect_token_text(e, TOK_IM_END, "<|im_end|>");
    expect_token_text(e, N_TOKENS, "");

    CHECK(renza_engine_token_eos(e) == TOK_IM_END);
    CHECK(renza_engine_token_is_stop(e, TOK_ENDOFTEXT, RENZA_THINK_HIGH));
    CHECK(renza_engine_token_is_stop(e, TOK_THINK, RENZA_THINK_NONE));
    CHECK(!renza_engine_token_is_stop(e, TOK_THINK, RENZA_THINK_HIGH));
    renza_engine_close(e);
}

/* Raw mode parses no special token; the chat mode does. */
static void test_raw_mode(void) {
    const bytes valid = vocab_gguf("qwen4exp", true, NULL);
    write_gguf(valid.v, valid.len);
    free(valid.v);
    char err[256];
    renza_model model;
    if (!renza_model_open(&model, gguf_path, err, sizeof(err))) {
        fprintf(stderr, "model open: %s\n", err);
        failures++;
        return;
    }
    renza_vocab *vocab = renza_vocab_create(&model, err, sizeof(err));
    CHECK(vocab != NULL);
    if (!vocab) {
        renza_model_close(&model);
        return;
    }

    static const char special[] = "<|im_end|>";
    static const char text[] = "hello<|im_end|>";
    int want[sizeof(special)];
    int n = 0;
    want[n++] = TOK_HELLO;
    for (const char *p = special; *p; p++) want[n++] = printable_id(*p);
    renza_tokens out = {0};
    renza_tokenize_text(vocab, text, strlen(text), &out);
    CHECK(tokens_are(&out, want, n));

    out.len = 0;
    renza_tokenize_rendered_chat(vocab, text, strlen(text), &out);
    const int chat[] = {TOK_HELLO, TOK_IM_END};
    CHECK(tokens_are(&out, chat, 2));

    renza_tokens_free(&out);
    renza_vocab_free(vocab);
    renza_model_close(&model);
}

int main(void) {
    const int fd = mkstemp(gguf_path);
    if (fd < 0) {
        perror(gguf_path);
        return 1;
    }
    close(fd);
    test_malformed();
    test_engine_routing();
    test_raw_mode();
    unlink(gguf_path);
    if (failures) {
        fprintf(stderr, "renza-text tests: %d failed\n", failures);
        return 1;
    }
    printf("renza-text tests: ok\n");
    return 0;
}
