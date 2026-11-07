/* The ds4 shim's generation (sample / eval / eval_speculative and the held
 * token) over a fake renza engine defined here, whose decode drafts: a window
 * holds min(budget, cap, spec) tokens, each the previous token plus one. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ds4.h"
#include "renza_common.h"

static int failures;

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                  \
        }                                                                \
    } while (0)

enum { FAKE_SPEC = 4, FAKE_CTX = 32, ACCEPTED_CAP = 17 };

struct renza_engine {
    int spec;
};

static struct {
    const char *pack_dir;
    int n_extra_args;
    const char *first_extra_arg;
    bool synced;
    int decodes;
    int token;
    int budget;
    int cap;
    renza_sampling sampling;
    int save_status;
    char saved_path[256];
    uint64_t load_bytes;
} fake;

const renza_engine_text *renza_engine_text_of(const renza_engine *e) {
    (void)e;
    return NULL;
}

int renza_engine_open(renza_engine **out, const renza_engine_options *opt,
                      char *err, size_t errlen) {
    (void)err;
    (void)errlen;
    fake.pack_dir = opt->pack_dir;
    fake.n_extra_args = opt->n_extra_args;
    fake.first_extra_arg = opt->n_extra_args > 0 ? opt->extra_args[0] : NULL;
    *out = renza_xcalloc(1, sizeof(**out));
    (*out)->spec = FAKE_SPEC;
    return RENZA_OK;
}

void renza_engine_close(renza_engine *e) {
    free(e);
}

int renza_engine_vocab_size(const renza_engine *e) {
    (void)e;
    return 0;
}

int renza_engine_embd_dim(const renza_engine *e) {
    (void)e;
    return 0;
}

int renza_engine_layer_count(const renza_engine *e) {
    (void)e;
    return 0;
}

int renza_engine_prefill_chunk(const renza_engine *e) {
    (void)e;
    return 0;
}

int renza_engine_spec(const renza_engine *e) {
    return e->spec;
}

bool renza_engine_has_vision(const renza_engine *e) {
    (void)e;
    return false;
}

int renza_engine_vision_encode_memory(renza_engine *e, const uint8_t *encoded,
                                      size_t encoded_len, renza_vision_embedding *out,
                                      char *err, size_t errlen) {
    (void)e;
    (void)encoded;
    (void)encoded_len;
    (void)out;
    renza_set_err(err, errlen, "no encoder");
    return RENZA_ERROR;
}

int renza_prompt_append_vision(const renza_engine *e, renza_tokens *prompt,
                               renza_vision_span *span, renza_vision_embedding *embedding,
                               char *err, size_t errlen) {
    (void)e;
    (void)prompt;
    (void)span;
    (void)embedding;
    renza_set_err(err, errlen, "no encoder");
    return RENZA_ERROR;
}

int renza_session_create(renza_session **out, renza_engine *e, int ctx_size,
                         char *err, size_t errlen) {
    (void)e;
    (void)err;
    (void)errlen;
    *out = renza_session_book_new();
    (*out)->ctx = ctx_size;
    return RENZA_OK;
}

void renza_session_free(renza_session *s) {
    renza_session_book_free(s);
}

int renza_session_sync_multimodal(renza_session *s, const renza_tokens *prompt,
                                  const renza_vision_span *images, size_t image_count,
                                  char *err, size_t errlen) {
    (void)err;
    (void)errlen;
    renza_session_book_assign(s, prompt->v, prompt->len, images, image_count);
    s->checkpoint_valid = true;
    fake.synced = true;
    return RENZA_OK;
}

int renza_session_decode(renza_session *s, int token, const renza_sampling *sampling,
                         int budget, int *out, int cap, char *err, size_t errlen) {
    fake.decodes++;
    fake.token = token;
    fake.budget = budget;
    fake.cap = cap;
    fake.sampling = *sampling;
    const int q = token >= 0 ? s->tokens.len : s->tokens.len - 1;
    if (!s->checkpoint_valid || (token < 0 && !fake.synced) || q + 2 > s->ctx) {
        renza_set_err(err, errlen, "refused");
        return RENZA_ERROR;
    }
    fake.synced = false;
    int n = budget < cap ? budget : cap;
    if (n > FAKE_SPEC) n = FAKE_SPEC;
    if (q + FAKE_SPEC > s->ctx) n = 1;
    int prev = token >= 0 ? token : s->tokens.v[s->tokens.len - 1];
    if (token >= 0) renza_tokens_push(&s->tokens, token);
    for (int i = 0; i < n; i++) {
        out[i] = ++prev;
        if (i + 1 < n) renza_tokens_push(&s->tokens, out[i]);
    }
    return n;
}

void renza_session_rewind(renza_session *s, int pos) {
    renza_session_book_rewind(s, pos);
}

void renza_session_invalidate(renza_session *s) {
    renza_session_book_invalidate(s);
}

/* A saved session here: the history's length, then its ids. */
int renza_session_save(renza_session *s, const char *path, char *err, size_t errlen) {
    snprintf(fake.saved_path, sizeof(fake.saved_path), "%s", path);
    if (fake.save_status != RENZA_OK) {
        renza_set_err(err, errlen, "unavailable");
        return fake.save_status;
    }
    FILE *fp = fopen(path, "wb");
    CHECK(fp != NULL);
    fwrite(&s->tokens.len, sizeof(int), 1, fp);
    fwrite(s->tokens.v, sizeof(int), (size_t)s->tokens.len, fp);
    fclose(fp);
    return RENZA_OK;
}

int renza_session_load(renza_session *s, FILE *fp, uint64_t bytes, char *err, size_t errlen) {
    fake.load_bytes = bytes;
    int n = 0;
    if (fread(&n, sizeof(n), 1, fp) != 1 || bytes != sizeof(int) * (uint64_t)(n + 1)) {
        renza_set_err(err, errlen, "not a saved session");
        return RENZA_ERROR;
    }
    int ids[FAKE_CTX];
    CHECK(n <= FAKE_CTX && fread(ids, sizeof(int), (size_t)n, fp) == (size_t)n);
    renza_session_book_assign(s, ids, n, NULL, 0);
    s->checkpoint_valid = true;
    return RENZA_OK;
}

static ds4_tokens tokens_of(int first, int n) {
    ds4_tokens t = {0};
    for (int i = 0; i < n; i++) ds4_tokens_push(&t, first + i);
    return t;
}

static bool history_is(ds4_session *s, int first, int n) {
    ds4_tokens want = tokens_of(first, n);
    const ds4_tokens *have = ds4_session_tokens(s);
    const bool ok = have->len == n && ds4_tokens_starts_with(have, &want);
    ds4_tokens_free(&want);
    return ok;
}

static ds4_engine *open_engine(void) {
    static const char *const engine_args[] = {"--adapt-swaps", "0"};
    ds4_engine_options opt = {0};
    opt.model_path = "native.gguf";
    opt.pack_dir = "pack";
    opt.extra_args = engine_args;
    opt.n_extra_args = 2;
    ds4_engine *e = NULL;
    CHECK(ds4_engine_open(&e, &opt) == 0);
    return e;
}

static ds4_session *synced_session(ds4_engine *e, int ctx, int prompt_len) {
    ds4_session *s = NULL;
    CHECK(ds4_session_create(&s, e, ctx) == 0);
    ds4_tokens prompt = tokens_of(1, prompt_len);
    char err[64];
    CHECK(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
    ds4_tokens_free(&prompt);
    return s;
}

static void test_engine_options(void) {
    ds4_engine *e = open_engine();
    CHECK(fake.pack_dir && !strcmp(fake.pack_dir, "pack"));
    CHECK(fake.n_extra_args == 2 && !strcmp(fake.first_extra_arg, "--adapt-swaps"));
    CHECK(ds4_engine_mtp_draft_tokens(e) == FAKE_SPEC);
    ds4_engine_close(e);
}

/* The server's loop: sample, then feed the sampled token, which returns the
 * window's accepted run and holds the token after it for the next sample. */
static void test_windows(void) {
    ds4_engine *e = open_engine();
    ds4_session *s = synced_session(e, FAKE_CTX, 3);
    uint64_t rng = 77;
    int accepted[ACCEPTED_CAP];
    char err[64];

    CHECK(ds4_session_sample(s, 0.6f, 20, 0.8f, 0.0f, &rng) == 4);
    CHECK(fake.token == RENZA_TOKEN_NONE && fake.budget == 1);
    CHECK(fake.sampling.temperature == 0.6f && fake.sampling.top_k == 20 &&
          fake.sampling.top_p == 0.8f && fake.sampling.seed == 77);
    CHECK(ds4_session_pos(s) == 3);
    const int decodes = fake.decodes;
    CHECK(ds4_session_sample(s, 0.6f, 20, 0.8f, 0.0f, &rng) == 4);
    CHECK(fake.decodes == decodes);

    int n = ds4_session_eval_speculative(s, 4, 10, -1, 0.6f, 20, 0.8f, 0.0f, &rng,
                                         accepted, ACCEPTED_CAP, err, sizeof(err));
    CHECK(n == FAKE_SPEC);
    CHECK(fake.token == 4 && fake.budget == 10 && fake.cap == ACCEPTED_CAP - 1);
    CHECK(accepted[0] == 4 && accepted[1] == 5 && accepted[2] == 6 && accepted[3] == 7);
    CHECK(history_is(s, 1, 7));
    CHECK(ds4_session_sample(s, 0.0f, 20, 0.8f, 0.0f, &rng) == 8);

    /* Two output tokens left: the window is bounded by them. */
    n = ds4_session_eval_speculative(s, 8, 2, -1, 0.0f, 20, 0.8f, 0.0f, &rng,
                                     accepted, ACCEPTED_CAP, err, sizeof(err));
    CHECK(n == 2 && accepted[0] == 8 && accepted[1] == 9);
    CHECK(fake.sampling.temperature == 0.0f);
    CHECK(history_is(s, 1, 9));
    CHECK(rng == 77);

    CHECK(ds4_session_sample(s, 0.6f, 20, 0.8f, 0.0f, &rng) == 10);
    CHECK(ds4_session_eval(s, 10, err, sizeof(err)) == 0);
    CHECK(fake.token == 10 && fake.budget == 1 && fake.sampling.temperature == 0.6f);
    CHECK(history_is(s, 1, 10));
    CHECK(ds4_session_sample(s, 0.6f, 20, 0.8f, 0.0f, &rng) == 11);

    ds4_session_free(s);
    ds4_engine_close(e);
}

static void test_held_token_dropped(void) {
    ds4_engine *e = open_engine();
    ds4_session *s = synced_session(e, FAKE_CTX, 3);
    uint64_t rng = 5;
    char err[64];

    CHECK(ds4_session_sample(s, 0.0f, 0, 0.0f, 0.0f, &rng) == 4);
    CHECK(ds4_session_eval(s, 4, err, sizeof(err)) == 0);
    ds4_session_rewind(s, 2);
    CHECK(ds4_session_sample(s, 0.0f, 0, 0.0f, 0.0f, &rng) == -1);

    ds4_tokens prompt = tokens_of(1, 6);
    CHECK(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
    CHECK(ds4_session_sample(s, 0.0f, 0, 0.0f, 0.0f, &rng) == 7);
    ds4_session_invalidate(s);
    CHECK(ds4_session_pos(s) == 0);
    CHECK(ds4_session_sample(s, 0.0f, 0, 0.0f, 0.0f, &rng) == -1);

    ds4_tokens_free(&prompt);
    ds4_session_free(s);
    ds4_engine_close(e);
}

/* Near the end of the context a window holds the fed token alone, and the
 * token at the last position is accepted without a window. */
static void test_context_end(void) {
    ds4_engine *e = open_engine();
    const int ctx = 8;
    ds4_session *s = synced_session(e, ctx, 6);
    uint64_t rng = 9;
    int accepted[ACCEPTED_CAP];
    char err[64];

    CHECK(ds4_session_sample(s, 0.0f, 0, 0.0f, 0.0f, &rng) == 7);
    int n = ds4_session_eval_speculative(s, 7, 2, -1, 0.0f, 0, 0.0f, 0.0f, &rng,
                                         accepted, ACCEPTED_CAP, err, sizeof(err));
    CHECK(n == 1 && accepted[0] == 7);
    CHECK(ds4_session_pos(s) == ctx - 1);
    CHECK(ds4_session_sample(s, 0.0f, 0, 0.0f, 0.0f, &rng) == 8);

    const int decodes = fake.decodes;
    n = ds4_session_eval_speculative(s, 8, 1, -1, 0.0f, 0, 0.0f, 0.0f, &rng,
                                     accepted, ACCEPTED_CAP, err, sizeof(err));
    CHECK(n == 1 && accepted[0] == 8);
    CHECK(fake.decodes == decodes);
    CHECK(history_is(s, 1, ctx - 1));

    ds4_session_free(s);
    ds4_engine_close(e);
}

/* What ds4's KV store does with a payload: stage it, copy it into its file
 * between its own header and trailer, load it from there, free it. */
static void test_payload(void) {
    ds4_engine *e = open_engine();
    CHECK(ds4_engine_routed_quant_bits(e) == 8 && ds4_engine_routed_quant_bits(NULL) == 0);
    ds4_session *s = synced_session(e, FAKE_CTX, 5);
    char err[64];
    ds4_session_payload_file staged = {0};
    CHECK(ds4_session_stage_payload(s, &staged, err, sizeof(err)) == 0);
    CHECK(staged.path && !strcmp(staged.path, fake.saved_path));
    CHECK(staged.bytes == sizeof(int) * 6);

    static const char head[] = "head", trailer[] = "trailer";
    FILE *store = tmpfile();
    fwrite(head, 1, sizeof(head), store);
    CHECK(ds4_session_write_staged_payload(&staged, store, err, sizeof(err)) == 0);
    fwrite(trailer, 1, sizeof(trailer), store);
    const uint64_t payload_bytes = staged.bytes;
    char *staged_path = strdup(staged.path);
    ds4_session_payload_file_free(&staged);
    CHECK(!staged.path && staged.bytes == 0 && access(staged_path, F_OK) != 0);
    free(staged_path);

    ds4_session *t = synced_session(e, FAKE_CTX, 2);
    uint64_t rng = 1;
    CHECK(ds4_session_sample(t, 0.0f, 0, 0.0f, 0.0f, &rng) == 3);
    fseek(store, sizeof(head), SEEK_SET);
    CHECK(ds4_session_load_payload(t, store, payload_bytes, err, sizeof(err)) == 0);
    CHECK(fake.load_bytes == payload_bytes && history_is(t, 1, 5));
    char after[sizeof(trailer)];
    CHECK(fread(after, 1, sizeof(after), store) == sizeof(after) && !memcmp(after, trailer, sizeof(after)));
    fclose(store);

    /* Nothing is left behind when the engine cannot save. */
    fake.save_status = RENZA_UNAVAILABLE;
    staged.bytes = 1;
    CHECK(ds4_session_stage_payload(s, &staged, err, sizeof(err)) == 1);
    CHECK(!staged.path && staged.bytes == 0 && access(fake.saved_path, F_OK) != 0);
    fake.save_status = RENZA_OK;

    ds4_session_free(t);
    ds4_session_free(s);
    ds4_engine_close(e);
}

int main(void) {
    test_engine_options();
    test_windows();
    test_held_token_dropped();
    test_context_end();
    const int before = failures;
    test_payload();
    printf("renza-shim tests: stage/write/load/free payload contract: %s\n",
           failures == before ? "ok" : "FAILED");
    if (failures) {
        fprintf(stderr, "renza-shim tests: %d failed\n", failures);
        return 1;
    }
    printf("renza-shim tests: ok\n");
    return 0;
}
