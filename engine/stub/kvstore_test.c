/* ds4's KV store (server/ds4_kvstore.c) over the ds4 shim and the stub
 * engine's saved sessions: an evicted conversation is stored, and a later
 * prompt that starts with its text finds it, loads it and reuses its tokens.
 * The tokenizer comes from the native GGUF (argv[1]). */

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ds4.h"
#include "ds4_kvstore.h"

static int failures;

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                  \
        }                                                                \
    } while (0)

enum { SKIPPED = 77, MIN_TOKENS = 4, BUDGET_MB = 64, KV_QUANT_TAG = 8 };

static const char CONVERSATION[] =
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
    "<|im_start|>assistant\nParis.<|im_end|>\n";
static const char NEXT_TURN[] = "<|im_start|>user\nAnd of Italy?<|im_end|>\n<|im_start|>assistant\n";
static const char OTHER_PROMPT[] = "<|im_start|>user\nSomething else entirely<|im_end|>\n";

static void log_line(void *ud, ds4_kvstore_log_type type, const char *msg) {
    (void)ud;
    (void)type;
    fprintf(stderr, "%s\n", msg);
}

static bool tokens_equal(const ds4_tokens *a, const ds4_tokens *b) {
    return a->len == b->len && ds4_tokens_starts_with(a, b);
}

/* The one entry file in dir, or NULL. */
static char *entry_path(const char *dir) {
    DIR *d = opendir(dir);
    if (!d) return NULL;
    char *path = NULL;
    for (struct dirent *de; (de = readdir(d)) != NULL;) {
        if (de->d_name[0] == '.') continue;
        free(path);
        const size_t n = strlen(dir) + strlen(de->d_name) + 2;
        path = malloc(n);
        snprintf(path, n, "%s/%s", dir, de->d_name);
    }
    closedir(d);
    return path;
}

static void remove_dir(const char *dir) {
    char *path;
    while ((path = entry_path(dir)) != NULL) {
        const int rc = unlink(path);
        free(path);
        if (rc != 0) break;
    }
    rmdir(dir);
}

static void test_store_round_trip(ds4_engine *e, ds4_kvstore *kc, const char *dir) {
    ds4_tokens live = {0};
    ds4_tokenize_rendered_chat(e, CONVERSATION, &live);
    ds4_session *evicted = ds4_session_new_test_checkpoint(live.v, live.len);
    char err[160] = {0};
    CHECK(ds4_kvstore_store_live_prefix(kc, e, evicted, ds4_session_tokens(evicted), live.len,
                                        "evict", NULL, err, sizeof(err)));

    char *path = entry_path(dir);
    CHECK(path != NULL);
    FILE *fp = path ? fopen(path, "rb") : NULL;
    ds4_kvstore_entry hdr = {0};
    uint32_t text_bytes = 0;
    CHECK(fp && ds4_kvstore_read_header(fp, &hdr, &text_bytes));
    CHECK(hdr.reason == DS4_KVSTORE_REASON_EVICT && hdr.tokens == (uint32_t)live.len &&
          hdr.quant_bits == KV_QUANT_TAG && text_bytes == strlen(CONVERSATION));
    if (fp) fclose(fp);
    free(path);

    const int other_ids[] = {1, 2, 3};
    ds4_session *next = ds4_session_new_test_checkpoint(other_ids, 3);
    const size_t prompt_len = strlen(CONVERSATION) + strlen(NEXT_TURN);
    char *prompt = calloc(1, prompt_len + 1);
    snprintf(prompt, prompt_len + 1, "%s%s", CONVERSATION, NEXT_TURN);
    ds4_tokens effective = {0};
    ds4_kvstore_load_result result = {0};
    const int loaded = ds4_kvstore_try_load_text(kc, e, next, prompt, &effective, &result, NULL, false);
    CHECK(loaded == live.len && result.tokens == live.len && result.text_bytes == strlen(CONVERSATION));
    CHECK(tokens_equal(ds4_session_tokens(next), &live) && ds4_session_checkpoint_valid(next));
    ds4_tokens full = {0};
    ds4_tokenize_rendered_chat(e, prompt, &full);
    CHECK(ds4_tokens_starts_with(&effective, &live) && tokens_equal(&effective, &full));

    ds4_tokens none = {0};
    ds4_kvstore_load_result missed = {0};
    CHECK(ds4_kvstore_try_load_text(kc, e, next, OTHER_PROMPT, &none, &missed, NULL, false) == 0);

    ds4_kvstore_load_result_free(&result);
    ds4_kvstore_load_result_free(&missed);
    ds4_tokens_free(&none);
    ds4_tokens_free(&full);
    ds4_tokens_free(&effective);
    free(prompt);
    ds4_session_free_test_checkpoint(next);
    ds4_session_free_test_checkpoint(evicted);
    ds4_tokens_free(&live);
}

int main(int argc, char **argv) {
    const char *gguf = argc > 1 ? argv[1] : "";
    if (access(gguf, R_OK) != 0) {
        printf("SKIP: no native GGUF at '%s'\n", gguf);
        return SKIPPED;
    }
    ds4_engine_options opt = {0};
    opt.model_path = gguf;
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) return 1;
    char dir[] = "/tmp/renza-kvstore-tests.XXXXXX";
    if (!mkdtemp(dir)) {
        perror("renza-kvstore-tests: mkdtemp");
        return 1;
    }
    ds4_kvstore kc = {0};
    ds4_kvstore_options kopt = ds4_kvstore_default_options();
    kopt.min_tokens = MIN_TOKENS;
    CHECK(ds4_kvstore_open(&kc, dir, BUDGET_MB, false, kopt, "renza-kvstore-tests", log_line, NULL));

    const int before = failures;
    test_store_round_trip(e, &kc, dir);
    printf("renza-kvstore-tests: evict save, prefix match, load, reused tokens: %s\n",
           failures == before ? "ok" : "FAILED");

    ds4_kvstore_close(&kc);
    remove_dir(dir);
    ds4_engine_close(e);
    if (failures) {
        fprintf(stderr, "renza-kvstore-tests: %d failed\n", failures);
        return 1;
    }
    printf("renza-kvstore-tests: ok\n");
    return 0;
}
