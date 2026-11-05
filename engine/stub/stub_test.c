/* renza.h's bookkeeping as the stub engine implements it, without the ds4 shim. */

#include <stdio.h>
#include <string.h>

#include "renza.h"

static int failures;

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                  \
        }                                                                \
    } while (0)

static renza_tokens tokens_of(const int *ids, int n) {
    renza_tokens t = {0};
    for (int i = 0; i < n; i++) renza_tokens_push(&t, ids[i]);
    return t;
}

static bool tokens_equal(const renza_tokens *t, const int *ids, int n) {
    const renza_tokens want = {(int *)ids, n, n};
    return t->len == n && renza_tokens_starts_with(t, &want);
}

static void test_history(void) {
    const int live[] = {10, 11, 12, 13, 14, 15};
    renza_session *s = renza_session_new_test_checkpoint(live, 6);
    CHECK(renza_session_checkpoint_valid(s));
    CHECK(renza_session_pos(s) == 6);
    CHECK(tokens_equal(renza_session_tokens(s), live, 6));

    const int diverging[] = {10, 11, 12, 99};
    renza_tokens prompt = tokens_of(diverging, 4);
    CHECK(renza_session_common_prefix(s, &prompt) == 3);
    renza_tokens extension = tokens_of(live, 6);
    renza_tokens_push(&extension, 16);
    CHECK(renza_session_common_prefix(s, &extension) == 6);

    renza_session_rewind(s, 4);
    CHECK(!renza_session_checkpoint_valid(s));
    CHECK(renza_session_pos(s) == 4);
    CHECK(tokens_equal(renza_session_tokens(s), live, 4));
    CHECK(renza_session_common_prefix(s, &extension) == 0);

    renza_session_invalidate(s);
    CHECK(renza_session_pos(s) == 0);
    CHECK(!renza_session_checkpoint_valid(s));

    renza_tokens_free(&prompt);
    renza_tokens_free(&extension);
    CHECK(prompt.v == NULL && prompt.len == 0);
    renza_session_free_test_checkpoint(s);
}

static renza_vision_span span_at(uint32_t start, uint32_t count, uint8_t mark) {
    renza_vision_span span = {0};
    span.token_start = start;
    span.embedding.token_count = count;
    span.embedding.fingerprint[0] = mark;
    return span;
}

static void test_vision_identity(void) {
    const int live[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    renza_session *s = renza_session_new_test_checkpoint(live, 12);
    const renza_vision_span held[] = {span_at(2, 3, 7), span_at(7, 2, 8)};
    renza_session_set_test_images(s, held, 2);
    CHECK(renza_session_has_vision_state(s));

    renza_vision_span request[] = {held[0], held[1], span_at(12, 4, 9)};
    CHECK(renza_session_vision_prefix_matches(s, request, 3));
    CHECK(!renza_session_vision_state_matches(s, request, 3));
    CHECK(renza_session_vision_state_matches(s, request, 2));
    request[2].token_start = 11;
    CHECK(!renza_session_vision_prefix_matches(s, request, 3));

    renza_vision_span moved[] = {span_at(3, 3, 7), span_at(9, 2, 8)};
    CHECK(!renza_session_vision_prefix_matches(s, moved, 2));
    CHECK(renza_session_vision_fingerprint_prefix_matches(s, moved, 2));
    CHECK(renza_session_rebase_vision_state(s, moved, 2));
    CHECK(moved[0].token_start == 2 && moved[1].token_start == 7);

    renza_vision_span changed[] = {span_at(2, 3, 7), span_at(7, 2, 6)};
    CHECK(!renza_session_vision_fingerprint_prefix_matches(s, changed, 2));
    CHECK(!renza_session_rebase_vision_state(s, changed, 2));
    CHECK(changed[1].token_start == 7);

    renza_session_rewind(s, 8);
    CHECK(renza_session_has_vision_state(s));
    CHECK(!renza_session_vision_fingerprint_prefix_matches(s, request, 1));
    renza_session_rewind(s, 4);
    CHECK(!renza_session_has_vision_state(s));
    renza_session_free_test_checkpoint(s);
}

static void test_no_inference(void) {
    const int live[] = {5, 6, 7};
    renza_session *s = renza_session_new_test_checkpoint(live, 3);
    renza_tokens prompt = tokens_of(live, 3);
    renza_tokens_push(&prompt, 8);
    char err[128] = {0};

    CHECK(renza_session_sync(s, &prompt, err, sizeof(err)) == RENZA_ERROR);
    CHECK(strstr(err, "no inference engine") != NULL);
    const renza_vision_span image = span_at(1, 1, 1);
    CHECK(renza_session_sync_multimodal(s, &prompt, &image, 1, err, sizeof(err)) == RENZA_ERROR);
    int out[4];
    const renza_sampling greedy = {0};
    CHECK(renza_session_decode(s, 8, &greedy, 4, out, 4, err, sizeof(err)) == RENZA_ERROR);
    CHECK(renza_session_checkpoint_valid(s));
    CHECK(tokens_equal(renza_session_tokens(s), live, 3));
    CHECK(!renza_session_has_vision_state(s));

    renza_tokens_free(&prompt);
    renza_session_free_test_checkpoint(s);
}

static void test_think_modes(void) {
    renza_think_mode level;
    CHECK(renza_think_mode_parse_level("40", &level));
    CHECK(renza_think_mode_level(level) == 40);
    CHECK(strcmp(renza_think_mode_name(level), "40") == 0);
    CHECK(renza_think_mode_enabled(level));
    CHECK(!renza_think_mode_parse_level("101", &level));
    CHECK(!renza_think_mode_enabled(RENZA_THINK_NONE));

    const int small = (int)renza_think_max_min_context() - 1;
    CHECK(renza_think_mode_for_context(RENZA_THINK_MAX, small, RENZA_MODEL_QWEN4) == RENZA_THINK_HIGH);
    CHECK(renza_think_mode_for_context(RENZA_THINK_MAX, small + 1, RENZA_MODEL_QWEN4) == RENZA_THINK_MAX);

    const char *xhigh = renza_think_effort_text(RENZA_THINK_HIGH, RENZA_MODEL_QWEN4);
    CHECK(xhigh && strstr(xhigh, "xhigh"));
    CHECK(renza_think_effort_text(RENZA_THINK_MAX, RENZA_MODEL_QWEN4) == xhigh);
    CHECK(renza_think_effort_text(RENZA_THINK_MEDIUM, RENZA_MODEL_QWEN4) == NULL);

    CHECK(!renza_session_rewrite_requires_rebuild(5, 7, 5));
    CHECK(renza_session_rewrite_requires_rebuild(5, 7, 4));
}

int main(void) {
    test_history();
    test_vision_identity();
    test_no_inference();
    test_think_modes();
    if (failures) {
        fprintf(stderr, "renza-stub tests: %d failed\n", failures);
        return 1;
    }
    printf("renza-stub tests: ok\n");
    return 0;
}
