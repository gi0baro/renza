/* Stub engine: implements renza.h with no inference behind it.  The token
 * history, the vision identities, the think-mode helpers and the tokenizer
 * (engine/renza_common.c over text/) are real; everything that needs the
 * model's weights (session create, sync, decode, image encoding) fails with
 * an error. */

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "renza_common.h"

static const char STUB_NO_ENGINE[] = "no inference engine linked (renza stub engine)";
static const char STUB_NO_ENCODER[] = "no vision encoder linked (renza stub engine)";

struct renza_engine {
    renza_engine_text *text;
    int embd_dim;
    int layer_count;
    int prefill_chunk;
    int spec;
};

const renza_engine_text *renza_engine_text_of(const renza_engine *e) {
    return e->text;
}

/* Only the tokenizer and the model's metadata are loaded, from native_gguf. */
int renza_engine_open(renza_engine **out, const renza_engine_options *opt,
                      char *err, size_t errlen) {
    if (out) *out = NULL;
    if (!out || !opt || !opt->native_gguf) {
        renza_set_err(err, errlen, "missing native GGUF path or engine output");
        return RENZA_ERROR;
    }
    renza_engine_text *text = renza_engine_text_open(opt->native_gguf, err, errlen);
    if (!text) return RENZA_ERROR;
    renza_engine *e = renza_xcalloc(1, sizeof(*e));
    e->text = text;
    e->embd_dim = renza_engine_text_arch_u32(text, "embedding_length");
    e->layer_count = renza_engine_text_arch_u32(text, "block_count");
    e->prefill_chunk = opt->prefill_chunk > 0 ? opt->prefill_chunk : 0;
    e->spec = opt->spec > 0 ? opt->spec : 1;
    *out = e;
    return RENZA_OK;
}

void renza_engine_close(renza_engine *e) {
    if (!e) return;
    renza_engine_text_close(e->text);
    free(e);
}

int renza_engine_vocab_size(const renza_engine *e) {
    return renza_engine_text_vocab_size(e->text);
}

int renza_engine_embd_dim(const renza_engine *e) {
    return e->embd_dim;
}

int renza_engine_layer_count(const renza_engine *e) {
    return e->layer_count;
}

int renza_engine_prefill_chunk(const renza_engine *e) {
    return e->prefill_chunk;
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
    if (out) memset(out, 0, sizeof(*out));
    renza_set_err(err, errlen, STUB_NO_ENCODER);
    return RENZA_ERROR;
}

/* The image token ids belong to the encoder's model; without one there is
 * no embedding to place. */
int renza_prompt_append_vision(const renza_engine *e, renza_tokens *prompt,
                               renza_vision_span *span, renza_vision_embedding *embedding,
                               char *err, size_t errlen) {
    (void)e;
    (void)prompt;
    (void)span;
    (void)embedding;
    renza_set_err(err, errlen, STUB_NO_ENCODER);
    return RENZA_ERROR;
}

int renza_session_create(renza_session **out, renza_engine *e, int ctx_size,
                         char *err, size_t errlen) {
    (void)e;
    (void)ctx_size;
    if (out) *out = NULL;
    renza_set_err(err, errlen, STUB_NO_ENGINE);
    return RENZA_ERROR;
}

void renza_session_free(renza_session *s) {
    renza_session_book_free(s);
}

int renza_session_sync(renza_session *s, const renza_tokens *prompt,
                       char *err, size_t errlen) {
    return renza_session_sync_multimodal(s, prompt, NULL, 0, err, errlen);
}

int renza_session_sync_multimodal(renza_session *s, const renza_tokens *prompt,
                                  const renza_vision_span *images, size_t image_count,
                                  char *err, size_t errlen) {
    (void)s;
    (void)prompt;
    (void)images;
    (void)image_count;
    renza_set_err(err, errlen, STUB_NO_ENGINE);
    return RENZA_ERROR;
}

int renza_session_decode(renza_session *s, int token, const renza_sampling *sampling,
                         int budget, int *out, int cap, char *err, size_t errlen) {
    (void)s;
    (void)token;
    (void)sampling;
    (void)budget;
    (void)out;
    (void)cap;
    renza_set_err(err, errlen, STUB_NO_ENGINE);
    return RENZA_ERROR;
}

void renza_session_rewind(renza_session *s, int pos) {
    renza_session_book_rewind(s, pos);
}

void renza_session_invalidate(renza_session *s) {
    renza_session_book_invalidate(s);
}
