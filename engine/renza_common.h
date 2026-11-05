#ifndef RENZA_COMMON_H
#define RENZA_COMMON_H

/* The parts of renza.h every engine implementation shares (engine/renza_common.c):
 * the pure helpers, the model's text side read from the native GGUF, and the
 * session bookkeeping (history, image identities, callbacks).  Private to
 * engine/. */

#include "renza.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The native GGUF's tokenizer and metadata (text/). */
typedef struct renza_engine_text renza_engine_text;

/* NULL with err set when the file cannot be read, is not a qwen4exp GGUF or
 * has invalid tokenizer metadata. */
renza_engine_text *renza_engine_text_open(const char *native_gguf, char *err, size_t errlen);
void renza_engine_text_close(renza_engine_text *t);
int renza_engine_text_vocab_size(const renza_engine_text *t);
/* The GGUF's qwen4exp.<name> value, or 0 when absent. */
int renza_engine_text_arch_u32(const renza_engine_text *t, const char *name);

/* Defined by each engine implementation: the text side the token and model
 * name functions of renza.h route to. */
const renza_engine_text *renza_engine_text_of(const renza_engine *e);

struct renza_session {
    renza_tokens tokens;
    bool checkpoint_valid;
    renza_vision_span *images;  /* identities only: embedding.rows is always NULL */
    size_t image_count;
    bool syncing_images;        /* a sync with images is in progress */
    int ctx;
    int reused;
    renza_progress_fn progress;
    void *progress_ud;
    renza_cancel_fn cancel;
    void *cancel_ud;
    void *backend;              /* the engine's own session; NULL for a test checkpoint */
};

renza_session *renza_session_book_new(void);
void renza_session_book_free(renza_session *s);
/* tokens = ids[0, n), the identities of the images wholly inside it;
 * checkpoint_valid is left to the caller. */
void renza_session_book_assign(renza_session *s, const int *ids, int n,
                               const renza_vision_span *images, size_t image_count);
/* renza_session_rewind's bookkeeping; false when pos changes nothing. */
bool renza_session_book_rewind(renza_session *s, int pos);
/* renza_session_invalidate's bookkeeping. */
void renza_session_book_invalidate(renza_session *s);

void renza_set_err(char *err, size_t errlen, const char *msg);
/* Allocation failure exits the process, as in ds4. */
void *renza_xcalloc(size_t n, size_t size);

#ifdef __cplusplus
}
#endif

#endif
