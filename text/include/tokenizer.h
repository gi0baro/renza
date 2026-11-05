#ifndef RENZA_TOKENIZER_H
#define RENZA_TOKENIZER_H

#include "gguf.h"
#include "renza.h"

typedef struct renza_vocab renza_vocab;

/* The qwen35 tokenizer.  Token strings and merge keys borrow the GGUF mapping;
 * it must outlive vocab.  NULL with err set on invalid tokenizer metadata. */
renza_vocab *renza_vocab_create(const renza_model *model, char *err, size_t errlen);
void renza_vocab_free(renza_vocab *vocab);
int renza_vocab_size(const renza_vocab *vocab);

/* Both encoders append to out. Lengths are bytes, including embedded NULs. */
void renza_tokenize_text(const renza_vocab *vocab, const char *text, size_t len,
                         renza_tokens *out);
void renza_tokenize_rendered_chat(const renza_vocab *vocab, const char *text,
                                  size_t len, renza_tokens *out);
/* The caller owns the result; len excludes its trailing NUL. */
char *renza_token_text(const renza_vocab *vocab, int token, size_t *len);
int renza_token_eos(const renza_vocab *vocab);
int renza_token_user(const renza_vocab *vocab);
int renza_token_assistant(const renza_vocab *vocab);
bool renza_token_is_stop_for_thinking(const renza_vocab *vocab, int token,
                                     bool thinking_enabled);

#endif
