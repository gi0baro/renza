/* Qwen35 tokenizer extracted from DwarfStar's ds4.c (MIT). */
#include "tokenizer.h"
#include "internal.h"


static uint64_t hash_bytes(const void *ptr, uint64_t len) {
    const uint8_t *p = ptr;
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

typedef struct {
    renza_str key;
    int value;
    bool used;
} str_i32_entry;

typedef struct {
    str_i32_entry *entry;
    uint64_t cap;
    uint64_t used;
} str_i32_table;

static uint64_t next_pow2(uint64_t n) {
    uint64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void table_init(str_i32_table *t, uint64_t expected) {
    t->cap = next_pow2(expected * 2 + 16);
    t->used = 0;
    t->entry = xcalloc((size_t)t->cap, sizeof(t->entry[0]));
}

static void table_free(str_i32_table *t) {
    free(t->entry);
    memset(t, 0, sizeof(*t));
}

static void table_put(str_i32_table *t, renza_str key, int value) {
    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(key.ptr, key.len) & mask;

    while (t->entry[i].used) {
        if (renza_str_eq(t->entry[i].key, key)) {
            t->entry[i].value = value;
            return;
        }
        i = (i + 1) & mask;
    }

    t->entry[i].used = true;
    t->entry[i].key = key;
    t->entry[i].value = value;
    t->used++;
}

static bool table_get(const str_i32_table *t, const char *ptr, uint64_t len, int *value) {
    if (t->cap == 0) return false;

    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(ptr, len) & mask;

    while (t->entry[i].used) {
        renza_str key = t->entry[i].key;
        if (key.len == len && memcmp(key.ptr, ptr, len) == 0) {
            *value = t->entry[i].value;
            return true;
        }
        i = (i + 1) & mask;
    }
    return false;
}

void renza_tokens_push(renza_tokens *tv, int token) {
    if (tv->len == tv->cap) {
        tv->cap = tv->cap ? tv->cap * 2 : 64;
        tv->v = xrealloc(tv->v, (size_t)tv->cap * sizeof(tv->v[0]));
    }
    tv->v[tv->len++] = token;
}

void renza_tokens_free(renza_tokens *tv) {
    free(tv->v);
    memset(tv, 0, sizeof(*tv));
}


struct renza_vocab {
    renza_str *token;
    int n_vocab;
    int eos_id;
    int user_id;
    int assistant_id;
    int think_start_id;
    int think_end_id;
    int tool_call_start_id;
    int tool_call_end_id;
    int tool_response_start_id;
    int tool_response_end_id;
    int im_start_id;
    int im_end_id;
    int endoftext_id;
    str_i32_table token_to_id;
    str_i32_table merge_rank;
};

static void utf8_put(char **p, uint32_t cp) {
    if (cp <= 0x7f) {
        *(*p)++ = (char)cp;
    } else if (cp <= 0x7ff) {
        *(*p)++ = (char)(0xc0 | (cp >> 6));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        *(*p)++ = (char)(0xe0 | (cp >> 12));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else {
        *(*p)++ = (char)(0xf0 | (cp >> 18));
        *(*p)++ = (char)(0x80 | ((cp >> 12) & 0x3f));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    }
}

static uint32_t gpt2_byte_to_codepoint(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
        return b;
    }

    uint32_t n = 0;
    for (uint32_t x = 0; x < 256; x++) {
        if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174)) {
            continue;
        }
        if (x == b) return 256 + n;
        n++;
    }
    return b;
}

/* GPT-2 byte-level BPE first maps raw bytes to printable Unicode codepoints
 * so merges can operate on UTF-8 strings without losing byte identity. */
static char *byte_encode(renza_str in, uint64_t *out_len) {
    char *out = xmalloc((size_t)in.len * 4 + 1);
    char *p = out;

    for (uint64_t i = 0; i < in.len; i++) {
        utf8_put(&p, gpt2_byte_to_codepoint((uint8_t)in.ptr[i]));
    }
    *p = '\0';
    *out_len = (uint64_t)(p - out);
    return out;
}

static int utf8_len_from_first_byte(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

typedef struct {
    char *ptr;
    uint64_t len;
} owned_str;

static owned_str owned_copy(const char *ptr, uint64_t len) {
    owned_str s;
    s.ptr = xmalloc((size_t)len);
    memcpy(s.ptr, ptr, (size_t)len);
    s.len = len;
    return s;
}

/* Look up the merge rank for two adjacent BPE symbols. */
static int bpe_rank(const renza_vocab *vocab, const owned_str *a, const owned_str *b) {
    uint64_t len = a->len + 1 + b->len;
    char stack[512];
    char *buf = len <= sizeof(stack) ? stack : xmalloc((size_t)len);

    memcpy(buf, a->ptr, (size_t)a->len);
    buf[a->len] = ' ';
    memcpy(buf + a->len + 1, b->ptr, (size_t)b->len);

    int rank = -1;
    table_get(&vocab->merge_rank, buf, len, &rank);

    if (buf != stack) free(buf);
    return rank;
}

/* Apply byte-level BPE to one regex-like pre-tokenized piece and emit token ids. */
static void bpe_emit_piece(const renza_vocab *vocab, renza_str raw_piece, renza_tokens *out) {
    uint64_t encoded_len = 0;
    char *encoded = byte_encode(raw_piece, &encoded_len);

    int n_sym = 0;
    int cap_sym = 32;
    owned_str *sym = xcalloc((size_t)cap_sym, sizeof(sym[0]));

    for (uint64_t off = 0; off < encoded_len;) {
        int n = utf8_len_from_first_byte((uint8_t)encoded[off]);
        if (off + (uint64_t)n > encoded_len) n = 1;
        if (n_sym == cap_sym) {
            cap_sym *= 2;
            sym = xrealloc(sym, (size_t)cap_sym * sizeof(sym[0]));
        }
        sym[n_sym++] = owned_copy(encoded + off, (uint64_t)n);
        off += (uint64_t)n;
    }

    for (;;) {
        int best_i = -1;
        int best_rank = INT32_MAX;

        for (int i = 0; i + 1 < n_sym; i++) {
            int rank = bpe_rank(vocab, &sym[i], &sym[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }

        if (best_i < 0) break;

        owned_str merged;
        merged.len = sym[best_i].len + sym[best_i + 1].len;
        merged.ptr = xmalloc((size_t)merged.len);
        memcpy(merged.ptr, sym[best_i].ptr, (size_t)sym[best_i].len);
        memcpy(merged.ptr + sym[best_i].len, sym[best_i + 1].ptr, (size_t)sym[best_i + 1].len);

        free(sym[best_i].ptr);
        free(sym[best_i + 1].ptr);
        sym[best_i] = merged;

        for (int j = best_i + 1; j + 1 < n_sym; j++) {
            sym[j] = sym[j + 1];
        }
        n_sym--;
    }

    for (int i = 0; i < n_sym; i++) {
        int token = -1;
        if (table_get(&vocab->token_to_id, sym[i].ptr, sym[i].len, &token)) {
            renza_tokens_push(out, token);
        } else {
            for (uint64_t j = 0; j < sym[i].len; j++) {
                if (table_get(&vocab->token_to_id, sym[i].ptr + j, 1, &token)) {
                    renza_tokens_push(out, token);
                }
            }
        }
        free(sym[i].ptr);
    }

    free(sym);
    free(encoded);
}

static uint64_t next_utf8_char(const char *s, uint64_t len, uint64_t pos) {
    int n = utf8_len_from_first_byte((uint8_t)s[pos]);
    if (pos + (uint64_t)n > len) n = 1;
    return pos + (uint64_t)n;
}

static uint32_t utf8_peek_one(const char *s, uint64_t len, uint64_t pos, uint64_t *next) {
    const uint8_t c0 = (uint8_t)s[pos];
    int n = utf8_len_from_first_byte(c0);
    if (pos + (uint64_t)n > len) n = 1;
    *next = pos + (uint64_t)n;

    if (n == 1) return c0;
    if (n == 2) {
        return ((uint32_t)(c0 & 0x1f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f));
    }
    if (n == 3) {
        return ((uint32_t)(c0 & 0x0f) << 12) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 2] & 0x3f));
    }
    return ((uint32_t)(c0 & 0x07) << 18) |
           ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 12) |
           ((uint32_t)((uint8_t)s[pos + 2] & 0x3f) << 6) |
           ((uint32_t)((uint8_t)s[pos + 3] & 0x3f));
}

static uint32_t ascii_tolower_cp(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + ('a' - 'A');
    return cp;
}

#include "qwen4_unicode.inc"

static bool qwen4_urange_has(const ds4_qwen4_urange *r, size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < r[mid].lo) hi = mid;
        else if (cp > r[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}

#define DS4_QWEN4_CLASS(name_, cp_) \
    qwen4_urange_has(ds4_qwen4_##name_##_ranges, \
                     sizeof(ds4_qwen4_##name_##_ranges) / sizeof(ds4_qwen4_##name_##_ranges[0]), (cp_))

typedef struct {
    uint32_t cp;
    uint64_t next;
    bool valid;
    bool letter;
    bool mark;
    bool number;
    bool space;
} qwen4_char_info;

static qwen4_char_info qwen4_char_at(const char *s, uint64_t len, uint64_t pos) {
    qwen4_char_info c;
    memset(&c, 0, sizeof(c));
    if (pos >= len) return c;
    c.valid = true;
    c.cp = utf8_peek_one(s, len, pos, &c.next);
    c.letter = DS4_QWEN4_CLASS(letter, c.cp);
    c.mark = DS4_QWEN4_CLASS(mark, c.cp);
    c.number = DS4_QWEN4_CLASS(number, c.cp);
    c.space = DS4_QWEN4_CLASS(space, c.cp);
    return c;
}

/* Qwen3.8 pre-tokenization (tokenizer.ggml.pre = "qwen35"), the ordered
 * alternation
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ | \p{N} |
 *    ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
 * with the Unicode classes generated from the `regex` module. */
static void bpe_tokenize_text_qwen35(const renza_vocab *vocab, const char *text, uint64_t len, renza_tokens *out) {
    uint64_t pos = 0;

    while (pos < len) {
        const uint64_t start = pos;
        qwen4_char_info cur = qwen4_char_at(text, len, pos);
        if (!cur.valid) break;

        if (cur.cp == '\'' && cur.next < len) {
            qwen4_char_info n1 = qwen4_char_at(text, len, cur.next);
            const uint32_t c1 = ascii_tolower_cp(n1.cp);
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                pos = n1.next;
                bpe_emit_piece(vocab, (renza_str){ text + start, pos - start }, out);
                continue;
            }
            if (n1.valid && n1.next < len) {
                qwen4_char_info n2 = qwen4_char_at(text, len, n1.next);
                const uint32_t c2 = ascii_tolower_cp(n2.cp);
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    pos = n2.next;
                    bpe_emit_piece(vocab, (renza_str){ text + start, pos - start }, out);
                    continue;
                }
            }
        }

        /* an optional non-letter prefix, then a run of letters and marks */
        {
            uint64_t run = UINT64_MAX;
            if (cur.letter || cur.mark) {
                run = cur.next;
            } else if (cur.cp != '\r' && cur.cp != '\n' && !cur.number) {
                qwen4_char_info n1 = qwen4_char_at(text, len, cur.next);
                if (n1.valid && (n1.letter || n1.mark)) run = n1.next;
            }
            if (run != UINT64_MAX) {
                pos = run;
                while (pos < len) {
                    qwen4_char_info scan = qwen4_char_at(text, len, pos);
                    if (!scan.valid || !(scan.letter || scan.mark)) break;
                    pos = scan.next;
                }
                bpe_emit_piece(vocab, (renza_str){ text + start, pos - start }, out);
                continue;
            }
        }

        if (cur.number) {
            pos = cur.next;
            bpe_emit_piece(vocab, (renza_str){ text + start, pos - start }, out);
            continue;
        }

        {
            qwen4_char_info punct = cur;
            uint64_t punct_pos = pos;
            if (cur.cp == ' ') {
                punct_pos = cur.next;
                punct = qwen4_char_at(text, len, punct_pos);
            }
            if (punct.valid && !punct.space && !punct.letter && !punct.mark && !punct.number) {
                pos = punct_pos;
                while (pos < len) {
                    qwen4_char_info scan = qwen4_char_at(text, len, pos);
                    if (!scan.valid || scan.space || scan.letter || scan.mark || scan.number) break;
                    pos = scan.next;
                }
                while (pos < len) {
                    qwen4_char_info scan = qwen4_char_at(text, len, pos);
                    if (!scan.valid || !(scan.cp == '\r' || scan.cp == '\n')) break;
                    pos = scan.next;
                }
                bpe_emit_piece(vocab, (renza_str){ text + start, pos - start }, out);
                continue;
            }
        }

        if (cur.space) {
            uint64_t p = pos;
            uint64_t last_newline_end = 0;
            uint64_t last_ws_start = pos;
            int nspace = 0;
            while (p < len) {
                qwen4_char_info scan = qwen4_char_at(text, len, p);
                if (!scan.valid || !scan.space) break;
                last_ws_start = p;
                if (scan.cp == '\r' || scan.cp == '\n') last_newline_end = scan.next;
                p = scan.next;
                nspace++;
            }
            if (last_newline_end) {
                pos = last_newline_end;
            } else if (nspace > 1 && p < len) {
                pos = last_ws_start;
            } else {
                pos = p;
            }
            bpe_emit_piece(vocab, (renza_str){ text + start, pos - start }, out);
            continue;
        }

        pos = cur.next;
        if (pos == start) pos = next_utf8_char(text, len, pos);
        bpe_emit_piece(vocab, (renza_str){ text + start, pos - start }, out);
    }
}
static int vocab_lookup(const renza_vocab *vocab, const char *text) {
    int token = -1;
    if (!table_get(&vocab->token_to_id, text, strlen(text), &token)) return -1;
    return token;
}

static bool lookup_required(const renza_vocab *vocab, const char *text, int *out,
                            char *err, size_t errlen) {
    *out = vocab_lookup(vocab, text);
    if (*out >= 0) return true;
    renza_set_error(err, errlen, "required tokenizer token is missing: %s", text);
    return false;
}

/* Load token strings, special token ids, and merge ranks from GGUF metadata. */

static bool vocab_load(renza_vocab *vocab, const renza_model *model, char *err, size_t errlen) {
    memset(vocab, 0, sizeof(*vocab));

    renza_array_ref tokens;
    renza_array_ref merges;
    if (!renza_model_get_array(model, "tokenizer.ggml.tokens", &tokens) ||
        tokens.type != GGUF_VALUE_STRING || tokens.len > INT32_MAX) {
        renza_set_error(err, errlen, "GGUF tokenizer token table is missing or invalid");
        return false;
    }
    if (!renza_model_get_array(model, "tokenizer.ggml.merges", &merges) ||
        merges.type != GGUF_VALUE_STRING || merges.len > INT32_MAX) {
        renza_set_error(err, errlen, "GGUF tokenizer merge table is missing or invalid");
        return false;
    }

    vocab->n_vocab = (int)tokens.len;
    vocab->token = xcalloc((size_t)vocab->n_vocab, sizeof(vocab->token[0]));
    table_init(&vocab->token_to_id, tokens.len);

    renza_cursor c = renza_cursor_at(model, tokens.data_pos);
    for (int i = 0; i < vocab->n_vocab; i++) {
        if (!renza_cursor_string(&c, &vocab->token[i])) {
            renza_set_error(err, errlen, "%s", c.error);
            return false;
        }
        table_put(&vocab->token_to_id, vocab->token[i], i);
    }

    table_init(&vocab->merge_rank, merges.len);
    c = renza_cursor_at(model, merges.data_pos);
    for (uint64_t i = 0; i < merges.len; i++) {
        renza_str merge;
        if (!renza_cursor_string(&c, &merge)) {
            renza_set_error(err, errlen, "%s", c.error);
            return false;
        }
        table_put(&vocab->merge_rank, merge, (int)i);
    }
    /* Qwen ChatML convention, as ds4 applies it (the metadata's bos/eos ids are
     * not read): no BOS, <|im_end|> ends a turn, <|endoftext|> is a second stop. */
    if (!lookup_required(vocab, "<|im_start|>", &vocab->im_start_id, err, errlen) ||
        !lookup_required(vocab, "<|im_end|>", &vocab->im_end_id, err, errlen) ||
        !lookup_required(vocab, "<|endoftext|>", &vocab->endoftext_id, err, errlen) ||
        !lookup_required(vocab, "<think>", &vocab->think_start_id, err, errlen) ||
        !lookup_required(vocab, "</think>", &vocab->think_end_id, err, errlen)) {
        return false;
    }
    vocab->eos_id = vocab->im_end_id;
    vocab->user_id = -1;
    vocab->assistant_id = -1;
    vocab->tool_call_start_id = vocab_lookup(vocab, "<tool_call>");
    vocab->tool_call_end_id = vocab_lookup(vocab, "</tool_call>");
    vocab->tool_response_start_id = vocab_lookup(vocab, "<tool_response>");
    vocab->tool_response_end_id = vocab_lookup(vocab, "</tool_response>");
    return true;
}

static bool special_token_at(const renza_vocab *vocab, const char *p, size_t remaining, int *token, size_t *len) {
    struct special {
        const char *text;
        int token;
    } specials[] = {
        {"<think>",                vocab->think_start_id},
        {"</think>",               vocab->think_end_id},
        {"<tool_call>",            vocab->tool_call_start_id},
        {"</tool_call>",           vocab->tool_call_end_id},
        {"<tool_response>",        vocab->tool_response_start_id},
        {"</tool_response>",       vocab->tool_response_end_id},
        {"<|im_start|>",           vocab->im_start_id},
        {"<|im_end|>",             vocab->im_end_id},
        {"<|endoftext|>",          vocab->endoftext_id},
    };

    for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); i++) {
        if (specials[i].token < 0) continue;
        size_t n = strlen(specials[i].text);
        if (n <= remaining && !memcmp(p, specials[i].text, n)) {
            *token = specials[i].token;
            *len = n;
            return true;
        }
    }
    return false;
}

static void tokenize_span(const renza_vocab *vocab, const char *p, size_t n, renza_tokens *out) {
    if (n) bpe_tokenize_text_qwen35(vocab, p, n, out);
}

static void tokenize_rendered_chat_vocab(const renza_vocab *vocab, const char *text, size_t size,
                                         renza_tokens *out) {
    if (!text) {
        text = "";
        size = 0;
    }

    const char *span = text;
    const char *p = text;
    const char *end = text + size;
    while (p < end) {
        int token = -1;
        size_t len = 0;
        if (special_token_at(vocab, p, (size_t)(end - p), &token, &len)) {
            tokenize_span(vocab, span, (size_t)(p - span), out);
            renza_tokens_push(out, token);
            p += len;
            span = p;
            continue;
        }
        p++;
    }
    tokenize_span(vocab, span, (size_t)(p - span), out);
}

static uint32_t utf8_decode_one(const char *s, uint64_t len, uint64_t *pos) {
    const uint8_t c = (uint8_t)s[*pos];
    if (c < 0x80 || *pos + 1 >= len) {
        (*pos)++;
        return c;
    }
    if ((c & 0xe0) == 0xc0 && *pos + 1 < len) {
        uint32_t cp = ((uint32_t)(c & 0x1f) << 6) | ((uint8_t)s[*pos + 1] & 0x3f);
        *pos += 2;
        return cp;
    }
    if ((c & 0xf0) == 0xe0 && *pos + 2 < len) {
        uint32_t cp = ((uint32_t)(c & 0x0f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 2] & 0x3f);
        *pos += 3;
        return cp;
    }
    if ((c & 0xf8) == 0xf0 && *pos + 3 < len) {
        uint32_t cp = ((uint32_t)(c & 0x07) << 18) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 2] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 3] & 0x3f);
        *pos += 4;
        return cp;
    }
    (*pos)++;
    return c;
}

static int gpt2_codepoint_to_byte(uint32_t cp) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) {
        return (int)cp;
    }

    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
            continue;
        }
        if (cp == 256 + n) return (int)b;
        n++;
    }
    return -1;
}

char *renza_token_text(const renza_vocab *vocab, int token, size_t *len) {
    if (!vocab || token < 0 || token >= vocab->n_vocab) {
        if (len) *len = 0;
        char *out = xmalloc(1);
        out[0] = '\0';
        return out;
    }

    renza_str s = vocab->token[token];
    char *out = xmalloc((size_t)s.len + 1);

    size_t n = 0;
    uint64_t pos = 0;
    while (pos < s.len) {
        uint32_t cp = utf8_decode_one(s.ptr, s.len, &pos);
        int b = gpt2_codepoint_to_byte(cp);
        if (b >= 0) out[n++] = (char)b;
    }
    out[n] = '\0';
    if (len) *len = n;
    return out;
}


renza_vocab *renza_vocab_create(const renza_model *model, char *err, size_t errlen) {
    renza_vocab *vocab = xcalloc(1, sizeof(*vocab));
    if (vocab_load(vocab, model, err, errlen)) return vocab;
    renza_vocab_free(vocab);
    return NULL;
}

void renza_vocab_free(renza_vocab *vocab) {
    if (!vocab) return;
    free(vocab->token);
    table_free(&vocab->token_to_id);
    table_free(&vocab->merge_rank);
    free(vocab);
}

int renza_vocab_size(const renza_vocab *vocab) {
    return vocab ? vocab->n_vocab : 0;
}

void renza_tokenize_text(const renza_vocab *vocab, const char *text, size_t len,
                         renza_tokens *out) {
    if (vocab && text) bpe_tokenize_text_qwen35(vocab, text, len, out);
}

void renza_tokenize_rendered_chat(const renza_vocab *vocab, const char *text,
                                  size_t len, renza_tokens *out) {
    if (vocab) tokenize_rendered_chat_vocab(vocab, text, len, out);
}

int renza_token_eos(const renza_vocab *vocab) {
    return vocab ? vocab->eos_id : -1;
}

int renza_token_user(const renza_vocab *vocab) {
    return vocab ? vocab->user_id : -1;
}

int renza_token_assistant(const renza_vocab *vocab) {
    return vocab ? vocab->assistant_id : -1;
}

static bool token_is_stop(const renza_vocab *vocab, int token) {
    if (!vocab || token < 0) return false;
    return token == vocab->eos_id ||
           (vocab->endoftext_id >= 0 && token == vocab->endoftext_id);
}

static bool token_is_thinking_control(const renza_vocab *vocab, int token) {
    if (!vocab || token < 0) return false;
    return (vocab->think_start_id >= 0 && token == vocab->think_start_id) ||
           (vocab->think_end_id >= 0 && token == vocab->think_end_id);
}

bool renza_token_is_stop_for_thinking(const renza_vocab *vocab, int token,
                                     bool thinking_enabled) {
    return token_is_stop(vocab, token) ||
           (!thinking_enabled && token_is_thinking_control(vocab, token));
}
