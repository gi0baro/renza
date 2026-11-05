#include "tokenizer.h"
#include "internal.h"

#include <ctype.h>
#include <limits.h>

static char *read_stdin(size_t *len) {
    size_t cap = 4096;
    char *buf = xmalloc(cap + 1);
    *len = 0;
    for (;;) {
        size_t n = fread(buf + *len, 1, cap - *len, stdin);
        *len += n;
        if (ferror(stdin)) renza_die("cannot read stdin");
        if (feof(stdin)) break;
        if (cap > (SIZE_MAX - 1) / 2) renza_die("stdin is too large");
        cap *= 2;
        buf = xrealloc(buf, cap + 1);
    }
    buf[*len] = '\0';
    return buf;
}

static void detokenize(const renza_vocab *vocab, char *input, size_t len) {
    char *p = input;
    char *end = input + len;
    while (p < end && isspace((unsigned char)*p)) p++;
    if (p == end) return;
    for (;;) {
        if (p == end || !isdigit((unsigned char)*p)) renza_die("expected a token id");
        errno = 0;
        char *next;
        long id = strtol(p, &next, 10);
        if (errno || id > INT_MAX || id >= renza_vocab_size(vocab)) {
            renza_die("token id is outside the vocabulary");
        }
        size_t n;
        char *text = renza_token_text(vocab, (int)id, &n);
        if (fwrite(text, 1, n, stdout) != n) renza_die("cannot write stdout");
        free(text);
        p = next;
        while (p < end && isspace((unsigned char)*p)) p++;
        if (p == end) break;
        if (*p++ != ',') renza_die("expected a comma between token ids");
        while (p < end && isspace((unsigned char)*p)) p++;
    }
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3 ||
        (argc == 3 && strcmp(argv[2], "--chat") && strcmp(argv[2], "--detok"))) {
        fprintf(stderr, "usage: renza-tokenize <gguf> [--chat | --detok]\n");
        return 2;
    }
    char err[256];
    renza_model model;
    if (!renza_model_open(&model, argv[1], err, sizeof(err))) renza_die(err);
    renza_vocab *vocab = renza_vocab_create(&model, err, sizeof(err));
    if (!vocab) renza_die(err);
    size_t len;
    char *input = read_stdin(&len);
    if (argc == 3 && !strcmp(argv[2], "--detok")) {
        detokenize(vocab, input, len);
    } else {
        renza_tokens tokens = {0};
        if (argc == 3) renza_tokenize_rendered_chat(vocab, input, len, &tokens);
        else renza_tokenize_text(vocab, input, len, &tokens);
        for (int i = 0; i < tokens.len; i++) {
            if (i) putchar(',');
            printf("%d", tokens.v[i]);
        }
        putchar('\n');
        renza_tokens_free(&tokens);
    }
    free(input);
    renza_vocab_free(vocab);
    renza_model_close(&model);
    if (fflush(stdout) || ferror(stdout)) renza_die("cannot write stdout");
    return 0;
}
