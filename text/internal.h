#ifndef RENZA_TEXT_INTERNAL_H
#define RENZA_TEXT_INTERNAL_H

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gguf.h"

static inline void renza_die(const char *msg) {
    fprintf(stderr, "renza: %s\n", msg);
    exit(1);
}

__attribute__((format(printf, 3, 4)))
static inline void renza_set_error(char *err, size_t errlen, const char *fmt, ...) {
    if (!err || errlen == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

static inline bool renza_str_eq(renza_str a, renza_str b) {
    return a.len == b.len && memcmp(a.ptr, b.ptr, (size_t)a.len) == 0;
}

static inline void *xmalloc(size_t size) {
    void *p = malloc(size ? size : 1);
    if (!p) renza_die("out of memory");
    return p;
}

static inline void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n ? n : 1, size);
    if (!p) renza_die("out of memory");
    return p;
}

static inline void *xrealloc(void *ptr, size_t size) {
    void *p = realloc(ptr, size ? size : 1);
    if (!p) renza_die("out of memory");
    return p;
}

#endif
