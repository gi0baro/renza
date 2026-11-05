/* GGUF reader extracted from DwarfStar's ds4.c (MIT). */
#include "gguf.h"
#include "internal.h"

#include <fcntl.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define RENZA_GGUF_MAGIC 0x46554747u
#define RENZA_GGUF_VERSION 3
#define RENZA_GGUF_MIN_SIZE 32

static void cursor_error(renza_cursor *c, const char *msg) {
    if (c->error[0] == '\0') {
        snprintf(c->error, sizeof(c->error), "%s at byte %" PRIu64, msg, c->pos);
    }
}

static bool cursor_has(renza_cursor *c, uint64_t n) {
    if (n > c->size || c->pos > c->size - n) {
        cursor_error(c, "truncated GGUF file");
        return false;
    }
    return true;
}

static bool cursor_read(renza_cursor *c, void *dst, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    memcpy(dst, c->base + c->pos, (size_t)n);
    c->pos += n;
    return true;
}

static bool cursor_skip(renza_cursor *c, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    c->pos += n;
    return true;
}

static bool cursor_u32(renza_cursor *c, uint32_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_u64(renza_cursor *c, uint64_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

bool renza_cursor_string(renza_cursor *c, renza_str *s) {
    uint64_t len;
    if (!cursor_u64(c, &len)) return false;
    if (!cursor_has(c, len)) return false;
    s->ptr = (const char *)(c->base + c->pos);
    s->len = len;
    c->pos += len;
    return true;
}

static uint64_t scalar_value_size(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_UINT8:
    case GGUF_VALUE_INT8:
    case GGUF_VALUE_BOOL:
        return 1;
    case GGUF_VALUE_UINT16:
    case GGUF_VALUE_INT16:
        return 2;
    case GGUF_VALUE_UINT32:
    case GGUF_VALUE_INT32:
    case GGUF_VALUE_FLOAT32:
        return 4;
    case GGUF_VALUE_UINT64:
    case GGUF_VALUE_INT64:
    case GGUF_VALUE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

static bool skip_value(renza_cursor *c, uint32_t type, int depth) {
    if (depth > 8) {
        cursor_error(c, "metadata array nesting is too deep");
        return false;
    }

    uint64_t scalar = scalar_value_size(type);
    if (scalar != 0) return cursor_skip(c, scalar);

    if (type == GGUF_VALUE_STRING) {
        renza_str ignored;
        return renza_cursor_string(c, &ignored);
    }

    if (type == GGUF_VALUE_ARRAY) {
        uint32_t item_type;
        uint64_t len;

        if (!cursor_u32(c, &item_type)) return false;
        if (!cursor_u64(c, &len)) return false;

        uint64_t item_size = scalar_value_size(item_type);
        if (item_size != 0) {
            if (len > UINT64_MAX / item_size) {
                cursor_error(c, "metadata array is too large");
                return false;
            }
            return cursor_skip(c, len * item_size);
        }

        for (uint64_t i = 0; i < len; i++) {
            if (!skip_value(c, item_type, depth + 1)) return false;
        }
        return true;
    }

    cursor_error(c, "unknown GGUF metadata type");
    return false;
}

renza_cursor renza_cursor_at(const renza_model *m, uint64_t pos) {
    renza_cursor c = {
        .base = m->map,
        .size = m->size,
        .pos = pos,
        .error = {0},
    };
    return c;
}

static renza_kv *model_find_kv(const renza_model *m, const char *key) {
    const renza_str want = {key, strlen(key)};
    for (uint64_t i = 0; i < m->n_kv; i++) {
        if (renza_str_eq(m->kv[i].key, want)) return &m->kv[i];
    }
    return NULL;
}

bool renza_model_get_string(const renza_model *m, const char *key, renza_str *out) {
    renza_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_STRING) return false;
    renza_cursor c = renza_cursor_at(m, kv->value_pos);
    return renza_cursor_string(&c, out);
}

bool renza_model_get_u32(const renza_model *m, const char *key, uint32_t *out) {
    renza_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_UINT32) return false;
    renza_cursor c = renza_cursor_at(m, kv->value_pos);
    return cursor_u32(&c, out);
}

bool renza_model_get_array(const renza_model *m, const char *key, renza_array_ref *out) {
    renza_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_ARRAY) return false;

    renza_cursor c = renza_cursor_at(m, kv->value_pos);
    if (!cursor_u32(&c, &out->type)) return false;
    if (!cursor_u64(&c, &out->len)) return false;
    out->data_pos = c.pos;
    return true;
}

typedef struct {
    const char *name;
    uint32_t block_elems;
    uint32_t block_bytes;
} gguf_type_info;

static const gguf_type_info gguf_types[] = {
    [0]  = {"f32",      1,   4},
    [1]  = {"f16",      1,   2},
    [2]  = {"q4_0",    32,  18},
    [3]  = {"q4_1",    32,  20},
    [6]  = {"q5_0",    32,  22},
    [7]  = {"q5_1",    32,  24},
    [8]  = {"q8_0",    32,  34},
    [9]  = {"q8_1",    32,  40},
    [10] = {"q2_k",   256,  84},
    [11] = {"q3_k",   256, 110},
    [12] = {"q4_k",   256, 144},
    [13] = {"q5_k",   256, 176},
    [14] = {"q6_k",   256, 210},
    [15] = {"q8_k",   256, 292},
    [16] = {"iq2_xxs",256,  66},
    [17] = {"iq2_xs", 256,  74},
    [18] = {"iq3_xxs",256,  98},
    [19] = {"iq1_s",  256, 110},
    [20] = {"iq4_nl", 256,  50},
    [21] = {"iq3_s",  256, 110},
    [22] = {"iq2_s",  256,  82},
    [23] = {"iq4_xs", 256, 136},
    [24] = {"i8",       1,   1},
    [25] = {"i16",      1,   2},
    [26] = {"i32",      1,   4},
    [27] = {"i64",      1,   8},
    [28] = {"f64",      1,   8},
    [29] = {"iq1_m",  256,  56},
    [30] = {"bf16",     1,   2},
    [39] = {"mxfp4",   32,  17},
};

static const gguf_type_info *tensor_type(uint32_t type) {
    uint32_t n = sizeof(gguf_types) / sizeof(gguf_types[0]);
    if (type >= n || gguf_types[type].name == NULL) return NULL;
    return &gguf_types[type];
}

const char *renza_tensor_type_name(uint32_t type) {
    const gguf_type_info *info = tensor_type(type);
    return info ? info->name : "unknown";
}

static bool tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes) {
    const gguf_type_info *info = tensor_type(type);
    if (!info || info->block_elems == 0) return false;
    uint64_t blocks = (elements + info->block_elems - 1) / info->block_elems;
    if (blocks > UINT64_MAX / info->block_bytes) return false;
    *bytes = blocks * info->block_bytes;
    return true;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    uint64_t rem = value % alignment;
    return rem == 0 ? value : value + alignment - rem;
}

void renza_tensor_table_free(renza_tensor_table *t) {
    if (!t) return;
    free(t->tensors);
    memset(t, 0, sizeof(*t));
}

/* Read the tensor directory and convert relative GGUF offsets to absolute
 * mmap offsets.  Tensor bytes are still never copied here. */
bool renza_model_read_tensors(const renza_model *m, renza_tensor_table *out, char *err, size_t errlen) {
    memset(out, 0, sizeof(*out));
    renza_cursor c = renza_cursor_at(m, m->tensor_dir_pos);
    /* As in parse_metadata: each tensor directory entry needs at least one byte
     * in the file, so reject a count larger than the bytes remaining before the
     * allocation. */
    if (m->n_tensors > c.size - c.pos) {
        renza_set_error(err, errlen, "GGUF tensor count exceeds file size");
        return false;
    }
    out->tensors = xcalloc((size_t)m->n_tensors, sizeof(out->tensors[0]));
    out->n = m->n_tensors;

    for (uint64_t i = 0; i < out->n; i++) {
        renza_tensor *t = &out->tensors[i];
        if (!renza_cursor_string(&c, &t->name)) goto fail_cursor;
        if (!cursor_u32(&c, &t->ndim)) goto fail_cursor;
        if (t->ndim == 0 || t->ndim > RENZA_GGUF_MAX_DIMS) {
            renza_set_error(err, errlen, "tensor has an unsupported number of dimensions");
            goto fail;
        }
        uint64_t elements = 1;
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (!cursor_u64(&c, &t->dim[d])) goto fail_cursor;
            if (t->dim[d] != 0 && elements > UINT64_MAX / t->dim[d]) {
                renza_set_error(err, errlen, "tensor element count overflow");
                goto fail;
            }
            elements *= t->dim[d];
        }
        if (!cursor_u32(&c, &t->type)) goto fail_cursor;
        if (!cursor_u64(&c, &t->offset)) goto fail_cursor;
        if (!tensor_nbytes(t->type, elements, &t->bytes)) t->bytes = 0;
    }

    uint32_t alignment = 32;
    if (!renza_model_get_u32(m, "general.alignment", &alignment) || alignment == 0) alignment = 32;
    out->data_pos = align_up(c.pos, alignment);

    for (uint64_t i = 0; i < out->n; i++) {
        renza_tensor *t = &out->tensors[i];
        if (t->offset > UINT64_MAX - out->data_pos) {
            renza_set_error(err, errlen, "tensor offset overflow");
            goto fail;
        }
        t->offset += out->data_pos;
        if (t->bytes != 0 && (t->offset > m->size || t->bytes > m->size - t->offset)) {
            renza_set_error(err, errlen, "tensor points outside GGUF file");
            goto fail;
        }
    }
    return true;

fail_cursor:
    renza_set_error(err, errlen, "%s", c.error);
fail:
    renza_tensor_table_free(out);
    return false;
}

const renza_tensor *renza_tensor_find(const renza_tensor_table *t, const char *name) {
    const renza_str want = {name, strlen(name)};
    for (uint64_t i = 0; i < t->n; i++) {
        if (renza_str_eq(t->tensors[i].name, want)) return &t->tensors[i];
    }
    return NULL;
}

void renza_model_close(renza_model *m) {
    if (!m) return;
    free(m->kv);
    if (m->map) munmap((void *)m->map, (size_t)m->size);
    if (m->fd >= 0) close(m->fd);
    memset(m, 0, sizeof(*m));
    m->fd = -1;
}

/* Read the GGUF metadata table.  Values stay in the mmap; we store offsets so
 * later validation can decode only the keys it needs. */
static bool parse_metadata(renza_model *m, renza_cursor *c) {
    /* n_kv comes from the header. Every entry consumes at least one byte in the
     * file, so a count larger than the bytes remaining cannot be real; reject it
     * before calloc so a tiny file can't request an enormous allocation. */
    if (m->n_kv > c->size - c->pos) {
        cursor_error(c, "GGUF metadata count exceeds file size");
        return false;
    }
    m->kv = calloc(m->n_kv ? (size_t)m->n_kv : 1, sizeof(m->kv[0]));
    if (!m->kv) {
        cursor_error(c, "GGUF metadata table is too large");
        return false;
    }

    for (uint64_t i = 0; i < m->n_kv; i++) {
        renza_kv *kv = &m->kv[i];
        if (!renza_cursor_string(c, &kv->key)) return false;
        if (!cursor_u32(c, &kv->type)) return false;
        kv->value_pos = c->pos;
        if (!skip_value(c, kv->type, 0)) return false;
    }
    return true;
}

bool renza_model_open(renza_model *m, const char *path, char *err, size_t errlen) {
    memset(m, 0, sizeof(*m));
    m->fd = open(path, O_RDONLY);
    if (m->fd == -1) {
        renza_set_error(err, errlen, "cannot open model '%s': %s", path, strerror(errno));
        goto fail;
    }

    struct stat st;
    if (fstat(m->fd, &st) == -1) {
        renza_set_error(err, errlen, "cannot stat model '%s': %s", path, strerror(errno));
        goto fail;
    }
    if (st.st_size < RENZA_GGUF_MIN_SIZE) {
        renza_set_error(err, errlen, "model file is too small to be GGUF");
        goto fail;
    }

    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, m->fd, 0);
    if (map == MAP_FAILED) {
        renza_set_error(err, errlen, "cannot mmap model '%s': %s", path, strerror(errno));
        goto fail;
    }
    m->map = map;
    m->size = (uint64_t)st.st_size;

    renza_cursor c = renza_cursor_at(m, 0);
    uint32_t magic, version;
    uint64_t n_tensors;
    if (!cursor_u32(&c, &magic)) goto fail_cursor;
    if (magic != RENZA_GGUF_MAGIC) {
        renza_set_error(err, errlen, "model is not a GGUF file");
        goto fail;
    }
    if (!cursor_u32(&c, &version)) goto fail_cursor;
    if (!cursor_u64(&c, &n_tensors)) goto fail_cursor;
    if (!cursor_u64(&c, &m->n_kv)) goto fail_cursor;
    if (version != RENZA_GGUF_VERSION) {
        renza_set_error(err, errlen, "only GGUF v3 is supported");
        goto fail;
    }
    if (!parse_metadata(m, &c)) goto fail_cursor;
    m->n_tensors = n_tensors;
    m->tensor_dir_pos = c.pos;
    return true;

fail_cursor:
    renza_set_error(err, errlen, "%s", c.error);
fail:
    renza_model_close(m);
    return false;
}
