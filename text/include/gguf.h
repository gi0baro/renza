#ifndef RENZA_GGUF_H
#define RENZA_GGUF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    const char *ptr;
    uint64_t len;
} renza_str;

typedef struct {
    const uint8_t *base;
    uint64_t size;
    uint64_t pos;
    char error[256];
} renza_cursor;

enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
};

typedef struct {
    renza_str key;
    uint32_t type;
    uint64_t value_pos;
} renza_kv;

/* The metadata section of a GGUF file; the tensor directory is read only by
 * renza_model_read_tensors. */
typedef struct renza_model {
    int fd;
    const uint8_t *map;
    uint64_t size;
    uint64_t n_kv;
    renza_kv *kv;
    uint64_t n_tensors;
    uint64_t tensor_dir_pos;
} renza_model;

#define RENZA_GGUF_MAX_DIMS 8

typedef struct {
    renza_str name;
    uint32_t ndim;
    uint64_t dim[RENZA_GGUF_MAX_DIMS];
    uint32_t type;
    uint64_t offset;   /* absolute, from the start of the file */
    uint64_t bytes;    /* 0 for a type the reader does not know */
} renza_tensor;

/* Names point into the model's map: the table must not outlive the model. */
typedef struct {
    uint64_t n;
    renza_tensor *tensors;
    uint64_t data_pos;  /* start of the tensor data section */
} renza_tensor_table;

typedef struct {
    uint32_t type;
    uint64_t len;
    uint64_t data_pos;
} renza_array_ref;

/* On failure m holds nothing to close and err holds the reason. */
bool renza_model_open(renza_model *m, const char *path, char *err, size_t errlen);
void renza_model_close(renza_model *m);
renza_cursor renza_cursor_at(const renza_model *m, uint64_t pos);
bool renza_cursor_string(renza_cursor *c, renza_str *s);
bool renza_model_get_string(const renza_model *m, const char *key, renza_str *out);
bool renza_model_get_u32(const renza_model *m, const char *key, uint32_t *out);
bool renza_model_get_array(const renza_model *m, const char *key, renza_array_ref *out);

/* On failure *out holds nothing to free and err holds the reason. */
bool renza_model_read_tensors(const renza_model *m, renza_tensor_table *out, char *err, size_t errlen);
void renza_tensor_table_free(renza_tensor_table *t);
const renza_tensor *renza_tensor_find(const renza_tensor_table *t, const char *name);
const char *renza_tensor_type_name(uint32_t type);

#endif
