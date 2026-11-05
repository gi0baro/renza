/* The Qwen3.8 vision encoder's host side, extracted from DwarfStar's ds4.c
 * (MIT): the mmproj binder, the position table and the per-image encode. */
#include "renza_vision.h"

#include "encoder.h"
#include "gguf.h"
#include "image.h"

#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TENSOR_F32 = 0,
    TENSOR_F16 = 1,
    TENSOR_Q8_0 = 8,
    TENSOR_BF16 = 30,
};

#define PATCHES_PER_TOKEN 4u
#define DEFAULT_LAYER_NORM_EPS 1e-6f

struct renza_vision_encoder {
    renza_qwen4_vision_weights w;
    float *pos_table;  /* v.position_embd.weight: [n_pos_side^2][n_embd] */
    uint32_t max_tokens;
};

/* ds4's binder exits on the first bad tensor; here the first failure is
 * recorded and the remaining lookups are no-ops. */
typedef struct {
    const renza_model *m;
    const renza_tensor_table *t;
    char *err;
    size_t errlen;
    bool failed;
} binder;

__attribute__((format(printf, 2, 3)))
static void bind_fail(binder *b, const char *fmt, ...) {
    if (b->failed) return;
    b->failed = true;
    if (!b->err || b->errlen == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b->err, b->errlen, fmt, ap);
    va_end(ap);
}

static bool str_is(renza_str s, const char *want) {
    return s.len == strlen(want) && memcmp(s.ptr, want, (size_t)s.len) == 0;
}

static bool model_get_f32_compat(const renza_model *m, const char *key, float *out) {
    for (uint64_t i = 0; i < m->n_kv; i++) {
        const renza_kv *kv = &m->kv[i];
        if (!str_is(kv->key, key)) continue;
        const uint8_t *p = m->map + kv->value_pos;
        switch (kv->type) {
        case GGUF_VALUE_FLOAT32: memcpy(out, p, sizeof(*out)); return true;
        case GGUF_VALUE_FLOAT64: { double v; memcpy(&v, p, sizeof(v)); *out = (float)v; return true; }
        case GGUF_VALUE_UINT32: { uint32_t v; memcpy(&v, p, sizeof(v)); *out = (float)v; return true; }
        case GGUF_VALUE_INT32: { int32_t v; memcpy(&v, p, sizeof(v)); *out = (float)v; return true; }
        default: return false;
        }
    }
    return false;
}

static uint32_t required_u32(binder *b, const char *key) {
    uint32_t v = 0;
    if (!renza_model_get_u32(b->m, key, &v)) bind_fail(b, "required metadata key is missing: %s", key);
    return v;
}

static void config_expect_u32(binder *b, const char *name, uint32_t got, uint32_t expected) {
    if (got != expected) bind_fail(b, "expected %s=%u, got %u", name, expected, got);
}

static const renza_tensor *required_tensor(binder *b, const char *name) {
    const renza_tensor *t = renza_tensor_find(b->t, name);
    if (!t) bind_fail(b, "required tensor is missing: %s", name);
    return t;
}

static uint64_t required_offset(binder *b, const char *name, uint32_t type, uint32_t ndim,
                                const uint64_t *dims) {
    const renza_tensor *t = required_tensor(b, name);
    if (!t) return 0;
    if (t->type != type || t->ndim != ndim) {
        bind_fail(b, "vision tensor %s has type %s/rank %u, expected %s/rank %u", name,
                  renza_tensor_type_name(t->type), t->ndim, renza_tensor_type_name(type), ndim);
        return 0;
    }
    for (uint32_t d = 0; d < ndim; d++) {
        if (t->dim[d] == dims[d]) continue;
        bind_fail(b, "vision tensor %s has dim[%u]=%" PRIu64 ", expected %" PRIu64, name, d, t->dim[d],
                  dims[d]);
        return 0;
    }
    return t->offset;
}

/* Qwen3.8 vision: llama.cpp's Qwen3-VL mmproj (clip / qwen3vl_merger).
 * Matmul weights may be f32, f16, bf16 or q8_0; everything else is f32. */
static uint64_t qwen4_vision_dense(binder *b, const char *name, uint64_t in_dim, uint64_t out_dim,
                                   uint32_t *type) {
    const renza_tensor *t = required_tensor(b, name);
    if (!t) return 0;
    uint64_t in = 1;
    for (uint32_t d = 0; d + 1 < t->ndim; d++) in *= t->dim[d];
    if ((t->type != TENSOR_F32 && t->type != TENSOR_F16 && t->type != TENSOR_BF16 &&
         t->type != TENSOR_Q8_0) ||
        t->ndim < 2 || in != in_dim || t->dim[t->ndim - 1] != out_dim) {
        bind_fail(b, "vision tensor %s has type %s/shape [%" PRIu64 " x %" PRIu64 "], expected "
                  "f32|f16|bf16|q8_0 [%" PRIu64 " x %" PRIu64 "]", name, renza_tensor_type_name(t->type),
                  in, t->ndim ? t->dim[t->ndim - 1] : 0, in_dim, out_dim);
        return 0;
    }
    *type = t->type;
    return t->offset;
}

static void qwen4_vision_weights_bind(binder *b, renza_qwen4_vision_weights *w) {
    const renza_model *m = b->m;
    renza_str arch = {0}, proj = {0};
    if (!renza_model_get_string(m, "general.architecture", &arch) || !str_is(arch, "clip") ||
        !renza_model_get_string(m, "clip.projector_type", &proj) || !str_is(proj, "qwen3vl_merger")) {
        bind_fail(b, "the vision file is not a Qwen3-VL mmproj GGUF (clip / qwen3vl_merger)");
        return;
    }
    memset(w, 0, sizeof(*w));
    w->n_embd = required_u32(b, "clip.vision.embedding_length");
    w->n_ff = required_u32(b, "clip.vision.feed_forward_length");
    w->n_head = required_u32(b, "clip.vision.attention.head_count");
    w->n_patch = required_u32(b, "clip.vision.patch_size");
    w->n_merge = required_u32(b, "clip.vision.spatial_merge_size");
    w->n_out = required_u32(b, "clip.vision.projection_dim");
    config_expect_u32(b, "vision block_count", required_u32(b, "clip.vision.block_count"),
                      RENZA_QWEN4_VISION_LAYERS);
    config_expect_u32(b, "vision embedding_length", w->n_embd, 1152u);
    config_expect_u32(b, "vision head_count", w->n_head, 16u);
    config_expect_u32(b, "vision patch_size", w->n_patch, 16u);
    config_expect_u32(b, "vision spatial_merge_size", w->n_merge, 2u);
    if (b->failed) return;
    w->eps = DEFAULT_LAYER_NORM_EPS;
    (void)model_get_f32_compat(m, "clip.vision.attention.layer_norm_epsilon", &w->eps);
    const uint64_t E = w->n_embd, FF = w->n_ff, P = w->n_patch, ME = E * w->n_merge * w->n_merge, O = w->n_out;
    const uint64_t d_e[1] = { E }, d_ff[1] = { FF }, d_3e[1] = { 3u * E };
    const uint64_t d_me[1] = { ME }, d_o[1] = { O };
    const renza_tensor *pos = required_tensor(b, "v.position_embd.weight");
    if (!pos) return;
    if (pos->type != TENSOR_F32 || pos->ndim != 2 || pos->dim[0] != E) {
        bind_fail(b, "vision tensor v.position_embd.weight must be f32 [embd, positions]");
        return;
    }
    w->n_pos_side = (uint32_t)sqrt((double)pos->dim[1]);
    if ((uint64_t)w->n_pos_side * w->n_pos_side != pos->dim[1] || w->n_pos_side < 2) {
        bind_fail(b, "vision position table is not square");
        return;
    }
    w->pos_embd = pos->offset;
    uint32_t patch_type1 = 0;
    w->patch_w0 = qwen4_vision_dense(b, "v.patch_embd.weight", 3u * P * P, E, &w->patch_type);
    w->patch_w1 = qwen4_vision_dense(b, "v.patch_embd.weight.1", 3u * P * P, E, &patch_type1);
    if (!b->failed && (patch_type1 != w->patch_type || w->patch_type == TENSOR_Q8_0))
        bind_fail(b, "vision patch weights must both be f32, f16 or bf16");
    w->patch_b = required_offset(b, "v.patch_embd.bias", TENSOR_F32, 1, d_e);
    w->post_ln_w = required_offset(b, "v.post_ln.weight", TENSOR_F32, 1, d_e);
    w->post_ln_b = required_offset(b, "v.post_ln.bias", TENSOR_F32, 1, d_e);
    w->mm0_w = qwen4_vision_dense(b, "mm.0.weight", ME, ME, &w->mm0_type);
    w->mm0_b = required_offset(b, "mm.0.bias", TENSOR_F32, 1, d_me);
    w->mm2_w = qwen4_vision_dense(b, "mm.2.weight", ME, O, &w->mm2_type);
    w->mm2_b = required_offset(b, "mm.2.bias", TENSOR_F32, 1, d_o);
    for (uint32_t il = 0; il < RENZA_QWEN4_VISION_LAYERS; il++) {
        renza_qwen4_vision_layer_weights *l = &w->layer[il];
        char name[96];
#define VIS_NAME(suffix_) (snprintf(name, sizeof(name), "v.blk.%u.%s", il, suffix_), name)
        l->ln1_w = required_offset(b, VIS_NAME("ln1.weight"), TENSOR_F32, 1, d_e);
        l->ln1_b = required_offset(b, VIS_NAME("ln1.bias"), TENSOR_F32, 1, d_e);
        l->qkv_w = qwen4_vision_dense(b, VIS_NAME("attn_qkv.weight"), E, 3u * E, &l->qkv_type);
        l->qkv_b = required_offset(b, VIS_NAME("attn_qkv.bias"), TENSOR_F32, 1, d_3e);
        l->out_w = qwen4_vision_dense(b, VIS_NAME("attn_out.weight"), E, E, &l->out_type);
        l->out_b = required_offset(b, VIS_NAME("attn_out.bias"), TENSOR_F32, 1, d_e);
        l->ln2_w = required_offset(b, VIS_NAME("ln2.weight"), TENSOR_F32, 1, d_e);
        l->ln2_b = required_offset(b, VIS_NAME("ln2.bias"), TENSOR_F32, 1, d_e);
        l->up_w = qwen4_vision_dense(b, VIS_NAME("ffn_up.weight"), E, FF, &l->up_type);
        l->up_b = required_offset(b, VIS_NAME("ffn_up.bias"), TENSOR_F32, 1, d_ff);
        l->down_w = qwen4_vision_dense(b, VIS_NAME("ffn_down.weight"), FF, E, &l->down_type);
        l->down_b = required_offset(b, VIS_NAME("ffn_down.bias"), TENSOR_F32, 1, d_e);
#undef VIS_NAME
    }
}

/* Learned 2D position table (side x side) resampled to the patch grid with
 * bilinear, align_corners interpolation; rows in 2x2 window order. */
static void qwen4_vision_pos_embed(const float *table, uint32_t side, uint32_t n_embd,
                                   uint32_t grid_h, uint32_t grid_w, float *out) {
    size_t row = 0;
    for (uint32_t by = 0; by < grid_h / 2; by++) {
        for (uint32_t bx = 0; bx < grid_w / 2; bx++) {
            for (uint32_t my = 0; my < 2; my++) {
                for (uint32_t mx = 0; mx < 2; mx++, row++) {
                    const uint32_t y = by * 2 + my, x = bx * 2 + mx;
                    const double sy = grid_h > 1 ? (double)y * (side - 1) / (grid_h - 1) : 0.0;
                    const double sx = grid_w > 1 ? (double)x * (side - 1) / (grid_w - 1) : 0.0;
                    const uint32_t y0 = (uint32_t)sy, x0 = (uint32_t)sx;
                    const uint32_t y1 = y0 + 1 < side ? y0 + 1 : y0, x1 = x0 + 1 < side ? x0 + 1 : x0;
                    const float wy = (float)(sy - y0), wx = (float)(sx - x0);
                    const float *t00 = table + ((size_t)y0 * side + x0) * n_embd;
                    const float *t01 = table + ((size_t)y0 * side + x1) * n_embd;
                    const float *t10 = table + ((size_t)y1 * side + x0) * n_embd;
                    const float *t11 = table + ((size_t)y1 * side + x1) * n_embd;
                    float *dst = out + row * n_embd;
                    for (uint32_t d = 0; d < n_embd; d++) {
                        dst[d] = (1.0f - wy) * ((1.0f - wx) * t00[d] + wx * t01[d]) +
                                 wy * ((1.0f - wx) * t10[d] + wx * t11[d]);
                    }
                }
            }
        }
    }
}

/* preprocess + encode one image; out gets malloc'd [tokens][n_out] embeddings */
static int qwen4_vision_encode_image(const renza_vision_encoder *enc, const renza_image *image,
                                     uint32_t min_tokens, uint32_t max_tokens, float **out,
                                     renza_image_patches *patches, char *error, size_t error_cap) {
    const renza_qwen4_vision_weights *w = &enc->w;
    *out = NULL;
    if (!renza_image_preprocess_qwen4(patches, image, min_tokens, max_tokens, error, error_cap)) return 0;
    const uint64_t n = patches->patch_count;
    float *pos = malloc((size_t)n * w->n_embd * sizeof(float));
    float *emb = malloc((size_t)patches->image_token_count * w->n_out * sizeof(float));
    if (!pos || !emb) {
        free(pos);
        free(emb);
        if (error && error_cap) snprintf(error, error_cap, "unable to allocate vision buffers");
        return 0;
    }
    qwen4_vision_pos_embed(enc->pos_table, w->n_pos_side, w->n_embd, patches->grid_height, patches->grid_width,
                           pos);
    int ok = renza_vision_gpu_encode(emb, patches->patches, pos, patches->patch_count, patches->grid_width, w,
                                     error, error_cap);
    free(pos);
    if (!ok) {
        free(emb);
        return 0;
    }
    *out = emb;
    return 1;
}

/* One encode of the largest image the token limit allows, so the scratch
 * and cuBLAS's own allocations are made at open too. */
static bool warm_up(const renza_vision_encoder *enc, char *err, size_t errlen) {
    const renza_qwen4_vision_weights *w = &enc->w;
    const uint32_t patches = enc->max_tokens * PATCHES_PER_TOKEN, grid_w = w->n_merge;
    const size_t patch_floats = (size_t)patches * 3 * w->n_patch * w->n_patch;
    float *in = calloc(patch_floats, sizeof(float));
    float *pos = calloc((size_t)patches * w->n_embd, sizeof(float));
    float *out = malloc((size_t)enc->max_tokens * w->n_out * sizeof(float));
    bool ok = in && pos && out;
    if (!ok) snprintf(err, errlen, "unable to allocate vision buffers");
    if (ok) ok = renza_vision_gpu_encode(out, in, pos, patches, grid_w, w, err, errlen);
    free(in);
    free(pos);
    free(out);
    return ok;
}

static bool copy_pos_table(renza_vision_encoder *enc, const renza_model *m) {
    const renza_qwen4_vision_weights *w = &enc->w;
    const size_t floats = (size_t)w->n_pos_side * w->n_pos_side * w->n_embd;
    enc->pos_table = malloc(floats * sizeof(float));
    if (!enc->pos_table) return false;
    memcpy(enc->pos_table, m->map + w->pos_embd, floats * sizeof(float));
    return true;
}

renza_vision_encoder *renza_vision_encoder_open(const char *mmproj_path, uint32_t max_tokens,
                                                char *err, size_t errlen) {
    if (max_tokens < RENZA_VISION_MIN_TOKENS || max_tokens > RENZA_VISION_MAX_TOKENS_LIMIT) {
        snprintf(err, errlen, "vision max tokens must be in [%u, %u], got %u", RENZA_VISION_MIN_TOKENS,
                 RENZA_VISION_MAX_TOKENS_LIMIT, max_tokens);
        return NULL;
    }
    renza_model m;
    if (!renza_model_open(&m, mmproj_path, err, errlen)) return NULL;
    renza_tensor_table t;
    if (!renza_model_read_tensors(&m, &t, err, errlen)) {
        renza_model_close(&m);
        return NULL;
    }
    renza_vision_encoder *enc = calloc(1, sizeof(*enc));
    bool ok = enc != NULL;
    if (!ok) snprintf(err, errlen, "unable to allocate the vision encoder");
    if (ok) {
        enc->max_tokens = max_tokens;
        binder b = { .m = &m, .t = &t, .err = err, .errlen = errlen, .failed = false };
        qwen4_vision_weights_bind(&b, &enc->w);
        ok = !b.failed;
    }
    if (ok && !copy_pos_table(enc, &m)) {
        snprintf(err, errlen, "unable to allocate the vision position table");
        ok = false;
    }
    const bool gpu_open = ok && renza_vision_gpu_open(m.map, t.data_pos, m.size, &enc->w,
                                                      max_tokens * PATCHES_PER_TOKEN, err, errlen);
    ok = gpu_open && warm_up(enc, err, errlen);
    renza_tensor_table_free(&t);
    renza_model_close(&m);
    if (ok) return enc;
    if (gpu_open) renza_vision_gpu_close();
    if (enc) free(enc->pos_table);
    free(enc);
    return NULL;
}

uint32_t renza_vision_encoder_embd_dim(const renza_vision_encoder *enc) {
    return enc->w.n_out;
}

bool renza_vision_encoder_encode(renza_vision_encoder *enc, const uint8_t *encoded, size_t encoded_len,
                                 renza_vision_rows *out, char *err, size_t errlen) {
    memset(out, 0, sizeof(*out));
    renza_image image = {0};
    if (!renza_image_decode_memory(&image, encoded, encoded_len, err, errlen)) return false;
    renza_image_patches patches = {0};
    float *rows = NULL;
    const bool ok = qwen4_vision_encode_image(enc, &image, RENZA_VISION_MIN_TOKENS, enc->max_tokens, &rows,
                                              &patches, err, errlen);
    if (ok) {
        out->rows = rows;
        out->token_count = patches.image_token_count;
        out->embd_dim = enc->w.n_out;
        out->grid_width = patches.grid_width / enc->w.n_merge;
        out->grid_height = patches.grid_height / enc->w.n_merge;
        memcpy(out->fingerprint, image.fingerprint, sizeof(out->fingerprint));
    }
    renza_image_patches_free(&patches);
    renza_image_free(&image);
    return ok;
}

void renza_vision_rows_free(renza_vision_rows *rows) {
    if (!rows) return;
    free(rows->rows);
    memset(rows, 0, sizeof(*rows));
}

void renza_vision_encoder_close(renza_vision_encoder *enc) {
    if (!enc) return;
    renza_vision_gpu_close();
    free(enc->pos_table);
    free(enc);
}
