/* renza's image encoder: DwarfStar's native Qwen3.8 (Qwen3-VL) vision tower
 * on CUDA, over a llama.cpp mmproj GGUF (clip / qwen3vl_merger). */
#ifndef RENZA_VISION_H
#define RENZA_VISION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RENZA_VISION_MIN_TOKENS 64u
#define RENZA_VISION_MAX_TOKENS_LIMIT 16383u
#define RENZA_VISION_IMAGE_FINGERPRINT_BYTES 32u

typedef struct renza_vision_encoder renza_vision_encoder;

/* One encoded image.  rows is token_count x embd_dim floats from malloc
 * (released with free()).  Rows are in raster order over the merged
 * grid: row i is at x = i % grid_width, y = i / grid_width (ds4's
 * 2x2-window patch order makes merged row i the window at that position).
 * fingerprint is the SHA-256 of the decoded, oriented image. */
typedef struct {
    float *rows;
    uint32_t token_count;
    uint32_t embd_dim;
    uint32_t grid_width;
    uint32_t grid_height;
    uint8_t fingerprint[RENZA_VISION_IMAGE_FINGERPRINT_BYTES];
} renza_vision_rows;

/* Reads the mmproj, uploads every weight to the GPU and reserves the work
 * memory for an image of max_tokens (in [RENZA_VISION_MIN_TOKENS,
 * RENZA_VISION_MAX_TOKENS_LIMIT]), so the encoder's VRAM is taken at open.
 * Only one encoder may be open in a process.  NULL with a message in err
 * on failure. */
renza_vision_encoder *renza_vision_encoder_open(const char *mmproj_path, uint32_t max_tokens,
                                                char *err, size_t errlen);
/* The width of a row (the mmproj's projection_dim). */
uint32_t renza_vision_encoder_embd_dim(const renza_vision_encoder *enc);
/* Decodes a PNG or JPEG held in memory, resizes it to between
 * RENZA_VISION_MIN_TOKENS and the encoder's max tokens and encodes it.  On
 * success *out is new rows the caller owns; otherwise *out is zeroed and
 * err holds the reason. */
bool renza_vision_encoder_encode(renza_vision_encoder *enc, const uint8_t *encoded, size_t encoded_len,
                                 renza_vision_rows *out, char *err, size_t errlen);
/* Frees rows and zeroes *rows.  NULL is a no-op. */
void renza_vision_rows_free(renza_vision_rows *rows);
void renza_vision_encoder_close(renza_vision_encoder *enc);

#ifdef __cplusplus
}
#endif

#endif
