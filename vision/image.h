#ifndef RENZA_VISION_IMAGE_H
#define RENZA_VISION_IMAGE_H

#include <stddef.h>
#include <stdint.h>

#define RENZA_IMAGE_MAX_ENCODED_BYTES (64u * 1024u * 1024u)
#define RENZA_IMAGE_MAX_DIMENSION 16384u
#define RENZA_IMAGE_MAX_PIXELS (64u * 1024u * 1024u)

/* The layouts are ds4's ds4_image and ds4_image_patches, field for field
 * (the preprocessing test calls ds4's own code through them). */
typedef struct {
    uint32_t width;
    uint32_t height;
    uint8_t *rgb;
    uint8_t fingerprint[32];  /* SHA-256 of width, height (host order) and rgb */
} renza_image;

typedef struct {
    uint32_t content_width;
    uint32_t content_height;
    uint32_t padded_width;
    uint32_t padded_height;
    uint32_t grid_width;
    uint32_t grid_height;
    uint32_t patch_count;
    uint32_t image_token_count;
    float *patches;
} renza_image_patches;

/* PNG or JPEG (EXIF orientation applied) to 8-bit RGB.  Return 1 on success,
 * 0 with a message in error. */
int renza_image_decode_memory(renza_image *out, const uint8_t *encoded, size_t encoded_len,
                              char *error, size_t error_cap);
int renza_image_decode_file(renza_image *out, const char *path, char *error, size_t error_cap);
void renza_image_free(renza_image *image);

/* Qwen3-VL style: resize to multiples of 32 within the token budget, normalize
 * with mean/std 0.5, emit 3*16*16 patches in 2x2 merge-window order. */
int renza_image_preprocess_qwen4(renza_image_patches *out, const renza_image *image,
                                 uint32_t min_image_tokens, uint32_t max_image_tokens,
                                 char *error, size_t error_cap);
void renza_image_patches_free(renza_image_patches *patches);

#endif
