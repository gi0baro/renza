/* The Qwen3.8 vision tower's weights and its GPU side, from DwarfStar's
 * ds4_qwen4_vision.h and ds4_qwen4_cuda.cuh (MIT). */
#ifndef RENZA_VISION_ENCODER_H
#define RENZA_VISION_ENCODER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RENZA_QWEN4_VISION_LAYERS 27

/* Weight fields are absolute offsets into the mmproj file; *_type fields
 * are GGUF tensor types. */
typedef struct {
    uint64_t ln1_w, ln1_b, qkv_w, qkv_b, out_w, out_b, ln2_w, ln2_b, up_w, up_b, down_w, down_b;
    uint32_t qkv_type, out_type, up_type, down_type;
} renza_qwen4_vision_layer_weights;

typedef struct {
    uint64_t patch_w0, patch_w1, patch_b, pos_embd, post_ln_w, post_ln_b, mm0_w, mm0_b, mm2_w, mm2_b;
    uint32_t mm0_type, mm2_type, patch_type;
    uint32_t n_embd, n_ff, n_head, n_patch, n_merge, n_pos_side, n_out;
    float eps;
    renza_qwen4_vision_layer_weights layer[RENZA_QWEN4_VISION_LAYERS];
} renza_qwen4_vision_weights;

/* The GPU state is process-wide: one open encoder at a time.  open copies
 * file[data_pos, size) (every tensor) to the device and sizes the work
 * buffers for max_patches.  All return 1 on success, 0 with a message. */
int renza_vision_gpu_open(const uint8_t *file, uint64_t data_pos, uint64_t size,
                          const renza_qwen4_vision_weights *w, uint32_t max_patches,
                          char *err, size_t errlen);
/* out gets [N / 4][n_out] floats; patches are [N][3 * P * P] and pos_embed
 * [N][n_embd], both in 2x2 merge-window order. */
int renza_vision_gpu_encode(float *out, const float *patches, const float *pos_embed,
                            uint32_t N, uint32_t grid_w, const renza_qwen4_vision_weights *w,
                            char *err, size_t errlen);
void renza_vision_gpu_close(void);

#ifdef __cplusplus
}
#endif

#endif
