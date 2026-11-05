/* renza-vision --mmproj PATH [--max-tokens N] IMAGE OUT.sve
 *
 * Encodes one image and writes strata-vision's SVE1 file: int32 LE
 * {0x31455653, n_tokens, nx, ny, n_embd}, then float32 [n_tokens][n_embd]
 * with row i at (x = i % nx, y = i / nx).  Prints "OK n nx ny ms" (ms from
 * reading the image to the written file, as strata-vision). */
#include "renza_vision.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SVE_MAGIC 0x31455653
#define DEFAULT_MAX_TOKENS 1024u
#define MAX_IMAGE_BYTES (64u * 1024u * 1024u)

static void usage(void) {
    fprintf(stderr, "usage: renza-vision --mmproj PATH [--max-tokens N] IMAGE OUT.sve\n");
    exit(2);
}

static void fail(const char *msg) {
    fprintf(stderr, "renza-vision: %s\n", msg);
    exit(1);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static uint8_t *read_image(const char *path, size_t *len, char *err, size_t errlen) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        snprintf(err, errlen, "cannot open %s", path);
        return NULL;
    }
    uint8_t *data = NULL;
    long end = -1;
    if (fseek(fp, 0, SEEK_END) == 0) end = ftell(fp);
    if (end <= 0 || (unsigned long)end > MAX_IMAGE_BYTES || fseek(fp, 0, SEEK_SET) != 0) {
        snprintf(err, errlen, "%s is empty, unreadable or over 64 MiB", path);
    } else {
        data = malloc((size_t)end);
        if (!data) fail("out of memory");
        if (fread(data, 1, (size_t)end, fp) != (size_t)end) {
            snprintf(err, errlen, "cannot read %s", path);
            free(data);
            data = NULL;
        }
    }
    fclose(fp);
    *len = data ? (size_t)end : 0;
    return data;
}

static bool write_sve(const char *path, const renza_vision_rows *rows) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return false;
    const int32_t header[5] = {
        SVE_MAGIC, (int32_t)rows->token_count, (int32_t)rows->grid_width, (int32_t)rows->grid_height,
        (int32_t)rows->embd_dim,
    };
    const size_t floats = (size_t)rows->token_count * rows->embd_dim;
    const bool ok = fwrite(header, sizeof(header), 1, fp) == 1 &&
                    fwrite(rows->rows, sizeof(float), floats, fp) == floats;
    return fclose(fp) == 0 && ok;
}

int main(int argc, char **argv) {
    const char *mmproj = NULL, *image = NULL, *out = NULL;
    unsigned long max_tokens = DEFAULT_MAX_TOKENS;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mmproj") == 0 && i + 1 < argc) {
            mmproj = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            char *end = NULL;
            max_tokens = strtoul(argv[++i], &end, 10);
            if (!end || *end || max_tokens > UINT32_MAX) usage();
        } else if (argv[i][0] == '-' && argv[i][1] != '\0') {
            usage();
        } else if (!image) {
            image = argv[i];
        } else if (!out) {
            out = argv[i];
        } else {
            usage();
        }
    }
    if (!mmproj || !image || !out) usage();

    char err[512] = {0};
    renza_vision_encoder *enc = renza_vision_encoder_open(mmproj, (uint32_t)max_tokens, err, sizeof(err));
    if (!enc) fail(err);

    const double t0 = now_ms();
    size_t len = 0;
    uint8_t *data = read_image(image, &len, err, sizeof(err));
    if (!data) fail(err);
    renza_vision_rows rows;
    if (!renza_vision_encoder_encode(enc, data, len, &rows, err, sizeof(err))) fail(err);
    free(data);
    if (!write_sve(out, &rows)) {
        snprintf(err, sizeof(err), "cannot write %s", out);
        fail(err);
    }
    printf("OK %u %u %u %.0f\n", rows.token_count, rows.grid_width, rows.grid_height, now_ms() - t0);
    renza_vision_rows_free(&rows);
    renza_vision_encoder_close(enc);
    return 0;
}
