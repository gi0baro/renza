/* Decode and Qwen3.8 preprocessing.  With --ds4 LIB, every IMAGE is also run
 * through ds4's own ds4_image.c (built as LIB) and the results compared byte
 * for byte; IMAGE=TOKENS:GW:GH adds the expected token count and merged grid. */
#include "image.h"

#include <dlfcn.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIN_TOKENS 64u
#define MAX_TOKENS 1024u
#define MAX_ASPECT 200u
#define MERGE 2u

static int failures;

#define CHECK(cond, ...) do { \
    if (!(cond)) { failures++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                   fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } \
} while (0)

typedef int (*decode_fn)(renza_image *, const uint8_t *, size_t, char *, size_t);
typedef void (*image_free_fn)(renza_image *);
typedef int (*preprocess_fn)(renza_image_patches *, const renza_image *, uint32_t, uint32_t, char *, size_t);
typedef void (*patches_free_fn)(renza_image_patches *);

typedef struct {
    decode_fn decode;
    image_free_fn image_free;
    preprocess_fn preprocess;
    patches_free_fn patches_free;
} image_api;

static const image_api renza_api = {
    renza_image_decode_memory, renza_image_free, renza_image_preprocess_qwen4, renza_image_patches_free,
};

static bool gray_image(renza_image *img, uint32_t width, uint32_t height) {
    memset(img, 0, sizeof(*img));
    img->rgb = malloc((size_t)width * height * 3);
    if (!img->rgb) return false;
    memset(img->rgb, 128, (size_t)width * height * 3);
    img->width = width;
    img->height = height;
    return true;
}

static void check_tokens(uint32_t width, uint32_t height, uint32_t tokens, uint32_t gw, uint32_t gh) {
    renza_image img;
    renza_image_patches p = {0};
    char err[256] = {0};
    CHECK(gray_image(&img, width, height), "allocation");
    const int ok = renza_image_preprocess_qwen4(&p, &img, MIN_TOKENS, MAX_TOKENS, err, sizeof(err));
    CHECK(ok, "%ux%u: %s", width, height, err);
    CHECK(p.image_token_count == tokens && p.grid_width == gw * MERGE && p.grid_height == gh * MERGE,
          "%ux%u: %u tokens, merged grid %ux%u; expected %u, %ux%u", width, height, p.image_token_count,
          p.grid_width / MERGE, p.grid_height / MERGE, tokens, gw, gh);
    renza_image_patches_free(&p);
    renza_image_free(&img);
}

static void check_aspect_refused(void) {
    renza_image img;
    renza_image_patches p = {0};
    char err[256] = {0};
    const uint32_t height = 2, width = height * MAX_ASPECT + 1;
    CHECK(gray_image(&img, width, height), "allocation");
    CHECK(!renza_image_preprocess_qwen4(&p, &img, MIN_TOKENS, MAX_TOKENS, err, sizeof(err)),
          "%ux%u was accepted", width, height);
    CHECK(strstr(err, "aspect") != NULL, "unexpected error: %s", err);
    renza_image_free(&img);
}

static uint8_t *read_file(const char *path, size_t *len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    uint8_t *data = NULL;
    if (fseek(fp, 0, SEEK_END) == 0) {
        const long end = ftell(fp);
        data = end > 0 && fseek(fp, 0, SEEK_SET) == 0 ? malloc((size_t)end) : NULL;
        if (data && fread(data, 1, (size_t)end, fp) != (size_t)end) {
            free(data);
            data = NULL;
        }
        *len = data ? (size_t)end : 0;
    }
    fclose(fp);
    return data;
}

/* The fixture is stored 300x200 with EXIF orientation 6 (rotate 90 CW). */
static void check_jpeg_orientation(const char *path) {
    size_t len = 0;
    uint8_t *data = read_file(path, &len);
    CHECK(data, "cannot read %s", path);
    if (!data) return;
    renza_image img = {0};
    char err[256] = {0};
    CHECK(renza_image_decode_memory(&img, data, len, err, sizeof(err)), "%s: %s", path, err);
    CHECK(img.width == 200 && img.height == 300, "%s decoded as %ux%u, expected 200x300", path, img.width,
          img.height);
    renza_image_free(&img);
    free(data);
}

static int check_jpeg_rejected(const char *path) {
    renza_image image = {0};
    char error[160] = {0};
    if (renza_image_decode_file(&image, path, error, sizeof(error))) {
        fprintf(stderr, "%s: malformed JPEG was accepted\n", path);
        renza_image_free(&image);
        return 0;
    }
    return 1;
}

/* 1x1 progressive grayscale JPEG with a single scan whose SOS carries the
 * given Ss/Se. Used for scan headers that must be rejected. */
static int check_jpeg_sos_rejected(uint8_t ss, uint8_t se) {
    uint8_t b[128];
    size_t n = 0;
    static const uint8_t sof2[] = {0xFF, 0xC2, 0x00, 0x0B, 0x08, 0x00, 0x01,
                                   0x00, 0x01, 0x01, 0x01, 0x11, 0x00};
    b[n++] = 0xFF; b[n++] = 0xD8;
    b[n++] = 0xFF; b[n++] = 0xDB; b[n++] = 0x00; b[n++] = 0x43; b[n++] = 0x00;
    for (int i = 0; i < 64; i++) b[n++] = 0x01;
    memcpy(b + n, sof2, sizeof(sof2)); n += sizeof(sof2);
    /* DC table 0: one 1-bit code for category 0. */
    b[n++] = 0xFF; b[n++] = 0xC4; b[n++] = 0x00; b[n++] = 0x14; b[n++] = 0x00;
    b[n++] = 0x01;
    for (int i = 1; i < 16; i++) b[n++] = 0x00;
    b[n++] = 0x00;
    b[n++] = 0xFF; b[n++] = 0xDA; b[n++] = 0x00; b[n++] = 0x08; b[n++] = 0x01;
    b[n++] = 0x01; b[n++] = 0x00; b[n++] = ss; b[n++] = se; b[n++] = 0x00;
    b[n++] = 0x00;
    b[n++] = 0xFF; b[n++] = 0xD9;

    renza_image image = {0};
    char error[160] = {0};
    if (renza_image_decode_memory(&image, b, n, error, sizeof(error))) {
        fprintf(stderr, "SOS Ss=%u Se=%u was accepted\n", ss, se);
        renza_image_free(&image);
        return 0;
    }
    return 1;
}

static bool load_ds4(const char *lib, image_api *api) {
    void *h = dlopen(lib, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!h) {
        fprintf(stderr, "FAIL: %s\n", dlerror());
        return false;
    }
    *(void **)&api->decode = dlsym(h, "ds4_image_decode_memory");
    *(void **)&api->image_free = dlsym(h, "ds4_image_free");
    *(void **)&api->preprocess = dlsym(h, "ds4_image_preprocess_qwen4");
    *(void **)&api->patches_free = dlsym(h, "ds4_image_patches_free");
    return api->decode && api->image_free && api->preprocess && api->patches_free;
}

static void compare_image(const image_api *ds4, const char *arg) {
    char path[4096];
    snprintf(path, sizeof(path), "%s", arg);
    unsigned want_tokens = 0, want_gw = 0, want_gh = 0;
    char *expect = strchr(path, '=');
    if (expect) {
        *expect++ = '\0';
        CHECK(sscanf(expect, "%u:%u:%u", &want_tokens, &want_gw, &want_gh) == 3, "bad expectation %s", expect);
    }
    size_t len = 0;
    uint8_t *data = read_file(path, &len);
    CHECK(data, "cannot read %s", path);
    if (!data) return;

    renza_image a = {0}, b = {0};
    renza_image_patches pa = {0}, pb = {0};
    char ea[256] = {0}, eb[256] = {0};
    const int da = renza_api.decode(&a, data, len, ea, sizeof(ea));
    const int db = ds4->decode(&b, data, len, eb, sizeof(eb));
    CHECK(da && db, "%s: decode renza=%d (%s) ds4=%d (%s)", path, da, ea, db, eb);
    if (da && db) {
        CHECK(a.width == b.width && a.height == b.height &&
              memcmp(a.rgb, b.rgb, (size_t)a.width * a.height * 3) == 0 &&
              memcmp(a.fingerprint, b.fingerprint, sizeof(a.fingerprint)) == 0,
              "%s: decoded image differs from ds4's", path);
        const int qa = renza_api.preprocess(&pa, &a, MIN_TOKENS, MAX_TOKENS, ea, sizeof(ea));
        const int qb = ds4->preprocess(&pb, &b, MIN_TOKENS, MAX_TOKENS, eb, sizeof(eb));
        CHECK(qa && qb, "%s: preprocess renza=%d (%s) ds4=%d (%s)", path, qa, ea, qb, eb);
        if (qa && qb) {
            const size_t floats = (size_t)pa.patch_count * 3 * 16 * 16;
            CHECK(memcmp(&pa, &pb, offsetof(renza_image_patches, patches)) == 0 &&
                  memcmp(pa.patches, pb.patches, floats * sizeof(float)) == 0,
                  "%s: patches differ from ds4's", path);
            CHECK(!expect || (pa.image_token_count == want_tokens && pa.grid_width == want_gw * MERGE &&
                              pa.grid_height == want_gh * MERGE),
                  "%s: %u tokens, merged grid %ux%u; expected %u, %ux%u", path, pa.image_token_count,
                  pa.grid_width / MERGE, pa.grid_height / MERGE, want_tokens, want_gw, want_gh);
            printf("%s: %ux%u -> %u tokens, merged grid %ux%u, identical to ds4\n", path, a.width, a.height,
                   pa.image_token_count, pa.grid_width / MERGE, pa.grid_height / MERGE);
        }
    }
    renza_api.patches_free(&pa);
    ds4->patches_free(&pb);
    renza_api.image_free(&a);
    ds4->image_free(&b);
    free(data);
}

int main(int argc, char **argv) {
    if (argc >= 3 && strcmp(argv[1], "--ds4") == 0) {
        image_api ds4;
        if (!load_ds4(argv[2], &ds4)) return 1;
        for (int i = 3; i < argc; i++) compare_image(&ds4, argv[i]);
    } else if (argc == 3) {
        check_tokens(1024, 768, 768, 32, 24);
        check_tokens(1024, 1024, 1024, 32, 32);
        check_tokens(4000, 3000, 972, 36, 27);
        check_tokens(320, 224, 70, 10, 7);
        check_tokens(8, 8, 64, 8, 8);
        check_aspect_refused();
        check_jpeg_orientation(argv[1]);
        /* Progressive AC scan with Se=200 (issue #1292). Unpatched Iris indexes
         * jpeg_zigzag[] past 63 and writes outside the coefficient block. */
        CHECK(check_jpeg_rejected(argv[2]), "malformed JPEG was accepted");
        /* DC scans must have Se == 0. */
        CHECK(check_jpeg_sos_rejected(0, 5), "invalid DC scan was accepted");
    } else {
        fprintf(stderr, "usage: %s ROTATED.jpg MALFORMED.jpg | --ds4 LIB IMAGE[=TOKENS:GW:GH]...\n", argv[0]);
        return 2;
    }
    if (failures) fprintf(stderr, "%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}
