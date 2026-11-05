/* renza_engine_open's refusals over the real Strata library.  The adapter's
 * own refusals come before CUDA starts; Strata's Engine::open starts the CUDA
 * runtime even when it only prints its version. */

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "renza.h"
#include "renza_strata.h"

static int failures;

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                  \
        }                                                                \
    } while (0)

enum { EXIT_USAGE = 2, EXIT_FAILURE_LOAD = 1 };

/* CUDA's initialisation starts a thread, even when it finds no device
 * (libcuda itself is a link dependency, loaded at start). */
static int thread_count(void) {
    DIR *d = opendir("/proc/self/task");
    if (!d) return -1;
    int n = 0;
    for (struct dirent *ent; (ent = readdir(d)) != NULL;) n += ent->d_name[0] != '.';
    closedir(d);
    return n;
}

/* Runs open, which must fail; returns its open status. */
static int failed_open(const renza_engine_options *opt, char *err, size_t errlen, bool *printed) {
    renza_engine *e = NULL;
    err[0] = '\0';
    CHECK(renza_engine_open(&e, opt, err, errlen) == RENZA_ERROR);
    CHECK(e == NULL);
    return renza_strata_open_status(printed);
}

int main(void) {
    const int threads = thread_count();
    char err[512];
    bool printed = true;
    renza_engine_options opt;
    renza_engine_options_init(&opt);

    CHECK(failed_open(&opt, err, sizeof(err), &printed) == EXIT_USAGE && !printed);
    CHECK(strstr(err, "pack_dir") != NULL);
    opt.pack_dir = "/nonexistent/pack";
    CHECK(failed_open(&opt, err, sizeof(err), &printed) == EXIT_USAGE && !printed);
    CHECK(strstr(err, "native_gguf") != NULL);
    opt.native_gguf = "/nonexistent/model-00001-of-00002.gguf";
    opt.spec = RENZA_OPTION_AUTO;
    CHECK(failed_open(&opt, err, sizeof(err), &printed) == EXIT_USAGE && !printed);
    CHECK(strstr(err, "--spec") != NULL);
    CHECK(thread_count() == threads);

    char path[] = "/tmp/renza-strata-open-XXXXXX";
    const char invalid_model[] = "This file is deliberately not a GGUF model.\n";
    int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd >= 0) {
        CHECK(write(fd, invalid_model, sizeof(invalid_model) - 1) == (ssize_t)(sizeof(invalid_model) - 1));
        CHECK(close(fd) == 0);
        renza_engine_options_init(&opt);
        opt.pack_dir = "/nonexistent/pack";
        opt.native_gguf = path;
        CHECK(failed_open(&opt, err, sizeof(err), &printed) == EXIT_FAILURE_LOAD && !printed);
        CHECK(strstr(err, "not a GGUF") != NULL);
        CHECK(thread_count() == threads);
        CHECK(unlink(path) == 0);
    }

    /* Strata prints its version and opens nothing; that was the process's one open. */
    const char *version[] = {"--version"};
    renza_engine_options_init(&opt);
    opt.extra_args = version;
    opt.n_extra_args = 1;
    CHECK(failed_open(&opt, err, sizeof(err), &printed) == 0 && printed && err[0] == '\0');

    CHECK(failed_open(&opt, err, sizeof(err), &printed) == EXIT_FAILURE_LOAD && !printed);
    CHECK(strstr(err, "one engine per process") != NULL);

    if (failures) {
        fprintf(stderr, "renza-strata-open-test: %d failed\n", failures);
        return 1;
    }
    printf("renza-strata-open-test: ok\n");
    return 0;
}
