#ifndef RENZA_STRATA_H
#define RENZA_STRATA_H

/* What `renza-engine --serve` (engine/serve/) needs from the Strata adapter
 * beyond renza.h to speak Strata's stdin/stdout protocol: open's exit status,
 * the INFO and READY values, the engine's end-of-generation set, an explicit
 * end of request with the DONE counters, and close's status.  Not for the
 * server. */

#include "renza.h"

#ifdef __cplusplus
extern "C" {
#endif

/* After a failed renza_engine_open: the exit status Strata's serve program
 * had for it (0 when it only printed --version or --help, 1 a load failure,
 * 2 bad options), and in *printed whether the message is already on stderr
 * (Strata writes its own; the adapter's refusals are only in err).  Options
 * whose extra_args hold --version, -V, --help or -h are passed to Strata
 * without the adapter's check of the required fields, so it can print them. */
int renza_strata_open_status(bool *printed);

/* The fields of the protocol's INFO line, measured at the end of open. */
const char *renza_strata_engine_info(const renza_engine *e);
int renza_strata_engine_max_context(const renza_engine *e);
/* token is in the set that ends a decode run (--eos-ids). */
bool renza_strata_token_is_eos(const renza_engine *e, int token);

/* The protocol's DONE counters of one request. */
typedef struct {
    int64_t generated;
    int64_t prompt_tokens;
    double prompt_ms;
    double decode_ms;
    int64_t draft_accepted;
    int64_t draft_offered;
    int64_t reused;
    int64_t expert_hits;
    int64_t expert_lookups;
    int64_t ram_reads;
    int64_t file_reads;
    double file_read_mb;
    int64_t prompt_read;
    int64_t offloaded;
    bool chain_counts;  /* --lookup-chain: the five counts below are reported */
    int64_t chain_accepted;
    int64_t chain_offered;
    int64_t suffix_accepted;
    int64_t suffix_offered;
    int64_t windows;
    int64_t chunk;
    int64_t experts_streamed;
} renza_strata_request_stats;

/* Ends the request the last sync opened (a sync that returned RENZA_OK,
 * RENZA_INTERRUPTED, or RENZA_ERROR after reading began): Strata's per-request
 * stderr summary, *stats, and the live state the next sync may resume from.
 * Without it the next sync, rewind, invalidate or free ends the request.
 * Afterwards checkpoint_valid is false.  RENZA_OK; RENZA_ERROR when no request
 * is open (nothing changes; a refused sync opens none); RENZA_ENGINE_DEAD. */
int renza_strata_session_finish(renza_session *s, renza_strata_request_stats *stats,
                                char *err, size_t errlen);

/* renza_engine_close with Strata's status: RENZA_ENGINE_DEAD (err says why)
 * when the engine was dead or died closing (ending the open request, landing
 * the adaptive tier's last round). */
int renza_strata_engine_close(renza_engine *e, char *err, size_t errlen);

#ifdef __cplusplus
}
#endif

#endif
