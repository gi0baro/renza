// engine/strata_adapter.cpp - renza.h over Strata's engine library (strata/program/engine.hpp).
//
// renza_engine_open turns renza_engine_options into Strata's argv and runs Engine::open; a session is Strata's one
// Session plus renza's bookkeeping (engine/renza_common.c): the history renza.h promises, the image identities, the
// callbacks.  Strata keeps its own state (live, checkpoints) and decides what a sync resumes from.
// docs/work/renza-h-mapping.md is the function-by-function map.

#include "renza.h"
#include "renza_common.h"
#include "renza_strata.h"
#include "renza_vision.h"
#include "strata/program/engine.hpp"

#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace program = strata::program;

static_assert(std::is_same_v<int, int32_t>, "renza.h's int token ids are Strata's int32_t ids");
static_assert(RENZA_OK == program::kOk && RENZA_INTERRUPTED == program::kInterrupted &&
              RENZA_ENGINE_DEAD == program::kDead && RENZA_TOKEN_NONE == program::kNoToken);

namespace {

const char kProgramName[] = "renza";
constexpr uint32_t kVisionMaxTokens = 1024;  // the user's Strata vision cap
constexpr int kPromptCacheDefault = 6;   // Strata's Options::prompt_cache
constexpr int kVisionStart = 248053;     // <|vision_start|>
constexpr int kVisionEnd = 248054;       // <|vision_end|>
const char kNoEncoder[] = "no vision encoder linked (the engine takes image rows; the encoder is unit 6)";
const char kNoEngine[] = "a test checkpoint has no engine";

// renza's part of a saved session, after Strata's session file: the history's last token when Strata had not read it
// (a save right after a sync: Strata reads a prompt's last token in the first decode window)
struct SavedTail {
    char magic[8];
    int32_t unread;   // 0 or 1
    int32_t token;
};
constexpr char kSavedTailMagic[8] = {'R', 'E', 'N', 'Z', 'A', 'K', 'V', '1'};
static_assert(sizeof(SavedTail) == 16);
// Strata reads a session file by name: a load copies the saved bytes here first
const char kLoadTemplate[] = "/tmp/renza-session-load.XXXXXX";
constexpr size_t kCopyBlock = 1 << 20;

struct OpenOutcome {
    int status = program::kExitOk;
    bool printed = false;
};
OpenOutcome g_open;
// Engine::open was called once: Strata refuses every later call, without printing (one engine per process)
bool g_open_called = false;

void set_err(char* err, size_t errlen, const std::string& message) { renza_set_err(err, errlen, message.c_str()); }

bool informational(const renza_engine_options& o) {
    for (int i = 0; i < o.n_extra_args; ++i) {
        const char* a = o.extra_args[i];
        if (!std::strcmp(a, "--version") || !std::strcmp(a, "-V") || !std::strcmp(a, "--help") || !std::strcmp(a, "-h"))
            return true;
    }
    return false;
}

// Strata's argv: the program, --serve, the set fields in renza.h's order, extra_args.  Every field flag is a plain
// assignment in Strata's parser, so their order among themselves does not matter; extra_args come last and win.
bool strata_args(const renza_engine_options& o, std::vector<std::string>& args, std::string& bad) {
    args = {kProgramName, "--serve"};
    auto text = [&](const char* flag, const char* v) {
        if (v == nullptr) return;
        args.emplace_back(flag);
        args.emplace_back(v);
    };
    auto count = [&](const char* flag, int v, bool auto_allowed) {
        if (v == RENZA_OPTION_DEFAULT) return true;
        if (v == RENZA_OPTION_AUTO && !auto_allowed) {
            bad = std::string(flag) + " has no auto value";
            return false;
        }
        args.emplace_back(flag);
        args.emplace_back(v == RENZA_OPTION_AUTO ? "auto" : std::to_string(v));
        return true;
    };
    auto real = [&](const char* flag, double v) {
        if (v == RENZA_OPTION_DEFAULT) return true;
        if (v == RENZA_OPTION_AUTO) {
            bad = std::string(flag) + " has no auto value";
            return false;
        }
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.17g", v);   // round-trips the double through Strata's atof
        args.emplace_back(flag);
        args.emplace_back(buf);
        return true;
    };
    text("--pack", o.pack_dir);
    text("--native", o.native_gguf);
    text("--ple-gguf", o.ple_gguf);
    text("--expert-profile", o.expert_profile);
    if (!count("--expert-cache", o.expert_cache, true) || !count("--prefill", o.prefill_chunk, true) ||
        !count("--spec", o.spec, false))
        return false;
    text("--mtp", o.mtp_dir);
    if (!real("--spec-min-p", o.spec_min_p) || !count("--max-context", o.max_context, false)) return false;
    text("--kv", o.kv);
    if (!count("--kv-resident", o.kv_resident, false)) return false;
    if (o.mmproj != nullptr) args.emplace_back("--vision");
    if (!count("--vram-reserve-mib", o.vram_reserve_mib, false) || !real("--pcie-frac", o.pcie_frac) ||
        !count("--pool-workers", o.pool_workers, false) || !count("--prompt-cache", o.prompt_cache, false))
        return false;
    for (int i = 0; i < o.n_extra_args; ++i) args.emplace_back(o.extra_args[i]);
    return true;
}

// --prompt-cache as Strata's parser resolves it (the last one wins); the engine has no query for it
int resolved_prompt_cache(const std::vector<std::string>& args) {
    int n = kPromptCacheDefault;
    for (size_t i = 2; i + 1 < args.size(); ++i)
        if (args[i] == "--prompt-cache") n = std::max(0, std::atoi(args[++i].c_str()));
    return n;
}

int refuse_open(int status, const std::string& message, char* err, size_t errlen) {
    g_open = {status, false};
    set_err(err, errlen, message);
    return RENZA_ERROR;
}

bool append_tail(const char* path, const SavedTail& tail, std::string& why) {
    std::FILE* f = std::fopen(path, "ab");
    if (f == nullptr) {
        why = std::strerror(errno);
        return false;
    }
    bool ok = std::fwrite(&tail, sizeof tail, 1, f) == 1 && std::fflush(f) == 0 && fsync(fileno(f)) == 0;
    if (!ok) why = std::strerror(errno);
    if (std::fclose(f) != 0 && ok) {
        why = std::strerror(errno);
        ok = false;
    }
    return ok;
}

bool copy_to_fd(std::FILE* in, int fd, uint64_t n, std::string& why) {
    std::vector<char> buf(kCopyBlock);
    while (n > 0) {
        const size_t want = (size_t) std::min<uint64_t>(n, buf.size());
        if (std::fread(buf.data(), 1, want, in) != want) {
            why = std::ferror(in) ? std::strerror(errno) : "the saved session is truncated";
            return false;
        }
        for (size_t at = 0; at < want;) {
            const ssize_t w = write(fd, buf.data() + at, want - at);
            if (w < 0 && errno == EINTR) continue;
            if (w <= 0) {
                why = std::strerror(errno);
                return false;
            }
            at += (size_t) w;
        }
        n -= want;
    }
    return true;
}

}  // namespace

struct renza_engine {
    std::unique_ptr<program::Engine> engine;
    renza_engine_text* text = nullptr;
    renza_vision_encoder* encoder = nullptr;
    int prompt_cache = kPromptCacheDefault;
};

namespace {

struct StrataSession {
    renza_engine* engine = nullptr;
    std::unique_ptr<program::Session> session;
    bool request_open = false;   // a sync opened a request that nothing has finished yet
    bool pending = false;        // a successful sync was the last call: its last prompt token is still unread
    std::vector<float> pending_rows;   // that sync's image rows, in image order, for a decode that feeds a token
};

StrataSession* backend(const renza_session* s) { return static_cast<StrataSession*>(s->backend); }

void clear_pending(StrataSession& b) {
    b.pending = false;
    b.pending_rows.clear();
    b.pending_rows.shrink_to_fit();
}

void end_request(StrataSession& b) {
    if (b.request_open) b.session->finish();
    b.request_open = false;
}

std::vector<program::ImageSpan> strata_spans(const renza_vision_span* images, size_t image_count) {
    std::vector<program::ImageSpan> spans(image_count);
    for (size_t i = 0; i < image_count; ++i) {
        const renza_vision_embedding& em = images[i].embedding;
        spans[i] = {images[i].token_start, em.token_count, em.grid_width, em.grid_height, em.rows, em.key};
    }
    return spans;
}

program::Sampling strata_sampling(const renza_sampling& r) {
    program::Sampling s;
    s.temperature = r.temperature;
    s.top_k = r.top_k;
    s.top_p = r.top_p <= 0.0f || r.top_p >= 1.0f ? 1.0f : r.top_p;
    s.min_p = r.min_p;
    s.penalty_last_n = r.penalty_last_n;
    s.penalty_repeat = r.penalty_repeat == 0.0f ? 1.0f : r.penalty_repeat;
    s.penalty_freq = r.penalty_freq;
    s.penalty_present = r.penalty_present;
    s.seed = r.seed;
    return s;
}

// The bookkeeping after one of Strata's sync results other than kRefused (which changes nothing).
int settle_sync(renza_session* s, StrataSession& b, int status, const int* ids, int n,
                const renza_vision_span* images, size_t image_count, char* err, size_t errlen) {
    clear_pending(b);
    b.request_open = status != program::kDead;
    if (status != program::kOk) set_err(err, errlen, b.session->error());
    switch (status) {
    case program::kOk:
        renza_session_book_assign(s, ids, n, images, image_count);
        s->checkpoint_valid = true;
        b.pending = true;
        return RENZA_OK;
    case program::kInterrupted: {
        const int held = (int) std::max<int64_t>(b.session->held(), 0);
        renza_session_book_assign(s, ids, held, images, image_count);
        // Strata keeps the held prefix as its live state when it ends the request, and only with the prompt cache on
        s->checkpoint_valid = held > 0 && b.engine->prompt_cache > 0;
        return RENZA_INTERRUPTED;
    }
    case program::kDead:
        s->checkpoint_valid = false;
        return RENZA_ENGINE_DEAD;
    default:   // kInvalidated: non-finite logits; the next sync reads from 0
        renza_session_book_invalidate(s);
        return RENZA_ERROR;
    }
}

}  // namespace

const renza_engine_text* renza_engine_text_of(const renza_engine* e) { return e->text; }

int renza_engine_open(renza_engine** out, const renza_engine_options* opt, char* err, size_t errlen) {
    if (out != nullptr) *out = nullptr;
    g_open = {};
    if (out == nullptr || opt == nullptr)
        return refuse_open(program::kExitUsage, "missing options or engine output", err, errlen);
    std::vector<std::string> args;
    std::string bad;
    if (!strata_args(*opt, args, bad)) return refuse_open(program::kExitUsage, bad, err, errlen);
    if (!informational(*opt)) {
        if (opt->pack_dir == nullptr)
            return refuse_open(program::kExitUsage, "pack_dir (--pack) is required", err, errlen);
        if (opt->native_gguf == nullptr)
            return refuse_open(program::kExitUsage, "native_gguf (--native) is required", err, errlen);
    }
    renza_engine_text* text = nullptr;
    if (opt->native_gguf != nullptr) {
        text = renza_engine_text_open(opt->native_gguf, err, errlen);
        if (text == nullptr) {
            g_open = {program::kExitFailure, false};
            return RENZA_ERROR;
        }
    }
    renza_vision_encoder* encoder = nullptr;
    if (opt->mmproj != nullptr) {
        if (text == nullptr)
            return refuse_open(program::kExitUsage, "native_gguf (--native) is required", err, errlen);
        // Reserve encoder VRAM before Strata sizes its expert cache.
        encoder = renza_vision_encoder_open(opt->mmproj, kVisionMaxTokens, err, errlen);
        if (encoder == nullptr) {
            renza_engine_text_close(text);
            g_open = {program::kExitFailure, false};
            return RENZA_ERROR;
        }
        const uint32_t width = renza_vision_encoder_embd_dim(encoder);
        const int model_width = renza_engine_text_arch_u32(text, "embedding_length");
        if (model_width <= 0 || width != (uint32_t) model_width) {
            renza_vision_encoder_close(encoder);
            renza_engine_text_close(text);
            return refuse_open(program::kExitFailure,
                               "vision embedding width " + std::to_string(width) +
                                   " does not match native GGUF embedding width " + std::to_string(model_width),
                               err, errlen);
        }
    }
    std::vector<char*> argv;
    for (std::string& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    const bool reopen = g_open_called;
    g_open_called = true;
    program::OpenError error;
    std::unique_ptr<program::Engine> engine = program::Engine::open((int) args.size(), argv.data(), error);
    if (engine == nullptr) {
        renza_vision_encoder_close(encoder);
        renza_engine_text_close(text);
        g_open = {error.status, !reopen};
        set_err(err, errlen, error.message);
        return RENZA_ERROR;
    }
    if (text == nullptr) {
        engine->close();
        set_err(err, errlen, "native_gguf (--native) is required");
        g_open = {program::kExitFailure, false};
        return RENZA_ERROR;
    }
    renza_engine* e = new renza_engine;
    e->engine = std::move(engine);
    e->text = text;
    e->encoder = encoder;
    e->prompt_cache = resolved_prompt_cache(args);
    *out = e;
    return RENZA_OK;
}

void renza_engine_close(renza_engine* e) { renza_strata_engine_close(e, nullptr, 0); }

int renza_engine_vocab_size(const renza_engine* e) { return (int) e->engine->vocab_size(); }
int renza_engine_embd_dim(const renza_engine* e) { return (int) e->engine->embd_dim(); }
int renza_engine_layer_count(const renza_engine* e) { return (int) e->engine->layer_count(); }
int renza_engine_prefill_chunk(const renza_engine* e) { return (int) e->engine->prefill_chunk(); }
int renza_engine_spec(const renza_engine* e) { return (int) e->engine->spec(); }
bool renza_engine_has_vision(const renza_engine* e) { return e->engine->has_vision(); }

int renza_engine_vision_encode_memory(renza_engine* e, const uint8_t* encoded, size_t encoded_len,
                                      renza_vision_embedding* out, char* err, size_t errlen) {
    std::memset(out, 0, sizeof(*out));
    if (e->engine->dead()) {
        set_err(err, errlen, e->engine->dead_message());
        return RENZA_ENGINE_DEAD;
    }
    if (e->encoder == nullptr) {
        set_err(err, errlen, kNoEncoder);
        return RENZA_ERROR;
    }
    renza_vision_rows rows = {};
    if (!renza_vision_encoder_encode(e->encoder, encoded, encoded_len, &rows, err, errlen)) return RENZA_ERROR;
    out->rows = rows.rows;
    out->token_count = rows.token_count;
    out->grid_width = rows.grid_width;
    out->grid_height = rows.grid_height;
    static_assert(sizeof(out->fingerprint) == sizeof(rows.fingerprint));
    std::memcpy(out->fingerprint, rows.fingerprint, sizeof(out->fingerprint));
    // Match ds4_shim.c's fingerprint_key: Strata's conversation cache matches ImageSpan::key.
    for (int i = 0; i < 8; ++i) out->key |= (uint64_t) out->fingerprint[i] << (8 * i);
    return RENZA_OK;
}

int renza_prompt_append_vision(const renza_engine* e, renza_tokens* prompt, renza_vision_span* span,
                               renza_vision_embedding* embedding, char* err, size_t errlen) {
    if (!e->engine->has_vision()) {
        set_err(err, errlen, "this engine was opened without vision");
        return RENZA_ERROR;
    }
    if (embedding->rows == nullptr || embedding->token_count == 0 ||
        (uint64_t) embedding->grid_width * embedding->grid_height != embedding->token_count) {
        set_err(err, errlen, "the embedding's rows are not token_count = grid_width x grid_height");
        return RENZA_ERROR;
    }
    renza_tokens_push(prompt, kVisionStart);
    span->token_start = (uint32_t) prompt->len;
    for (uint32_t i = 0; i < embedding->token_count; ++i) renza_tokens_push(prompt, (int) program::kImagePad);
    renza_tokens_push(prompt, kVisionEnd);
    span->embedding = *embedding;
    std::memset(embedding, 0, sizeof(*embedding));
    return RENZA_OK;
}

int renza_session_create(renza_session** out, renza_engine* e, int ctx_size, char* err, size_t errlen) {
    *out = nullptr;
    if (e->engine->dead()) {
        set_err(err, errlen, e->engine->dead_message());
        return RENZA_ENGINE_DEAD;
    }
    std::string error;
    std::unique_ptr<program::Session> session = e->engine->create_session(ctx_size, error);
    if (session == nullptr) {
        set_err(err, errlen, error);
        return RENZA_ERROR;
    }
    renza_session* s = renza_session_book_new();
    s->ctx = ctx_size;
    StrataSession* b = new StrataSession;
    b->engine = e;
    b->session = std::move(session);
    s->backend = b;
    *out = s;
    return RENZA_OK;
}

void renza_session_free(renza_session* s) {
    if (s == nullptr) return;
    delete backend(s);   // ~Session ends an open request and drops Strata's conversation state
    renza_session_book_free(s);
}

int renza_session_sync(renza_session* s, const renza_tokens* prompt, char* err, size_t errlen) {
    return renza_session_sync_multimodal(s, prompt, nullptr, 0, err, errlen);
}

int renza_session_sync_multimodal(renza_session* s, const renza_tokens* prompt, const renza_vision_span* images,
                                  size_t image_count, char* err, size_t errlen) {
    StrataSession* b = backend(s);
    if (b == nullptr) {
        set_err(err, errlen, kNoEngine);
        return RENZA_ERROR;
    }
    if (b->engine->engine->dead()) {
        set_err(err, errlen, b->engine->engine->dead_message());
        return RENZA_ENGINE_DEAD;
    }
    const int* ids = prompt->v;
    program::ProgressFn progress;
    if (s->progress != nullptr)
        progress = [s, ids, images, image_count](const program::SyncProgress& pr) {
            // the queries a progress callback may call reflect the state at done
            renza_session_book_assign(s, ids, (int) pr.done, images, image_count);
            s->checkpoint_valid = false;
            s->reused = (int) pr.from;
            s->progress(s->progress_ud, (int) pr.done, (int) pr.total);
        };
    program::CancelFn cancel;
    if (s->cancel != nullptr) cancel = [s] { return s->cancel(s->cancel_ud); };
    const std::vector<program::ImageSpan> spans = strata_spans(images, image_count);
    s->syncing_images = image_count > 0;
    const int status = b->session->sync(ids, prompt->len, spans.data(), spans.size(), progress, cancel);
    s->syncing_images = false;
    if (status == program::kRefused) {
        set_err(err, errlen, b->session->error());
        return RENZA_ERROR;
    }
    s->reused = (int) b->session->reused();
    const int result = settle_sync(s, *b, status, ids, prompt->len, images, image_count, err, errlen);
    if (result == RENZA_OK && image_count > 0) {
        const size_t width = (size_t) b->engine->engine->embd_dim();
        for (size_t i = 0; i < image_count; ++i) {
            const renza_vision_embedding& em = images[i].embedding;
            b->pending_rows.insert(b->pending_rows.end(), em.rows, em.rows + (size_t) em.token_count * width);
        }
    }
    return result;
}

int renza_session_decode(renza_session* s, int token, const renza_sampling* sampling, int budget, int* out, int cap,
                         char* err, size_t errlen) {
    StrataSession* b = backend(s);
    if (b == nullptr) {
        set_err(err, errlen, kNoEngine);
        return RENZA_ERROR;
    }
    if (b->engine->engine->dead()) {
        set_err(err, errlen, b->engine->engine->dead_message());
        return RENZA_ENGINE_DEAD;
    }
    if (!s->checkpoint_valid) {
        set_err(err, errlen, "decode needs a valid checkpoint: a successful sync first");
        return RENZA_ERROR;
    }
    if (token == RENZA_TOKEN_NONE && !b->pending) {
        set_err(err, errlen, "RENZA_TOKEN_NONE is only for the first decode after a sync");
        return RENZA_ERROR;
    }
    int32_t fed = token;
    // Strata reads a synced prompt's last token in the first decode window, which feeds nothing, and a save or a load
    // ends Strata's request: the token (if any) joins the prompt instead, and a sync from the state Strata kept opens
    // the request whose first window reads it
    const bool resync = token >= 0 ? b->pending || !b->request_open : !b->request_open;
    if (resync) {
        const int fed_tokens = token >= 0 ? 1 : 0;
        if (budget < 1 || cap < 1 || s->tokens.len + fed_tokens + 1 > s->ctx) {
            set_err(err, errlen, "the context has no room for the token and the one after it, or no budget");
            return RENZA_ERROR;
        }
        std::vector<int> ids(s->tokens.v, s->tokens.v + s->tokens.len);
        if (token >= 0) ids.push_back(token);
        std::vector<renza_vision_span> images(s->images, s->images + s->image_count);
        const size_t width = (size_t) b->engine->engine->embd_dim();
        size_t offset = 0;
        for (renza_vision_span& im : images) {
            im.embedding.rows = b->pending_rows.data() + offset;
            offset += (size_t) im.embedding.token_count * width;
        }
        const std::vector<program::ImageSpan> spans = strata_spans(images.data(), images.size());
        std::vector<float> rows = std::move(b->pending_rows);
        const int status = b->session->sync(ids.data(), (int64_t) ids.size(), spans.data(), spans.size(), {}, {});
        if (status == program::kRefused) {
            b->pending_rows = std::move(rows);
            set_err(err, errlen, b->session->error());
            return RENZA_ERROR;
        }
        const int result = settle_sync(s, *b, status, ids.data(), (int) ids.size(), images.data(), images.size(),
                                       err, errlen);
        if (result != RENZA_OK) return result == RENZA_INTERRUPTED ? RENZA_ERROR : result;
        b->pending_rows = std::move(rows);
        fed = program::kNoToken;
    }
    const int n = b->session->decode(fed, strata_sampling(*sampling), budget, out, cap);
    if (n >= 1) {
        if (fed >= 0) renza_tokens_push(&s->tokens, fed);
        for (int i = 0; i + 1 < n; ++i) renza_tokens_push(&s->tokens, out[i]);
        clear_pending(*b);
        return n;
    }
    set_err(err, errlen, b->session->error());
    if (n == program::kDead) {
        s->checkpoint_valid = false;
        clear_pending(*b);
        return RENZA_ENGINE_DEAD;
    }
    if (n == program::kInvalidated) {
        renza_session_book_invalidate(s);
        clear_pending(*b);
    }
    return RENZA_ERROR;
}

void renza_session_rewind(renza_session* s, int pos) {
    if (!renza_session_book_rewind(s, pos)) return;
    StrataSession* b = backend(s);
    if (b == nullptr) return;
    end_request(*b);
    clear_pending(*b);
}

void renza_session_invalidate(renza_session* s) {
    if (s == nullptr) return;
    renza_session_book_invalidate(s);
    StrataSession* b = backend(s);
    if (b == nullptr) return;
    b->session->invalidate();
    b->request_open = false;
    clear_pending(*b);
}

int renza_session_save(renza_session* s, const char* path, char* err, size_t errlen) {
    StrataSession* b = backend(s);
    if (b == nullptr) {
        set_err(err, errlen, kNoEngine);
        return RENZA_ERROR;
    }
    if (b->engine->engine->dead()) {
        set_err(err, errlen, b->engine->engine->dead_message());
        return RENZA_ENGINE_DEAD;
    }
    if (!s->checkpoint_valid || b->engine->prompt_cache <= 0) {
        set_err(err, errlen, "the session is not at a state it can save");
        return RENZA_UNAVAILABLE;
    }
    if (s->image_count > 0) {
        set_err(err, errlen, "a session holding images is not saved (the file keeps no image rows)");
        return RENZA_ERROR;
    }
    end_request(*b);
    std::vector<int32_t> saved;
    const int status = b->session->save(path, saved);
    if (status != program::kOk) set_err(err, errlen, b->session->error());
    if (status == program::kDead) {
        s->checkpoint_valid = false;
        clear_pending(*b);
        return RENZA_ENGINE_DEAD;
    }
    if (status == program::kUnavailable) return RENZA_UNAVAILABLE;
    if (status != program::kOk) return RENZA_ERROR;
    const int unread = b->pending ? 1 : 0;
    std::string why;
    if ((int) saved.size() + unread != s->tokens.len || !std::equal(saved.begin(), saved.end(), s->tokens.v))
        why = "the engine saved another history than the session's";
    SavedTail tail = {};
    std::memcpy(tail.magic, kSavedTailMagic, sizeof tail.magic);
    tail.unread = unread;
    tail.token = unread ? s->tokens.v[s->tokens.len - 1] : 0;
    if (why.empty() && append_tail(path, tail, why)) return RENZA_OK;
    std::remove(path);
    set_err(err, errlen, why);
    return RENZA_ERROR;
}

int renza_session_load(renza_session* s, std::FILE* fp, uint64_t bytes, char* err, size_t errlen) {
    StrataSession* b = backend(s);
    if (b == nullptr) {
        set_err(err, errlen, kNoEngine);
        return RENZA_ERROR;
    }
    if (b->engine->engine->dead()) {
        set_err(err, errlen, b->engine->engine->dead_message());
        return RENZA_ENGINE_DEAD;
    }
    if (bytes < sizeof(SavedTail)) {
        set_err(err, errlen, "not a saved session (too short)");
        return RENZA_ERROR;
    }
    char path[sizeof kLoadTemplate];
    std::memcpy(path, kLoadTemplate, sizeof path);
    const int fd = mkstemp(path);
    if (fd < 0) {
        set_err(err, errlen, std::string("cannot create a file for the saved session: ") + std::strerror(errno));
        return RENZA_ERROR;
    }
    std::string why;
    bool ok = copy_to_fd(fp, fd, bytes - sizeof(SavedTail), why);
    if (close(fd) != 0 && ok) {
        why = std::strerror(errno);
        ok = false;
    }
    SavedTail tail = {};
    if (ok && std::fread(&tail, sizeof tail, 1, fp) != 1) {
        why = "the saved session is truncated";
        ok = false;
    }
    if (ok && (std::memcmp(tail.magic, kSavedTailMagic, sizeof tail.magic) != 0 || tail.unread < 0 ||
               tail.unread > 1 || tail.token < 0 || tail.token >= b->engine->engine->vocab_size())) {
        why = "not a saved session (renza's part of it is invalid)";
        ok = false;
    }
    if (!ok) {
        unlink(path);
        set_err(err, errlen, why);
        return RENZA_ERROR;
    }
    end_request(*b);
    std::vector<int32_t> ids;
    const int status = b->session->load(path, ids);
    unlink(path);
    if (status != program::kOk) set_err(err, errlen, b->session->error());
    switch (status) {
    case program::kOk:
        break;
    case program::kDead:
        s->checkpoint_valid = false;
        clear_pending(*b);
        return RENZA_ENGINE_DEAD;
    case program::kInvalidated:
        renza_session_book_invalidate(s);
        clear_pending(*b);
        return RENZA_ERROR;
    default:   // refused: Strata's state is as it was
        return RENZA_ERROR;
    }
    if ((int64_t) ids.size() + tail.unread >= s->ctx) {
        renza_session_invalidate(s);
        set_err(err, errlen, "the saved history does not fit this session");
        return RENZA_ERROR;
    }
    if (tail.unread) ids.push_back(tail.token);
    clear_pending(*b);
    renza_session_book_assign(s, ids.data(), (int) ids.size(), nullptr, 0);
    s->checkpoint_valid = true;
    b->pending = tail.unread == 1;
    return RENZA_OK;
}

int renza_strata_open_status(bool* printed) {
    *printed = g_open.printed;
    return g_open.status;
}

const char* renza_strata_engine_info(const renza_engine* e) { return e->engine->info().c_str(); }

int renza_strata_engine_max_context(const renza_engine* e) { return (int) e->engine->max_context(); }

bool renza_strata_token_is_eos(const renza_engine* e, int token) {
    const std::vector<int64_t>& eos = e->engine->eos_ids();
    return std::find(eos.begin(), eos.end(), (int64_t) token) != eos.end();
}

int renza_strata_session_finish(renza_session* s, renza_strata_request_stats* stats, char* err, size_t errlen) {
    StrataSession* b = backend(s);
    if (b == nullptr) {
        set_err(err, errlen, kNoEngine);
        return RENZA_ERROR;
    }
    if (b->engine->engine->dead()) {
        set_err(err, errlen, b->engine->engine->dead_message());
        return RENZA_ENGINE_DEAD;
    }
    if (!b->request_open) {
        set_err(err, errlen, "no request is open");
        return RENZA_ERROR;
    }
    b->request_open = false;
    clear_pending(*b);
    s->checkpoint_valid = false;
    if (b->session->finish() != program::kOk) {
        set_err(err, errlen, b->session->error());
        return RENZA_ENGINE_DEAD;
    }
    const program::RequestStats& st = b->session->stats();
    *stats = {st.generated,      st.prompt_tokens,   st.prompt_ms,       st.decode_ms,      st.draft_accepted,
              st.draft_offered,  st.reused,          st.expert_hits,     st.expert_lookups, st.ram_reads,
              st.file_reads,     st.file_read_mb,    st.prompt_read,     st.offloaded,      st.chain_counts,
              st.chain_accepted, st.chain_offered,   st.suffix_accepted, st.suffix_offered, st.windows,
              st.chunk,          st.experts_streamed};
    return RENZA_OK;
}

int renza_strata_engine_close(renza_engine* e, char* err, size_t errlen) {
    if (e == nullptr) return RENZA_OK;
    const int status = e->engine->close();
    if (status != program::kOk) set_err(err, errlen, e->engine->dead_message());
    // Close Strata's CUDA state before releasing the encoder's GPU resources.
    e->engine.reset();
    renza_vision_encoder_close(e->encoder);
    renza_engine_text_close(e->text);
    delete e;
    return status == program::kOk ? RENZA_OK : RENZA_ENGINE_DEAD;
}
