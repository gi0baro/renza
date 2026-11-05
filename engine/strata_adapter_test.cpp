// The adapter's renza.h bookkeeping over a fake of strata/program/engine.hpp (linked instead of Strata's library):
// the argv it builds, the history and status rules of sync / decode / rewind, the explicit end of a request.

#include "renza.h"
#include "renza_common.h"
#include "renza_strata.h"
#include "renza_vision.h"
#include "strata/program/engine.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace program = strata::program;

namespace {

int failures = 0;

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                                   \
        }                                                                                 \
    } while (0)

constexpr int kEmbd = 2;
constexpr int kEos = 99;
constexpr int kSkipped = 77;
const char* gguf = nullptr;   // the native GGUF: the adapter reads the tokenizer from it
std::vector<std::string> calls;
uint32_t vision_width = 0;
int engine_width = kEmbd;
bool vision_open_fails = false;
bool engine_open_fails = false;
bool encode_fails = false;
const uint8_t encoded_image[] = {1, 2, 3};
float* encoded_rows = nullptr;

// What the fake engine was asked and what it answers next.
struct Fake {
    std::vector<std::string> argv;
    bool dead = false;
    std::string message = "fake failure";
    std::string info = "fake=1";
    std::vector<int64_t> eos = {kEos};
    int sync_status = program::kOk;
    int64_t held = -1;
    int64_t reused = 0;
    std::vector<std::vector<int32_t>> syncs;
    std::vector<std::vector<float>> sync_rows;
    std::vector<program::ImageSpan> sync_images;
    std::vector<int32_t> decode_out = {7, 8, 9};
    int decode_status = program::kOk;
    std::vector<int32_t> decode_fed;
    program::Sampling sampling;
    int finishes = 0;
    int invalidates = 0;
    bool request_open = false;
    program::RequestStats stats;
};

Fake* fake = nullptr;

}  // namespace

extern "C" {

renza_engine_text* __real_renza_engine_text_open(const char*, char*, size_t);
void __real_renza_engine_text_close(renza_engine_text*);

renza_engine_text* __wrap_renza_engine_text_open(const char* path, char* err, size_t errlen) {
    calls.emplace_back("text open");
    return __real_renza_engine_text_open(path, err, errlen);
}

void __wrap_renza_engine_text_close(renza_engine_text* text) {
    if (text != nullptr) calls.emplace_back("text close");
    __real_renza_engine_text_close(text);
}

}

struct renza_vision_encoder {};

renza_vision_encoder* renza_vision_encoder_open(const char* path, uint32_t max_tokens, char* err, size_t errlen) {
    calls.emplace_back("vision open");
    CHECK(std::strcmp(path, "/mmproj.gguf") == 0 && max_tokens == 1024);
    if (vision_open_fails) {
        renza_set_err(err, errlen, "fake vision open failure");
        return nullptr;
    }
    return new renza_vision_encoder;
}

uint32_t renza_vision_encoder_embd_dim(const renza_vision_encoder*) { return vision_width; }

bool renza_vision_encoder_encode(renza_vision_encoder*, const uint8_t* encoded, size_t len,
                                  renza_vision_rows* out, char* err, size_t errlen) {
    calls.emplace_back("vision encode");
    CHECK(len == sizeof(encoded_image) && std::memcmp(encoded, encoded_image, len) == 0);
    *out = {};
    if (encode_fails) {
        renza_set_err(err, errlen, "fake encode failure");
        return false;
    }
    out->token_count = 2;
    out->embd_dim = vision_width;
    out->grid_width = 2;
    out->grid_height = 1;
    out->rows = static_cast<float*>(std::malloc(out->token_count * vision_width * sizeof(float)));
    for (size_t i = 0; i < out->token_count * vision_width; ++i) out->rows[i] = (float) i + 0.5f;
    for (size_t i = 0; i < sizeof(out->fingerprint); ++i) out->fingerprint[i] = (uint8_t) (0x80 + i);
    encoded_rows = out->rows;
    return true;
}

void renza_vision_encoder_close(renza_vision_encoder* encoder) {
    if (encoder == nullptr) return;
    calls.emplace_back("vision close");
    delete encoder;
}

struct program::Engine::State : Fake {};

namespace strata::program {

Engine::Engine() : state_(std::make_unique<State>()) {}
Engine::~Engine() { close(); }

std::unique_ptr<Engine> Engine::open(int argc, char** argv, OpenError& error) {
    calls.emplace_back("engine open");
    if (engine_open_fails) {
        error.status = kExitFailure;
        error.message = "fake engine open failure";
        return nullptr;
    }
    std::unique_ptr<Engine> e(new Engine());
    e->state_->argv.assign(argv, argv + argc);
    fake = e->state_.get();
    return e;
}

std::unique_ptr<Session> Engine::create_session(int64_t ctx, std::string&) {
    std::unique_ptr<Session> s(new Session(*this, ctx));
    session_ = s.get();
    return s;
}

int Engine::close() {
    if (state_ == nullptr) return kOk;
    calls.emplace_back("engine close");
    if (session_ != nullptr) session_->engine_ = nullptr;
    session_ = nullptr;
    if (fake == static_cast<Fake*>(state_.get())) fake = nullptr;
    state_.reset();
    return kOk;
}

bool Engine::dead() const { return state_->dead; }
const std::string& Engine::dead_message() const { return state_->message; }
const std::string& Engine::info() const { return state_->info; }
int64_t Engine::spec() const { return 4; }
int64_t Engine::prefill_chunk() const { return 8192; }
int64_t Engine::max_context() const { return 64; }
int64_t Engine::vocab_size() const { return 100; }
int64_t Engine::embd_dim() const { return engine_width; }
int64_t Engine::layer_count() const { return 48; }
bool Engine::has_vision() const { return true; }
const std::vector<int64_t>& Engine::eos_ids() const { return state_->eos; }

Session::Session(Engine& engine, int64_t ctx) : engine_(&engine), ctx_(ctx) {}
Session::~Session() {
    if (engine_ != nullptr) engine_->session_ = nullptr;
}

int Session::sync(const int32_t* ids, int64_t n, const ImageSpan* images, size_t image_count,
                  const ProgressFn& progress, const CancelFn&) {
    Fake& st = *engine_->state_;
    if (st.dead) return kDead;
    if (st.sync_status == kRefused) return kRefused;
    st.syncs.emplace_back(ids, ids + n);
    st.sync_images.clear();
    std::vector<float> rows;
    for (size_t i = 0; i < image_count; ++i) {
        st.sync_images.push_back(images[i]);
        rows.insert(rows.end(), images[i].rows, images[i].rows + images[i].token_count * engine_->embd_dim());
    }
    st.sync_rows.push_back(rows);
    st.request_open = true;
    if (progress) progress(SyncProgress{st.reused, n, st.reused, 0.0});
    return st.sync_status;
}

int64_t Session::reused() const { return engine_->state_->reused; }
int64_t Session::held() const { return engine_->state_->held; }

int Session::decode(int32_t token, const Sampling& sampling, int, int32_t* out, int cap) {
    Fake& st = *engine_->state_;
    st.decode_fed.push_back(token);
    st.sampling = sampling;
    if (st.decode_status != kOk) return st.decode_status;
    const int n = std::min<int>(cap, (int) st.decode_out.size());
    std::copy(st.decode_out.begin(), st.decode_out.begin() + n, out);
    return n;
}

int Session::finish() {
    Fake& st = *engine_->state_;
    if (st.request_open) ++st.finishes;
    st.request_open = false;
    return st.dead ? kDead : kOk;
}

const RequestStats& Session::stats() const { return engine_->state_->stats; }

void Session::invalidate() {
    finish();
    ++engine_->state_->invalidates;
}

int64_t Session::ctx() const { return ctx_; }
const std::string& Session::error() const { return engine_->state_->message; }

uint64_t image_rows_key(const float*, size_t) { return 0; }

}  // namespace strata::program

namespace {

renza_tokens tokens_of(std::vector<int> ids) {
    renza_tokens t = {};
    for (int id : ids) renza_tokens_push(&t, id);
    return t;
}

std::vector<int> history(const renza_session* s) {
    const renza_tokens* t = renza_session_tokens(s);
    return std::vector<int>(t->v, t->v + t->len);
}

struct Opened {
    renza_engine* engine = nullptr;
    renza_session* session = nullptr;
};

Opened open_engine(const char* const* extra, int n_extra) {
    renza_engine_options opt;
    renza_engine_options_init(&opt);
    opt.pack_dir = "/pack";
    opt.native_gguf = gguf;
    opt.extra_args = extra;
    opt.n_extra_args = n_extra;
    Opened o;
    char err[256];
    CHECK(renza_engine_open(&o.engine, &opt, err, sizeof err) == RENZA_OK);
    CHECK(renza_session_create(&o.session, o.engine, 32, err, sizeof err) == RENZA_OK);
    return o;
}

void close_engine(Opened& o) {
    renza_session_free(o.session);
    renza_engine_close(o.engine);
}

void test_argv() {
    renza_engine_options opt;
    renza_engine_options_init(&opt);
    opt.pack_dir = "/pack";
    opt.native_gguf = gguf;
    opt.expert_cache = RENZA_OPTION_AUTO;
    opt.prefill_chunk = 8192;
    opt.spec_min_p = 0.70;
    opt.mmproj = "/mmproj.gguf";
    const char* extra[] = {"--adapt-swaps", "0"};
    opt.extra_args = extra;
    opt.n_extra_args = 2;
    renza_engine* e = nullptr;
    char err[256];
    CHECK(renza_engine_open(&e, &opt, err, sizeof err) == RENZA_OK);
    const std::vector<std::string> want = {"renza", "--serve", "--pack", "/pack", "--native", gguf,
                                           "--expert-cache", "auto", "--prefill", "8192", "--spec-min-p",
                                           "0.69999999999999996", "--vision", "--adapt-swaps", "0"};
    CHECK(fake != nullptr && fake->argv == want);
    CHECK(std::strcmp(renza_strata_engine_info(e), "fake=1") == 0);
    CHECK(renza_strata_token_is_eos(e, kEos) && !renza_strata_token_is_eos(e, kEos - 1));
    renza_engine_close(e);
}

void test_vision_open_failures() {
    renza_engine_options opt;
    renza_engine_options_init(&opt);
    opt.native_gguf = gguf;
    opt.mmproj = "/mmproj.gguf";
    renza_engine* e = nullptr;
    char err[256];
    calls.clear();
    CHECK(renza_engine_open(&e, &opt, err, sizeof err) == RENZA_ERROR && e == nullptr);
    CHECK(calls.empty());

    opt.pack_dir = "/pack";
    opt.native_gguf = "/nonexistent/native.gguf";
    CHECK(renza_engine_open(&e, &opt, err, sizeof err) == RENZA_ERROR && e == nullptr);
    CHECK((calls == std::vector<std::string>{"text open"}));
    opt.native_gguf = gguf;

    calls.clear();
    vision_open_fails = true;
    CHECK(renza_engine_open(&e, &opt, err, sizeof err) == RENZA_ERROR && e == nullptr);
    CHECK(std::strcmp(err, "fake vision open failure") == 0);
    CHECK((calls == std::vector<std::string>{"text open", "vision open", "text close"}));
    vision_open_fails = false;

    calls.clear();
    ++vision_width;
    CHECK(renza_engine_open(&e, &opt, err, sizeof err) == RENZA_ERROR && e == nullptr);
    CHECK(std::strstr(err, "does not match native GGUF embedding width") != nullptr);
    CHECK((calls == std::vector<std::string>{"text open", "vision open", "vision close", "text close"}));
    --vision_width;

    calls.clear();
    engine_open_fails = true;
    CHECK(renza_engine_open(&e, &opt, err, sizeof err) == RENZA_ERROR && e == nullptr);
    CHECK(std::strcmp(err, "fake engine open failure") == 0);
    CHECK((calls == std::vector<std::string>{"text open", "vision open", "engine open", "vision close", "text close"}));
    engine_open_fails = false;
}

bool empty_embedding(const renza_vision_embedding& em) {
    return em.rows == nullptr && em.token_count == 0 && em.grid_width == 0 && em.grid_height == 0 && em.key == 0 &&
           std::all_of(std::begin(em.fingerprint), std::end(em.fingerprint), [](uint8_t b) { return b == 0; });
}

void test_vision_encode() {
    renza_engine_options opt;
    renza_engine_options_init(&opt);
    opt.pack_dir = "/pack";
    opt.native_gguf = gguf;
    opt.mmproj = "/mmproj.gguf";
    renza_engine* e = nullptr;
    char err[256];
    engine_width = (int) vision_width;
    calls.clear();
    CHECK(renza_engine_open(&e, &opt, err, sizeof err) == RENZA_OK);
    CHECK((calls == std::vector<std::string>{"text open", "vision open", "engine open"}));
    CHECK(renza_engine_has_vision(e));
    renza_vision_embedding em = {};
    CHECK(renza_engine_vision_encode_memory(e, encoded_image, sizeof(encoded_image), &em, err, sizeof err) == RENZA_OK);
    CHECK(em.rows == encoded_rows && em.token_count == 2 && em.grid_width == 2 && em.grid_height == 1);
    CHECK(em.key == UINT64_C(0x8786858483828180));
    for (size_t i = 0; i < sizeof(em.fingerprint); ++i) CHECK(em.fingerprint[i] == (uint8_t) (0x80 + i));
    const std::vector<float> want_rows(em.rows, em.rows + em.token_count * vision_width);

    renza_tokens prompt = tokens_of({1});
    renza_vision_span span = {};
    CHECK(renza_prompt_append_vision(e, &prompt, &span, &em, err, sizeof err) == RENZA_OK);
    CHECK(empty_embedding(em));
    CHECK((std::vector<int>(prompt.v, prompt.v + prompt.len) ==
           std::vector<int>{1, 248053, (int) program::kImagePad, (int) program::kImagePad, 248054}));
    CHECK(span.token_start == 2 && span.embedding.rows == encoded_rows);
    renza_session* s = nullptr;
    CHECK(renza_session_create(&s, e, 32, err, sizeof err) == RENZA_OK);
    CHECK(renza_session_sync_multimodal(s, &prompt, &span, 1, err, sizeof err) == RENZA_OK);
    CHECK(fake->sync_rows.back() == want_rows && fake->sync_images.size() == 1);
    const program::ImageSpan& image = fake->sync_images.front();
    CHECK(image.start == span.token_start && image.token_count == 2 && image.grid_width == 2 &&
          image.grid_height == 1 && image.rows == encoded_rows && image.key == span.embedding.key);
    CHECK(renza_session_vision_prefix_matches(s, &span, 1));
    renza_vision_embedding_free(&span.embedding);
    CHECK(empty_embedding(span.embedding));
    renza_tokens_free(&prompt);
    renza_session_free(s);

    encode_fails = true;
    em.key = 1;
    CHECK(renza_engine_vision_encode_memory(e, encoded_image, sizeof(encoded_image), &em, err, sizeof err) == RENZA_ERROR);
    CHECK(empty_embedding(em) && std::strcmp(err, "fake encode failure") == 0);
    encode_fails = false;
    calls.clear();
    fake->dead = true;
    em.key = 1;
    CHECK(renza_engine_vision_encode_memory(e, encoded_image, sizeof(encoded_image), &em, err, sizeof err) == RENZA_ENGINE_DEAD);
    CHECK(empty_embedding(em) && calls.empty());
    renza_engine_close(e);
    CHECK((calls == std::vector<std::string>{"engine close", "vision close", "text close"}));
    engine_width = kEmbd;

    const char* vision_only[] = {"--vision"};
    Opened o = open_engine(vision_only, 1);
    calls.clear();
    em.key = 1;
    CHECK(renza_engine_vision_encode_memory(o.engine, encoded_image, sizeof(encoded_image), &em, err, sizeof err) == RENZA_ERROR);
    CHECK(empty_embedding(em) && std::strstr(err, "no vision encoder linked") != nullptr && calls.empty());
    close_engine(o);
    CHECK((calls == std::vector<std::string>{"engine close", "text close"}));
}

struct Progress {
    std::vector<int> done;
    int pos_seen = -1;
    const renza_session* session = nullptr;
};

void on_progress(void* ud, int done, int total) {
    Progress* p = static_cast<Progress*>(ud);
    p->done.push_back(done);
    p->done.push_back(total);
    p->pos_seen = renza_session_pos(p->session);
}

void test_sync_decode() {
    Opened o = open_engine(nullptr, 0);
    Progress progress;
    progress.session = o.session;
    renza_session_set_progress(o.session, on_progress, &progress);
    fake->reused = 2;
    renza_tokens prompt = tokens_of({1, 2, 3, 4});
    char err[256];
    CHECK(renza_session_sync(o.session, &prompt, err, sizeof err) == RENZA_OK);
    CHECK((progress.done == std::vector<int>{2, 4}) && progress.pos_seen == 2);
    CHECK(history(o.session) == (std::vector<int>{1, 2, 3, 4}));
    CHECK(renza_session_checkpoint_valid(o.session) && renza_session_reused(o.session) == 2);

    renza_sampling sampling = {};
    int out[4];
    CHECK(renza_session_decode(o.session, RENZA_TOKEN_NONE, &sampling, 8, out, 4, err, sizeof err) == 3);
    CHECK(fake->decode_fed.back() == program::kNoToken);
    CHECK(fake->sampling.top_p == 1.0f && fake->sampling.penalty_repeat == 1.0f && fake->sampling.top_k == 0);
    CHECK(history(o.session) == (std::vector<int>{1, 2, 3, 4, 7, 8}));
    CHECK(renza_session_decode(o.session, RENZA_TOKEN_NONE, &sampling, 8, out, 4, err, sizeof err) == RENZA_ERROR);
    CHECK(renza_session_decode(o.session, 9, &sampling, 8, out, 4, err, sizeof err) == 3);
    CHECK(history(o.session) == (std::vector<int>{1, 2, 3, 4, 7, 8, 9, 7, 8}));

    // a token right after a sync: the engine syncs it as part of the prompt, then decodes with none
    CHECK(renza_session_sync(o.session, &prompt, err, sizeof err) == RENZA_OK);
    CHECK(renza_session_decode(o.session, 5, &sampling, 8, out, 4, err, sizeof err) == 3);
    CHECK((fake->syncs.back() == std::vector<int32_t>{1, 2, 3, 4, 5}));
    CHECK(fake->decode_fed.back() == program::kNoToken);
    CHECK(history(o.session) == (std::vector<int>{1, 2, 3, 4, 5, 7, 8}));

    renza_strata_request_stats stats;
    fake->stats.generated = 3;
    CHECK(renza_strata_session_finish(o.session, &stats, err, sizeof err) == RENZA_OK && stats.generated == 3);
    CHECK(!renza_session_checkpoint_valid(o.session));
    CHECK(renza_strata_session_finish(o.session, &stats, err, sizeof err) == RENZA_ERROR);

    fake->decode_status = program::kInvalidated;
    CHECK(renza_session_sync(o.session, &prompt, err, sizeof err) == RENZA_OK);
    CHECK(renza_session_decode(o.session, RENZA_TOKEN_NONE, &sampling, 8, out, 4, err, sizeof err) == RENZA_ERROR);
    CHECK(!renza_session_checkpoint_valid(o.session) && renza_session_pos(o.session) == 0);
    renza_tokens_free(&prompt);
    close_engine(o);
}

void test_sync_results() {
    const char* no_cache[] = {"--prompt-cache", "0"};
    for (int cache_off = 0; cache_off < 2; ++cache_off) {
        Opened o = open_engine(no_cache, cache_off ? 2 : 0);
        renza_tokens prompt = tokens_of({1, 2, 3, 4, 5, 6});
        char err[256];
        renza_strata_request_stats stats;
        fake->sync_status = program::kInterrupted;
        fake->held = 3;
        CHECK(renza_session_sync(o.session, &prompt, err, sizeof err) == RENZA_INTERRUPTED);
        CHECK(history(o.session) == (std::vector<int>{1, 2, 3}));
        CHECK(renza_session_checkpoint_valid(o.session) == !cache_off);

        fake->sync_status = program::kRefused;
        CHECK(renza_session_sync(o.session, &prompt, err, sizeof err) == RENZA_ERROR);
        CHECK(history(o.session) == (std::vector<int>{1, 2, 3}));
        CHECK(renza_strata_session_finish(o.session, &stats, err, sizeof err) == RENZA_OK);
        CHECK(renza_session_sync(o.session, &prompt, err, sizeof err) == RENZA_ERROR);
        CHECK(renza_strata_session_finish(o.session, &stats, err, sizeof err) == RENZA_ERROR);

        fake->sync_status = program::kInvalidated;
        CHECK(renza_session_sync(o.session, &prompt, err, sizeof err) == RENZA_ERROR);
        CHECK(renza_session_pos(o.session) == 0 && !renza_session_checkpoint_valid(o.session));
        CHECK(renza_strata_session_finish(o.session, &stats, err, sizeof err) == RENZA_OK);
        renza_tokens_free(&prompt);
        close_engine(o);
    }
}

void test_images_rewind_dead() {
    Opened o = open_engine(nullptr, 0);
    renza_tokens prompt = tokens_of({1, (int) program::kImagePad, (int) program::kImagePad, 2, 3});
    float rows[2 * kEmbd] = {0.5f, 1.5f, 2.5f, 3.5f};
    renza_vision_span image = {};
    image.token_start = 1;
    image.embedding.rows = rows;
    image.embedding.token_count = 2;
    image.embedding.grid_width = 2;
    image.embedding.grid_height = 1;
    char err[256];
    CHECK(renza_session_sync_multimodal(o.session, &prompt, &image, 1, err, sizeof err) == RENZA_OK);
    CHECK(renza_session_has_vision_state(o.session));
    // the rows need only live for the call: a fed token re-syncs them from the adapter's copy
    std::memset(rows, 0, sizeof rows);
    renza_sampling sampling = {};
    int out[4];
    CHECK(renza_session_decode(o.session, 4, &sampling, 8, out, 4, err, sizeof err) == 3);
    CHECK((fake->sync_rows.back() == std::vector<float>{0.5f, 1.5f, 2.5f, 3.5f}));

    const int finishes = fake->finishes;
    renza_session_rewind(o.session, 2);
    CHECK(fake->finishes == finishes + 1);
    CHECK(renza_session_pos(o.session) == 2 && !renza_session_checkpoint_valid(o.session));
    CHECK(!renza_session_has_vision_state(o.session));

    renza_session_invalidate(o.session);
    CHECK(fake->invalidates == 1 && renza_session_pos(o.session) == 0);

    fake->dead = true;
    CHECK(renza_session_sync(o.session, &prompt, err, sizeof err) == RENZA_ENGINE_DEAD);
    CHECK(std::strcmp(err, "fake failure") == 0);
    renza_strata_request_stats stats;
    CHECK(renza_strata_session_finish(o.session, &stats, err, sizeof err) == RENZA_ENGINE_DEAD);
    renza_tokens_free(&prompt);
    close_engine(o);
}

}  // namespace

int main(int argc, char** argv) {
    gguf = argc > 1 ? argv[1] : "";
    if (std::FILE* f = std::fopen(gguf, "rb")) {
        std::fclose(f);
    } else {
        std::printf("SKIP: no native GGUF at '%s'\n", gguf);
        return kSkipped;
    }
    char err[256];
    renza_engine_text* text = renza_engine_text_open(gguf, err, sizeof err);
    if (text == nullptr) {
        std::fprintf(stderr, "renza-adapter-tests: %s\n", err);
        return 1;
    }
    vision_width = (uint32_t) renza_engine_text_arch_u32(text, "embedding_length");
    CHECK(vision_width > 0);
    renza_engine_text_close(text);
    test_vision_open_failures();
    test_vision_encode();
    test_argv();
    test_sync_decode();
    test_sync_results();
    test_images_rewind_dead();
    if (failures) {
        std::fprintf(stderr, "renza-adapter-tests: %d failed\n", failures);
        return 1;
    }
    std::printf("renza-adapter-tests: ok\n");
    return 0;
}
