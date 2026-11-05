// include/strata/core/family_layout.hpp - per model family: the geometry read from a GGUF's metadata.
//
// Header-only and GPU-free: the engine, the tools and the unit tests share it.
//
//   * geometry_from_gguf: fills core::ModelGeometry from "<arch>.*" keys.  For qwen4exp the keys a file does not
//     carry keep Flash-Next's values (the struct defaults, which every published qwen4exp file matches).
//   * compiled_geometry_mismatch: the first field where a geometry differs from the shapes this engine's kernels
//     are built for (the expert count and top-k are run-time values and are not compared).
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/model_family.hpp"
#include "strata/core/geometry.hpp"

#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::core {

/// Fills `g` (and `top_k`, the experts per token) from the metadata shard `meta`.  False with `err` when the
/// architecture guard refuses the file or a key has an impossible value.
inline bool geometry_from_gguf(const strata::GgufFile& meta, ModelGeometry& g, int64_t& top_k, std::string& err) {
    err = strata::check_architecture(meta);
    if (!err.empty()) return false;
    const strata::ModelFamily family = strata::model_family(meta);
    const std::string p = std::string(strata::model_family_arch(family)) + ".";
    g = ModelGeometry{};
    g.family = family;
    // a key the file carries replaces the default; `required` keys were checked present by the guard
    auto num = [&](const char* key, int64_t& dst) {
        if (const strata::MetaValue* v = meta.get(p + key))
            if (v->is_num()) dst = (int64_t) v->u;
    };
    int64_t block_count = g.n_layers, key_length = g.head_dim, value_length = -1;
    num("block_count", block_count);
    num("embedding_length", g.n_embd);
    num("attention.head_count", g.n_head);
    num("attention.head_count_kv", g.n_head_kv);
    num("attention.key_length", key_length);
    num("attention.value_length", value_length);
    num("expert_count", g.n_expert);
    top_k = 10;
    num("expert_used_count", top_k);
    g.n_expert_used = top_k;
    num("expert_feed_forward_length", g.n_ff);
    num("expert_shared_feed_forward_length", g.n_ff_shexp);
    num("ssm.conv_kernel", g.ssm_d_conv);
    num("ssm.state_size", g.ssm_state_size);
    num("ssm.group_count", g.ssm_k_heads);
    num("ssm.time_step_rank", g.ssm_v_heads);
    num("ssm.inner_size", g.ssm_value_dim);
    num("full_attention_interval", g.qsa_interval);
    g.head_dim = key_length;
    num("hyper_connection.count", g.hc);
    num("hyper_connection.low_rank", g.hc_lr);
    num("attention.indexer.head_count", g.idx_q_heads);
    num("attention.indexer.key_length", g.idx_key_dim);
    g.n_layers = block_count;
    g.ssm_conv_channels = 2 * g.ssm_state_size * g.ssm_k_heads + g.ssm_value_dim;
    if (const strata::MetaValue* v = meta.get("tokenizer.ggml.tokens"))
        if (v->type == strata::MetaType::ARRAY && v->count > 0) g.n_vocab = (int64_t) v->count;

    char buf[256];
    auto bad = [&](const char* what) {
        std::snprintf(buf, sizeof buf, "%s: %s", meta.path().c_str(), what);
        err = buf;
        return false;
    };
    if (g.n_layers <= 0 || g.n_embd <= 0 || g.n_head <= 0 || g.n_head_kv <= 0 || g.head_dim <= 0 || g.n_expert <= 0 ||
        top_k <= 0 || top_k > g.n_expert || g.n_ff <= 0 || g.ssm_state_size <= 0 || g.ssm_k_heads <= 0 ||
        g.ssm_v_heads <= 0 || g.ssm_d_conv <= 0 || g.qsa_interval < 2)
        return bad("a geometry key is zero or out of range");
    if (value_length >= 0 && value_length != key_length) return bad("attention.key_length != attention.value_length");
    if (g.ssm_value_dim != g.ssm_state_size * g.ssm_v_heads) return bad("ssm.inner_size != ssm.state_size * ssm.time_step_rank");
    if (g.ssm_v_heads % g.ssm_k_heads) return bad("ssm.time_step_rank is not a multiple of ssm.group_count");
    if (g.n_head % g.n_head_kv) return bad("attention.head_count is not a multiple of attention.head_count_kv");
    if (g.n_layers % g.qsa_interval) return bad("the layer count is not a multiple of full_attention_interval");
    return true;
}

/// The first geometry field that differs from the shapes this engine's kernels are built for (ModelGeometry's
/// defaults, Flash-Next's), as "<what> is <got>, this engine's kernels are built for <want>"; empty when none.
/// n_expert and top-k are run-time sizes (the Coder ships 256 experts) and n_vocab is the embedding's business.
inline std::string compiled_geometry_mismatch(const ModelGeometry& g) {
    const ModelGeometry c;
    struct F {
        const char* what;
        int64_t got, want;
    };
    const F fields[] = {
        {"embedding_length", g.n_embd, c.n_embd},
        {"the layer count", g.n_layers, c.n_layers},
        {"full_attention_interval", g.qsa_interval, c.qsa_interval},
        {"ssm.state_size", g.ssm_state_size, c.ssm_state_size},
        {"ssm.group_count", g.ssm_k_heads, c.ssm_k_heads},
        {"ssm.time_step_rank", g.ssm_v_heads, c.ssm_v_heads},
        {"ssm.conv_kernel", g.ssm_d_conv, c.ssm_d_conv},
        {"the GDN conv channels", g.ssm_conv_channels, c.ssm_conv_channels},
        {"ssm.inner_size", g.ssm_value_dim, c.ssm_value_dim},
        {"attention.head_count", g.n_head, c.n_head},
        {"attention.head_count_kv", g.n_head_kv, c.n_head_kv},
        {"attention.key_length", g.head_dim, c.head_dim},
        {"attention.indexer.head_count", g.idx_q_heads, c.idx_q_heads},
        {"attention.indexer.key_length", g.idx_key_dim, c.idx_key_dim},
        {"hyper_connection.count", g.hc, c.hc},
        {"hyper_connection.low_rank", g.hc_lr, c.hc_lr},
        {"expert_feed_forward_length", g.n_ff, c.n_ff},
        {"expert_shared_feed_forward_length", g.n_ff_shexp, c.n_ff_shexp},
    };
    for (const F& f : fields)
        if (f.got != f.want) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "%s is %lld, this engine's kernels are built for %lld", f.what,
                          (long long) f.got, (long long) f.want);
            return buf;
        }
    return {};
}

}  // namespace strata::core
