// Bench-only shim, force-included (clang `-include`) when the dispatch bench
// compiles the ARCHIVED lib/Runtime/real/gqa_back.cpp into gqa_back.dll.
//
// The runtime entry symbol is `wrap_group_query_attention`. gqa_back.cpp's
// wrapper definition inherits C linkage from the header declaration; without it
// the definition would be C++-mangled and the BACK export shim could not resolve
// it. This forward declaration (injected before gqa_back.cpp's own includes)
// guarantees the extern "C" linkage -- WITHOUT modifying the archived
// gqa_back.cpp.
#pragma once

#include <cstdint>

struct RuntimeState;

extern "C" int wrap_group_query_attention(
    RuntimeState *state, int op_state_slot, void *query, void *key, void *value,
    void *past_key, void *past_value, void *seqlens_k, void *total_seq_len,
    void *cos_cache, void *sin_cache, void *position_ids, void *attention_bias,
    void *head_sink, void *k_scale, void *v_scale, void *output,
    void *present_key, void *present_value, void *output_qk, int64_t num_heads,
    int64_t kv_num_heads, float scale, int64_t do_rotary,
    int64_t rotary_interleaved, float softcap, int64_t local_window_size,
    int64_t smooth_softmax, int64_t qk_output, int64_t k_quant_type,
    int64_t v_quant_type, int64_t kv_cache_bit_width, int32_t no_causal,
    int64_t batch_size, int64_t seq_len_q, int64_t seq_len_kv,
    int64_t past_buf_seq, int64_t head_dim, int64_t element_size_bytes);
