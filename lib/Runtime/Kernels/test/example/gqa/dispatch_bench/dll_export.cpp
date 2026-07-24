// ============================================================
// DLL export shims for the GQA dispatch harness.
//
// Each flow's wrapper and op-state constructor (`hipdnn_ep_op_state_construct_gqa`)
// share the entry symbol `wrap_group_query_attention` across the two flows (the
// NEW slim flow in gqa.cpp and the BACK legacy flow in gqa_back.cpp). Because
// each flow is compiled into its own DLL, there is no symbol collision.
// We re-export each flow's entry under stable, distinct names (`gqa_dispatch` /
// `gqa_construct`) that the harness resolves via GetProcAddress. This file is
// compiled once per DLL with GQA_WRAPPER set to the wrapper that DLL contains
// (defaults to the legacy name for the BACK build).
// ============================================================

#include "hipdnn_ep_runtime.h" // RuntimeState

#include <cstdint>

#ifndef GQA_WRAPPER
#define GQA_WRAPPER wrap_group_query_attention
#endif

// Forward-declare the flow's wrapper (name chosen by GQA_WRAPPER) with the
// canonical 39-arg ABI, independent of which name the included header declares.
extern "C" int GQA_WRAPPER(
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

extern "C" int8_t hipdnn_ep_op_state_construct_gqa(RuntimeState *state,
                                                   int32_t slot);

extern "C" __declspec(dllexport) int
gqa_dispatch(RuntimeState *state, int op_state_slot, void *query, void *key,
             void *value, void *past_key, void *past_value, void *seqlens_k,
             void *total_seq_len, void *cos_cache, void *sin_cache,
             void *position_ids, void *attention_bias, void *head_sink,
             void *k_scale, void *v_scale, void *output, void *present_key,
             void *present_value, void *output_qk, int64_t num_heads,
             int64_t kv_num_heads, float scale, int64_t do_rotary,
             int64_t rotary_interleaved, float softcap,
             int64_t local_window_size, int64_t smooth_softmax,
             int64_t qk_output, int64_t k_quant_type, int64_t v_quant_type,
             int64_t kv_cache_bit_width, int32_t no_causal, int64_t batch_size,
             int64_t seq_len_q, int64_t seq_len_kv, int64_t past_buf_seq,
             int64_t head_dim, int64_t element_size_bytes) {
  return GQA_WRAPPER(
      state, op_state_slot, query, key, value, past_key, past_value, seqlens_k,
      total_seq_len, cos_cache, sin_cache, position_ids, attention_bias,
      head_sink, k_scale, v_scale, output, present_key, present_value,
      output_qk, num_heads, kv_num_heads, scale, do_rotary, rotary_interleaved,
      softcap, local_window_size, smooth_softmax, qk_output, k_quant_type,
      v_quant_type, kv_cache_bit_width, no_causal, batch_size, seq_len_q,
      seq_len_kv, past_buf_seq, head_dim, element_size_bytes);
}

extern "C" __declspec(dllexport) int8_t gqa_construct(RuntimeState *state,
                                                      int32_t slot) {
  return hipdnn_ep_op_state_construct_gqa(state, slot);
}
