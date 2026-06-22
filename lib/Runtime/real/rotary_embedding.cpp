/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

//===----------------------------------------------------------------------===//
// RotaryEmbedding via custom HIP kernel
//===----------------------------------------------------------------------===//
//
// The HipToLLVM lowering passes the four logical dimensions (batch_size,
// seq_len, num_heads, head_dim) plus rotary_dim and a layout flag.  This
// supports the general M-RoPE / partial-RoPE case where rotary_dim may be
// smaller than head_dim, as well as the standard Llama-style case where they
// are equal.
//
// Layouts:
//   is_bnsh == 0 : BSNH [batch, seq_len, num_heads, head_dim]
//                  (also covers 3D [batch, seq_len, num_heads*head_dim])
//   is_bnsh != 0 : BNSH [batch, num_heads, seq_len, head_dim]
//                  (ONNX com.microsoft.RotaryEmbedding 4D default)
//
// max_seq_len for cos/sin cache bounds is derived from
// cos_cache_num_elements / (rotary_dim / 2).
//===----------------------------------------------------------------------===//

int wrap_rotary_embedding(RuntimeState *state, void *input, void *position_ids,
                          void *cos_cache, void *sin_cache, void *output,
                          int64_t interleaved, int64_t batch_size,
                          int64_t seq_len, int64_t num_heads, int64_t head_dim,
                          int64_t rotary_dim, int64_t cos_cache_num_elements,
                          int64_t element_size_bytes, int64_t is_bnsh) {
  OP_PROFILE(
      "rotary_emb",
      [&] {
        char b[96];
        snprintf(b, sizeof(b), "h=%lld,hd=%lld,rd=%lld,bnsh=%lld",
                 (long long)num_heads, (long long)head_dim,
                 (long long)rotary_dim, (long long)is_bnsh);
        return std::string(b);
      },
      state);
  if (!state) {
    hipdnn_ep_log_emit("wrap_rotary_embedding: null state\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    hipdnn_ep_log_emit("wrap_rotary_embedding: null stream\n");
    return -1;
  }

  if (batch_size <= 0 || seq_len <= 0 || num_heads <= 0 || head_dim <= 0 ||
      rotary_dim <= 0) {
    hipdnn_ep_log_emit("wrap_rotary_embedding: invalid dims "
                       "(batch=%lld, seq=%lld, num_heads=%lld, head_dim=%lld, "
                       "rotary_dim=%lld)\n",
                       (long long)batch_size, (long long)seq_len,
                       (long long)num_heads, (long long)head_dim,
                       (long long)rotary_dim);
    return -1;
  }
  if (rotary_dim > head_dim) {
    hipdnn_ep_log_emit(
        "wrap_rotary_embedding: rotary_dim (%lld) must be <= head_dim "
        "(%lld)\n",
        (long long)rotary_dim, (long long)head_dim);
    return -1;
  }

  int64_t half_rot = rotary_dim / 2;
  int64_t max_seq_len = (half_rot > 0) ? cos_cache_num_elements / half_rot : 1;

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_rotary_embedding: batch=%lld, seq_len=%lld, "
      "num_heads=%lld, head_dim=%lld, rotary_dim=%lld, max_seq_len=%lld, "
      "interleaved=%lld, elem_size=%lld, is_bnsh=%lld\n",
      (long long)batch_size, (long long)seq_len, (long long)num_heads,
      (long long)head_dim, (long long)rotary_dim, (long long)max_seq_len,
      (long long)interleaved, (long long)element_size_bytes,
      (long long)is_bnsh);

  int rc = hip_rope_forward(stream, input, position_ids, cos_cache, sin_cache,
                            output, batch_size, seq_len, num_heads, head_dim,
                            rotary_dim, max_seq_len, interleaved,
                            element_size_bytes, is_bnsh);

  if (rc != 0) {
    hipdnn_ep_log_emit(
        "wrap_rotary_embedding: hip_rope_forward failed (rc=%d)\n", rc);
  } else {
    RUNTIME_DEBUG_LOG("[REAL] wrap_rotary_embedding: completed successfully\n");
  }

  return rc;
}
