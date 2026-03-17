/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../hipdnn_ep_runtime.h"
#include "../debug_log.h"
#include "udna_custom_kernels.h"

#include <cstdio>

// =============================================================================
// RotaryEmbedding via custom HIP kernel
// =============================================================================
//
// The HipToLLVM lowering passes flat element counts (input_num_elements,
// cos_cache_num_elements) rather than decomposed shape dimensions.
//
// Shape reconstruction:
//   input:       [batch, seq_len, num_heads * head_dim]
//   cos_cache:   [max_seq, rotary_dim / 2]
//   position_ids:[batch, seq_len]
//
//   half_rot   = rotary_dim / 2
//   head_dim   = rotary_dim  (standard case: rotation covers full head)
//   hidden     = num_heads * head_dim
//   num_positions = input_num_elements / hidden
//
// We treat the input as [num_positions, num_heads, head_dim] with batch=1,
// seq_len=num_positions. This linearization is correct for any batch size
// because the kernel indexes position_ids linearly.
// =============================================================================

int wrap_rotary_embedding(RuntimeState* state,
                          void* input, void* position_ids,
                          void* cos_cache, void* sin_cache,
                          void* output,
                          int64_t interleaved, int64_t num_heads,
                          int64_t rotary_dim,
                          int64_t input_num_elements,
                          int64_t cos_cache_num_elements,
                          int64_t element_size_bytes) {
  if (!state) {
    fprintf(stderr, "wrap_rotary_embedding: null state\n");
    return -1;
  }

  void* stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    fprintf(stderr, "wrap_rotary_embedding: null stream\n");
    return -1;
  }

  int64_t head_dim = rotary_dim;
  int64_t hidden = num_heads * head_dim;
  if (hidden <= 0) {
    fprintf(stderr, "wrap_rotary_embedding: invalid hidden=%lld "
            "(num_heads=%lld, head_dim=%lld)\n",
            (long long)hidden, (long long)num_heads, (long long)head_dim);
    return -1;
  }

  int64_t num_positions = input_num_elements / hidden;
  int64_t half_rot = rotary_dim / 2;
  int64_t max_seq_len = (half_rot > 0) ? cos_cache_num_elements / half_rot : 1;

  RUNTIME_DEBUG_LOG("[REAL] wrap_rotary_embedding: num_positions=%lld, num_heads=%lld, "
                    "head_dim=%lld, rotary_dim=%lld, max_seq_len=%lld, "
                    "interleaved=%lld, elem_size=%lld\n",
                    (long long)num_positions, (long long)num_heads,
                    (long long)head_dim, (long long)rotary_dim,
                    (long long)max_seq_len,
                    (long long)interleaved, (long long)element_size_bytes);

  int rc = udna_rope_forward(
      stream,
      input, position_ids, cos_cache, sin_cache, output,
      /*batch_size=*/1,
      /*seq_len=*/num_positions,
      num_heads, head_dim, rotary_dim, max_seq_len,
      interleaved, element_size_bytes);

  if (rc != 0) {
    fprintf(stderr, "wrap_rotary_embedding: udna_rope_forward failed (rc=%d)\n", rc);
  } else {
    RUNTIME_DEBUG_LOG("[REAL] wrap_rotary_embedding: completed successfully\n");
  }

  return rc;
}
