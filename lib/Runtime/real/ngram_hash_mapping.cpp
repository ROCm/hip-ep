/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

int wrap_ngram_hash_mapping(RuntimeState *state, void *input_ids,
                            void *multipliers, void *vocab_sizes,
                            void *past_ids, void *head_offsets,
                            void *eos_token_id, void *segment_ids,
                            void *hash_ids, void *present_ids, int64_t batch,
                            int64_t seq_len, int64_t max_ngram_size,
                            int64_t n_head_per_ngram, int64_t pad_id,
                            int64_t reset_on_eos, int64_t element_size_bytes) {
  OP_PROFILE(
      "ngram_hash_mapping",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "B=%lld:S=%lld", (long long)batch,
                 (long long)seq_len);
        return std::string(b);
      },
      state);

  if (!state || !input_ids || !multipliers || !vocab_sizes || !hash_ids) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_ngram_hash_mapping: null argument\n");
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_ngram_hash_mapping: B=%lld S=%lld N=%lld "
                    "heads=%lld elem=%lld -> hip_ngram_hash_mapping\n",
                    (long long)batch, (long long)seq_len,
                    (long long)max_ngram_size, (long long)n_head_per_ngram,
                    (long long)element_size_bytes);

  return hip_ngram_hash_mapping(
      hipdnn_ep_state_get_stream(state), input_ids, multipliers, vocab_sizes,
      past_ids, head_offsets, eos_token_id,
      static_cast<const int32_t *>(segment_ids), hash_ids, present_ids, batch,
      seq_len, max_ngram_size, n_head_per_ngram, pad_id, reset_on_eos,
      static_cast<int>(element_size_bytes));
}
