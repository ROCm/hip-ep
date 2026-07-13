/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- paged_attention.cpp - Paged Attention runtime wrapper --------------===//
//
// Implements wrap_paged_attention for com.microsoft.PagedAttention (Phase 4b).
//
// Call flow:
//   1. hip_paged_kv_write — scatter new K/V tokens into the paged cache slab
//      using slot_mapping[token_idx] to identify the physical slot.
//      For the fused path (packed QKV), K and V are split first.
//   2. hip_paged_flash_decode — FA-2 split-K decode over the paged KV slab
//      (decode path: num_tokens == batch_size, one query token per sequence).
//   3. hip_paged_flash_decode_reduce — merge split-K partials into output.
//
// Prefill (num_tokens > batch_size) is not yet implemented — the function
// returns -1 for that case with a diagnostic log.
//
// KV slab layout (NHD, ORT-compatible):
//   key_cache / value_cache: [num_blocks, block_size, kv_num_heads, head_dim]
//
//===----------------------------------------------------------------------===//

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../runtime_state_internal.h"
#include "error_check_macros.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>

// Number of split-K partitions for the paged decode kernel.
// Mirrors kFlashDecodeKSplits in gqa_kernel.hip; 8 gives good occupancy
// for the gfx1151 16-CU target at sequence lengths 256–16K.
static constexpr int kPagedKSplits = 8;

// =========================================================================
// wrap_paged_attention
// =========================================================================

extern "C" int wrap_paged_attention(
    RuntimeState *state,
    // Core tensors
    void *query,            // [num_tokens, H*D] or [num_tokens, (H+2G)*D] packed
    void *key,              // [num_tokens, G*D] — nullptr if packed QKV
    void *value,            // [num_tokens, G*D] — nullptr if packed QKV
    void *key_cache,        // [num_blocks, block_size, G, D]  paged K slab
    void *value_cache,      // [num_blocks, block_size, G, D]  paged V slab
    void *block_table,      // [batch, max_blocks_per_seq]  int32
    void *slot_mapping,     // [num_tokens]  int32  physical slot per new token
    void *sequence_lengths, // [batch]  int32  KV sequence lengths (post-write)
    // Output
    void *output, // [num_tokens, H*D]
    // Attributes
    int64_t num_heads, int64_t kv_num_heads, float scale, int64_t do_rotary,
    void *cos_cache, void *sin_cache, int64_t num_tokens, int64_t batch_size,
    int64_t head_dim, int64_t element_size_bytes, int64_t block_size,
    int64_t max_blocks_per_seq) {
  OP_PROFILE(
      "paged_attention",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "B%lldH%lldG%lldD%lld", (long long)batch_size,
                 (long long)num_heads, (long long)kv_num_heads,
                 (long long)head_dim);
        return std::string(b);
      },
      state);

  if (!state || !query || !key_cache || !value_cache || !block_table ||
      !slot_mapping || !sequence_lengths || !output) {
    fprintf(stderr,
            "wrap_paged_attention: null argument (state=%p query=%p "
            "key_cache=%p value_cache=%p block_table=%p slot_mapping=%p "
            "seq_lens=%p output=%p)\n",
            (void *)state, query, key_cache, value_cache, block_table,
            slot_mapping, sequence_lengths, output);
    return -1;
  }

  if (element_size_bytes != 2) {
    fprintf(stderr,
            "wrap_paged_attention: unsupported element_size_bytes=%lld "
            "(only fp16 supported)\n",
            (long long)element_size_bytes);
    return -1;
  }

  if (block_size != 16) {
    fprintf(stderr,
            "wrap_paged_attention: unsupported block_size=%lld (only 16 "
            "supported)\n",
            (long long)block_size);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    fprintf(stderr, "wrap_paged_attention: null stream\n");
    return -1;
  }

  const int H = static_cast<int>(num_heads);
  const int G = static_cast<int>(kv_num_heads);
  const int D = static_cast<int>(head_dim);
  const int B = static_cast<int>(batch_size);
  const int N = static_cast<int>(num_tokens);
  const int mbs = static_cast<int>(max_blocks_per_seq);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_paged_attention: B=%d N=%d H=%d G=%d D=%d block_size=%lld "
      "max_blocks=%d\n",
      B, N, H, G, D, (long long)block_size, mbs);

  // Currently only support decode (one new token per sequence in the batch).
  // Prefill (N > B) requires varlen attention — deferred to Milestone 3.
  if (N != B) {
    fprintf(
        stderr,
        "wrap_paged_attention: prefill not yet supported (num_tokens=%d != "
        "batch_size=%d). Returning -1.\n",
        N, B);
    return -1;
  }

  // -------------------------------------------------------------------------
  // Step 1: Write new K/V tokens into the paged cache.
  // slot_mapping[i] gives the flat physical slot for token i.
  // -------------------------------------------------------------------------
  {
    const void *K_src = key;
    const void *V_src = value;

    // When packed QKV (key == nullptr), we'd split here; for now reject.
    if (!K_src || !V_src) {
      fprintf(stderr,
              "wrap_paged_attention: packed QKV not yet supported; key and "
              "value must be non-null\n");
      return -1;
    }

    RUNTIME_DEBUG_LOG("[REAL] wrap_paged_attention: step 1 — KV write\n");
    HIP_CHECK_GOTO(
        hip_paged_kv_write(
            stream, K_src, V_src, key_cache, value_cache,
            static_cast<const int *>(slot_mapping), N, G, D,
            static_cast<int>(block_size), static_cast<int>(element_size_bytes),
            do_rotary ? cos_cache : nullptr, do_rotary ? sin_cache : nullptr,
            /*past_lens=*/nullptr, /*token_to_seq=*/nullptr),
        cleanup);
  }

  // -------------------------------------------------------------------------
  // Step 2: Paged flash decode.
  // Allocate partials scratch [B, H, K_SPLITS, D+2] from the scratch arena.
  // -------------------------------------------------------------------------
  {
    const size_t partials_bytes =
        static_cast<size_t>(B) * H * kPagedKSplits * (D + 2) * sizeof(float);
    hipdnn_ep_scratch_restore(state, 0);
    hipdnn_ep_scratch_reserve(state, partials_bytes);
    float *partials = static_cast<float *>(
        hipdnn_ep_scratch_alloc(state, partials_bytes));
    if (!partials) {
      fprintf(stderr,
              "wrap_paged_attention: scratch_alloc failed for partials "
              "(%zu bytes)\n",
              partials_bytes);
      return -1;
    }

    RUNTIME_DEBUG_LOG("[REAL] wrap_paged_attention: step 2 — paged decode\n");
    HIP_CHECK_GOTO(
        hip_paged_flash_decode(stream, query, key_cache, value_cache,
                               static_cast<const int *>(block_table),
                               static_cast<const int *>(sequence_lengths),
                               partials, B, H, G, D, mbs,
                               static_cast<int>(block_size), kPagedKSplits,
                               scale, static_cast<int>(element_size_bytes)),
        cleanup);

    // -----------------------------------------------------------------------
    // Step 3: Reduce partials → output.
    // -----------------------------------------------------------------------
    RUNTIME_DEBUG_LOG("[REAL] wrap_paged_attention: step 3 — reduce\n");
    HIP_CHECK_GOTO(
        hip_paged_flash_decode_reduce(stream, partials, output, B, H, D,
                                      kPagedKSplits,
                                      static_cast<int>(element_size_bytes)),
        cleanup);
  }

  return 0;

cleanup:
  return -1;
}
