/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>

int wrap_transpose(RuntimeState *state, const void *input, void *output,
                   int64_t rank, const int64_t *input_shape,
                   const int64_t *perm, int64_t num_elements,
                   int64_t element_size_bytes) {
  OP_PROFILE(
      "transpose",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "n=%lld rank=%lld", (long long)num_elements,
                 (long long)rank);
        return std::string(b);
      },
      state);
  // Empty transpose is a no-op (NonZero count=0 → [3,0] indices in embedding).
  if (num_elements <= 0)
    return 0;

  if (!state || !input || !output || !input_shape || !perm) {
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_transpose: null argument (state=%p input=%p output=%p "
        "input_shape=%p perm=%p rank=%lld num=%lld elem=%lld)\n",
        (void *)state, input, output, (const void *)input_shape,
        (const void *)perm, (long long)rank, (long long)num_elements,
        (long long)element_size_bytes);
    if (input_shape && rank > 0 && rank <= 8) {
      fprintf(stderr, "  input_shape=[");
      for (int64_t i = 0; i < rank; ++i)
        fprintf(stderr, "%lld%s", (long long)input_shape[i],
                i + 1 == rank ? "" : ",");
      fprintf(stderr, "]");
      if (perm) {
        fprintf(stderr, "  perm=[");
        for (int64_t i = 0; i < rank; ++i)
          fprintf(stderr, "%lld%s", (long long)perm[i],
                  i + 1 == rank ? "" : ",");
        fprintf(stderr, "]");
      }
      fprintf(stderr, "\n");
    }
    return -1;
  }
  if (rank <= 0) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_transpose: invalid rank=%lld\n",
                      (long long)rank);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_transpose: rank=%lld, num_elements=%lld, elem_size=%lld\n",
      (long long)rank, (long long)num_elements, (long long)element_size_bytes);

  // Fast path: a batched last-two-dim swap of 1/2/4/8-byte elements
  // (perm = [0,1,...,r-3, r-1, r-2]) is the layout flip around
  // linear_attention ([1, S, F] <-> [1, F, S]) that dominates transpose GPU
  // time. The generic kernel below gathers per output element with an
  // uncoalesced read there; the tiled kernel restores coalescing. Bit-exact
  // data movement, so it cannot change any model's output. Falls through to the
  // generic path if the tiled launcher declines (unsupported width or
  // batch > gridDim.z limit).
  if (rank >= 2 && (element_size_bytes == 1 || element_size_bytes == 2 ||
                    element_size_bytes == 4 || element_size_bytes == 8)) {
    bool last_two_swap =
        perm[rank - 1] == rank - 2 && perm[rank - 2] == rank - 1;
    for (int64_t i = 0; last_two_swap && i < rank - 2; ++i)
      last_two_swap = perm[i] == i;
    if (last_two_swap) {
      int64_t rows = input_shape[rank - 2];
      int64_t cols = input_shape[rank - 1];
      int64_t batch = 1;
      for (int64_t i = 0; i < rank - 2; ++i)
        batch *= input_shape[i];
      int rc = hip_transpose_2d_tiled(stream, input, output, batch, rows, cols,
                                      static_cast<int>(element_size_bytes));
      if (rc == 0) {
        RUNTIME_DEBUG_LOG("[REAL] wrap_transpose: tiled 2D path "
                          "(batch=%lld rows=%lld cols=%lld)\n",
                          (long long)batch, (long long)rows, (long long)cols);
        return 0;
      }
      if (rc < 0) // hard launch failure -> surface it
        return rc;
      // rc == 1: declined (e.g. batch > gridDim.z limit) -> generic fallback.
    }
  }

  return hip_transpose(stream, input, output, rank, input_shape, perm,
                       num_elements, static_cast<int>(element_size_bytes));
}
