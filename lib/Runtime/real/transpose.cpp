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
  if (!state || !input || !output || !input_shape || !perm) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_transpose: null argument\n");
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

  return hip_transpose(stream, input, output, rank, input_shape, perm,
                       num_elements, static_cast<int>(element_size_bytes));
}
