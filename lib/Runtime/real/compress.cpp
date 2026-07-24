/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

int wrap_compress(RuntimeState *state, void *input, void *condition,
                  void *output, int64_t flatten, int64_t axis,
                  int64_t input_rank, int64_t output_rank,
                  const int64_t *input_shape, const int64_t *output_shape,
                  int64_t condition_len, int64_t num_output_elements,
                  int64_t element_size_bytes) {
  OP_PROFILE(
      "compress",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "flat=%lld:axis=%lld:n=%lld", (long long)flatten,
                 (long long)axis, (long long)num_output_elements);
        return std::string(b);
      },
      state);

  if (!state || !input || !condition || !output || !input_shape ||
      !output_shape) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_compress: null argument\n");
    return -1;
  }

  if (condition_len <= 0 || num_output_elements <= 0)
    return 0;

  size_t workspace_bytes =
      static_cast<size_t>(condition_len) * sizeof(int64_t) +
      sizeof(unsigned long long);
  if (hipdnn_ep_state_ensure_workspace(state, workspace_bytes) != 0) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_compress: ensure_workspace failed\n");
    return -1;
  }
  void *workspace = hipdnn_ep_state_get_workspace(state);

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_compress: flatten=%lld, axis=%lld, cond_len=%lld, "
      "num_out=%lld -> hip_compress\n",
      (long long)flatten, (long long)axis, (long long)condition_len,
      (long long)num_output_elements);

  return hip_compress(stream, input, condition, output, flatten, axis,
                      input_rank, output_rank, input_shape, output_shape,
                      condition_len, num_output_elements, workspace,
                      workspace_bytes, static_cast<int>(element_size_bytes));
}
