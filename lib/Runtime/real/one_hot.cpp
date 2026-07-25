/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>
#include <cstring>
#include <hip/hip_runtime.h>

static int64_t read_depth_scalar(const void *bytes, int depth_elem_bytes) {
  int64_t depth_val = 0;
  if (depth_elem_bytes == 8) {
    int64_t v = 0;
    std::memcpy(&v, bytes, sizeof(v));
    depth_val = v;
  } else if (depth_elem_bytes == 4) {
    int32_t v = 0;
    std::memcpy(&v, bytes, sizeof(v));
    depth_val = v;
  } else if (depth_elem_bytes == 2) {
    int16_t v = 0;
    std::memcpy(&v, bytes, sizeof(v));
    depth_val = v;
  } else if (depth_elem_bytes == 1) {
    int8_t v = 0;
    std::memcpy(&v, bytes, sizeof(v));
    depth_val = v;
  }
  return depth_val;
}

int wrap_one_hot(RuntimeState *state, void *indices, void *depth, void *values,
                 void *output, int64_t axis, int64_t indices_rank,
                 int64_t output_rank, const int64_t *indices_shape,
                 const int64_t *output_shape, int64_t num_indices,
                 int64_t num_output_elements, int64_t element_size_bytes,
                 int64_t indices_element_size_bytes,
                 int64_t depth_element_size_bytes) {
  OP_PROFILE(
      "one_hot",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "axis=%lld:n=%lld", (long long)axis,
                 (long long)num_indices);
        return std::string(b);
      },
      state);

  if (!state || !indices || !depth || !values || !output || !indices_shape ||
      !output_shape) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_one_hot: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  char depth_bytes[8] = {};
  hipError_t err = hipMemcpyAsync(
      depth_bytes, depth, static_cast<size_t>(depth_element_size_bytes),
      hipMemcpyDeviceToHost, static_cast<hipStream_t>(stream));
  if (err != hipSuccess)
    return static_cast<int>(err);
  err = hipStreamSynchronize(static_cast<hipStream_t>(stream));
  if (err != hipSuccess)
    return static_cast<int>(err);

  int64_t depth_host = read_depth_scalar(
      depth_bytes, static_cast<int>(depth_element_size_bytes));

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_one_hot: axis=%lld, depth=%lld, idx_rank=%lld -> "
      "hip_one_hot\n",
      (long long)axis, (long long)depth_host, (long long)indices_rank);

  return hip_one_hot(stream, indices, depth, values, output, axis, indices_rank,
                     output_rank, indices_shape, output_shape, num_indices,
                     num_output_elements, depth_host,
                     static_cast<int>(element_size_bytes),
                     static_cast<int>(indices_element_size_bytes));
}
