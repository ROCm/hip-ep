/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>
#include <hip/hip_runtime.h>

namespace {

int launchTopK(RuntimeState *state, void *x, int64_t k, void *values,
               void *indices, int64_t axis, int64_t largest, int64_t sorted,
               int64_t rank, const int64_t *xShape, int64_t elementSizeBytes,
               const char *abiName) {
  if (!state || !x || !values || !indices || !xShape) {
    RUNTIME_DEBUG_LOG("[REAL] %s: null argument\n", abiName);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    RUNTIME_DEBUG_LOG("[REAL] %s: null HIP stream\n", abiName);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] %s: axis=%lld, k=%lld, rank=%lld, largest=%lld, "
                    "sorted=%lld -> hip_top_k\n",
                    abiName, (long long)axis, (long long)k, (long long)rank,
                    (long long)largest, (long long)sorted);

  return hip_top_k(stream, x, values, indices, axis, largest, sorted, rank,
                   xShape, k, static_cast<int>(elementSizeBytes));
}

} // namespace

int wrap_top_k(RuntimeState *state, void *x, void *k, void *values,
               void *indices, int64_t axis, int64_t largest, int64_t sorted,
               int64_t rank, const int64_t *x_shape, int64_t num_elements,
               int64_t element_size_bytes) {
  OP_PROFILE(
      "top_k",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "r%lld:axis=%lld", (long long)rank,
                 (long long)axis);
        return std::string(b);
      },
      state);

  (void)num_elements;
  if (!state || !k) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_top_k: null argument\n");
    return -1;
  }

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  if (!stream) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_top_k: null HIP stream\n");
    return -1;
  }

  int64_t kValue = 0;
  hipError_t status =
      hipMemcpyAsync(&kValue, k, sizeof(kValue), hipMemcpyDefault, stream);
  if (status != hipSuccess) {
    std::fprintf(stderr, "[REAL] wrap_top_k: failed to read K: %s\n",
                 hipGetErrorString(status));
    return -1;
  }
  status = hipStreamSynchronize(stream);
  if (status != hipSuccess) {
    std::fprintf(stderr, "[REAL] wrap_top_k: stream sync failed: %s\n",
                 hipGetErrorString(status));
    return -1;
  }

  return launchTopK(state, x, kValue, values, indices, axis, largest, sorted,
                    rank, x_shape, element_size_bytes, "wrap_top_k");
}

int wrap_top_k_static(RuntimeState *state, void *x, int64_t k, void *values,
                      void *indices, int64_t axis, int64_t largest,
                      int64_t sorted, int64_t rank, const int64_t *x_shape,
                      int64_t num_elements, int64_t element_size_bytes) {
  OP_PROFILE(
      "top_k",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "r%lld:axis=%lld", (long long)rank,
                 (long long)axis);
        return std::string(b);
      },
      state);

  (void)num_elements;
  return launchTopK(state, x, k, values, indices, axis, largest, sorted, rank,
                    x_shape, element_size_bytes, "wrap_top_k_static");
}
