/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdint>
#include <cstdio>

extern "C" int wrap_transpose(RuntimeState *state, void *input, void *output,
                               int64_t rank, const int64_t *in_shape,
                               int64_t dim0, int64_t dim1, int64_t hip_dtype) {
  if (!state || !input || !output || !in_shape) {
    fprintf(stderr, "wrap_transpose: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_transpose: rank=%lld, dim0=%lld, dim1=%lld, dtype=%lld\n",
      (long long)rank, (long long)dim0, (long long)dim1, (long long)hip_dtype);

  return hip_transpose_nd(stream, input, output, static_cast<int>(rank),
                           in_shape, static_cast<int>(dim0),
                           static_cast<int>(dim1),
                           static_cast<int>(hip_dtype));
}
