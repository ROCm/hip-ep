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
#include <string>

// wrap_grid_sample(state, input, grid, output,
//                  data_type, N, C, inH, inW, outH, outW,
//                  mode, padding_mode, align_corners)
//
// mode:          0 = nearest, 1 = bilinear
// padding_mode:  0 = zeros, 1 = border, 2 = reflection
// align_corners: 0 or 1

static int hipdnn_ep_to_hip_dtype_grid_sample(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  case HIPDNN_EP_DATATYPE_DOUBLE:
    return HIP_DTYPE_FLOAT64;
  default:
    return -1;
  }
}

HIPDNN_EP_RT_EXPORT int
wrap_grid_sample(RuntimeState *state, void *input, void *grid, void *output,
                 int64_t data_type, int64_t n, int64_t c, int64_t in_h,
                 int64_t in_w, int64_t out_h, int64_t out_w, int64_t mode,
                 int64_t padding_mode, int64_t align_corners) {
  OP_PROFILE(
      "gridsample",
      [&] {
        char b[160];
        snprintf(b, sizeof(b),
                 "N=%lld C=%lld in=%lldx%lld out=%lldx%lld mode=%lld pad=%lld",
                 (long long)n, (long long)c, (long long)in_h, (long long)in_w,
                 (long long)out_h, (long long)out_w, (long long)mode,
                 (long long)padding_mode);
        return std::string(b);
      },
      state);

  if (!state || !input || !grid || !output) {
    fprintf(stderr, "[REAL] wrap_grid_sample: null argument\n");
    return -1;
  }
  int hip_dtype = hipdnn_ep_to_hip_dtype_grid_sample(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr, "[REAL] wrap_grid_sample: unsupported data_type %lld\n",
            (long long)data_type);
    return -1;
  }
  if (n <= 0 || c <= 0 || in_h <= 0 || in_w <= 0 || out_h <= 0 || out_w <= 0)
    return 0;

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_grid_sample: dtype=%lld N=%lld C=%lld in=%lldx%lld "
      "out=%lldx%lld mode=%lld pad=%lld align=%lld\n",
      (long long)data_type, (long long)n, (long long)c, (long long)in_h,
      (long long)in_w, (long long)out_h, (long long)out_w, (long long)mode,
      (long long)padding_mode, (long long)align_corners);

  int rc = hip_grid_sample(stream, input, grid, output, n, c, in_h, in_w, out_h,
                           out_w, static_cast<int>(mode),
                           static_cast<int>(padding_mode),
                           static_cast<int>(align_corners), hip_dtype);
  if (rc != 0) {
    fprintf(stderr, "[REAL] wrap_grid_sample: kernel launch failed (%d)\n", rc);
    return -1;
  }
  return 0;
}
