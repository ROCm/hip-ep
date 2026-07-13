/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

extern "C" int wrap_quantize_linear(
    RuntimeState *state, const void *x, const void *y_scale,
    const void *y_zero_point, void *output, const int64_t *x_shape,
    int64_t x_rank, const int64_t *scale_shape, int64_t scale_rank,
    const int64_t *output_shape, int64_t output_rank, int64_t axis,
    int64_t block_size, int64_t output_dtype_attr, int64_t precision_attr,
    int64_t saturate_attr, int64_t x_dtype, int64_t scale_dtype,
    int64_t output_dtype) {
  OP_PROFILE(
      "quantize_linear",
      [&] {
        char b[96];
        snprintf(b, sizeof(b), "x_rank=%lld scale_rank=%lld axis=%lld",
                 (long long)x_rank, (long long)scale_rank, (long long)axis);
        return std::string(b);
      },
      state);

  (void)output_shape;
  (void)output_rank;
  (void)output_dtype_attr;
  (void)precision_attr;
  (void)saturate_attr;

  if (!state || !x || !y_scale || !output || !x_shape || !scale_shape) {
    fprintf(stderr, "[REAL] wrap_quantize_linear: null required argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  return hip_quantize_linear(stream, x, y_scale, y_zero_point, output, x_shape,
                             static_cast<int>(x_rank), scale_shape,
                             static_cast<int>(scale_rank),
                             static_cast<int>(axis),
                             static_cast<int>(block_size),
                             static_cast<int>(x_dtype),
                             static_cast<int>(scale_dtype),
                             static_cast<int>(output_dtype));
}
