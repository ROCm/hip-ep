/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"

#include <cstdio>

// Map HIPDNN_EP_DATATYPE_* to hip_dtype_t (X/Y element type for Where).
// The condition tensor is always 1-byte bool; only the X/Y dtype is needed
// here so the kernel can dispatch to the correct templated launcher.
static int hipdnn_to_hip_dtype_where(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  case HIPDNN_EP_DATATYPE_INT32:
    return HIP_DTYPE_INT32;
  case HIPDNN_EP_DATATYPE_INT64:
    return HIP_DTYPE_INT64;
  default:
    return -1;
  }
}

int wrap_where(RuntimeState *state, void *condition, void *x, void *y,
               void *output, const int64_t *cond_shape, int64_t cond_rank,
               const int64_t *x_shape, int64_t x_rank, const int64_t *y_shape,
               int64_t y_rank, const int64_t *out_shape, int64_t out_rank,
               int64_t data_type) {
  if (!state || !condition || !x || !y || !output) {
    fprintf(stderr, "wrap_where: null tensor argument\n");
    return -1;
  }
  // Shape pointers may be null only when the corresponding rank is 0 (scalar).
  if ((cond_rank > 0 && !cond_shape) || (x_rank > 0 && !x_shape) ||
      (y_rank > 0 && !y_shape) || (out_rank > 0 && !out_shape)) {
    fprintf(stderr, "wrap_where: null shape argument with non-zero rank\n");
    return -1;
  }

  int hip_dtype = hipdnn_to_hip_dtype_where(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr, "[REAL] wrap_where: unsupported data_type %lld (%s)\n",
            (long long)data_type, hipdnn_ep_datatype_name(data_type));
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  if (hipdnn_ep_debug_enabled()) {
    auto dump_shape = [](const char *name, const int64_t *shape, int64_t rank) {
      fprintf(stderr, "[REAL] wrap_where: %s rank=%lld shape=[", name,
              (long long)rank);
      for (int64_t i = 0; i < rank; ++i)
        fprintf(stderr, "%s%lld", i ? "," : "", (long long)shape[i]);
      fprintf(stderr, "]\n");
    };
    fprintf(stderr, "[REAL] wrap_where: dtype=%s(%lld)\n",
            hipdnn_ep_datatype_name(data_type), (long long)data_type);
    dump_shape("condition", cond_shape, cond_rank);
    dump_shape("x", x_shape, x_rank);
    dump_shape("y", y_shape, y_rank);
    dump_shape("output", out_shape, out_rank);
  }

  return hip_elementwise_where(stream, condition, x, y, output, cond_shape,
                               cond_rank, x_shape, x_rank, y_shape, y_rank,
                               out_shape, out_rank, hip_dtype);
}
