/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Runtime wrapper for hip_layer_norm.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "nan_check.h"
#include "runtime_types.h"

#include <cstdio>
#include <cstdint>

static int hipdnn_ep_to_hip_dtype_layer_norm(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  default:
    return -1;
  }
}

extern "C" int wrap_layer_norm(RuntimeState *state, void *x, void *gamma,
                               void *beta, void *y, int64_t outer,
                               int64_t norm_size, double epsilon,
                               int64_t data_type) {
  if (!state || !x || !gamma || !y) {
    fprintf(stderr, "wrap_layer_norm: null argument\n");
    return -1;
  }

  int hip_dtype = hipdnn_ep_to_hip_dtype_layer_norm(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr, "wrap_layer_norm: unsupported data_type %lld (%s)\n",
            (long long)data_type, hipdnn_ep_datatype_name(data_type));
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_layer_norm: dtype=%s, outer=%lld, norm=%lld, eps=%.6f\n",
      hipdnn_ep_datatype_name(data_type), (long long)outer,
      (long long)norm_size, epsilon);

  int rc = hip_layer_norm(stream, x, gamma, beta, y, outer, norm_size,
                          static_cast<float>(epsilon), hip_dtype);
  nan_trace_check("layer_norm", y, outer * norm_size,
                  hipdnn_ep_datatype_size(data_type));
  return rc;
}
