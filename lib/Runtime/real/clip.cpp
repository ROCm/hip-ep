/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Clip: y[i] = min(max(x[i], lo), hi).
// `lo` and `hi` arrive as scalar device pointers (rank-0 memrefs).
// OnnxToHipConversion always provides them, synthesizing dtype defaults
// (numeric_limits::lowest()/max()) when the ONNX model omits the input.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

static int clip_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_INT32:
    return HIP_DTYPE_INT32;
  case HIPDNN_EP_DATATYPE_INT64:
    return HIP_DTYPE_INT64;
  default:
    return -1;
  }
}

int wrap_clip(RuntimeState *state, void *input, void *lo, void *hi,
              void *output, int64_t num_elements, int64_t data_type) {
  OP_PROFILE(
      "clip",
      [&] {
        char buf[48];
        snprintf(buf, sizeof(buf), "%lld:%s", (long long)num_elements,
                 hipdnn_ep_datatype_name(data_type));
        return std::string(buf);
      },
      state);

  if (!state || !input || !lo || !hi || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_clip: null argument\n");
    return -1;
  }
  if (num_elements <= 0) {
    return 0;
  }

  int hip_dtype = clip_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_clip: unsupported data_type=%s(%lld) "
            "(supported: f16, f32, i32, i64)\n",
            hipdnn_ep_datatype_name(data_type), (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_clip: num=%lld, type=%s -> hip_elementwise_clip\n",
      (long long)num_elements, hipdnn_ep_datatype_name(data_type));
  return hip_elementwise_clip(stream, input, lo, hi, output, num_elements,
                              hip_dtype);
}
