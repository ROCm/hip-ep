/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Div: y = a / b (element-wise, same-shape).
//
// Source: onnxruntime/core/providers/cuda/math/binary_elementwise_ops_impl.cu
//         @ v1.22.2 (BINARY_OP_NAME_EXPR(Div, (a / b)))
//
// Broadcasting is NOT supported here -- the HipToLLVM Div lowering passes
// num_elements only (no per-input shapes). Any broadcasting must be
// materialised upstream via Expand. This matches the
// BinaryElementWiseNoBroadcastImpl fast-path in the CUDA EP.
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

static int div_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
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

int wrap_div(RuntimeState *state, void *lhs, void *rhs, void *output,
             int64_t num_elements, int64_t data_type) {
  OP_PROFILE(
      "div",
      [&] {
        char b[48];
        snprintf(b, sizeof(b), "%lld:%s", (long long)num_elements,
                 hipdnn_ep_datatype_name(data_type));
        return std::string(b);
      },
      state);

  if (!state || !lhs || !rhs || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_div: null argument\n");
    return -1;
  }
  if (num_elements <= 0) {
    return 0;
  }

  int hip_dtype = div_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_div: unsupported data_type=%s(%lld) "
            "(supported: f16, f32, i32, i64)\n",
            hipdnn_ep_datatype_name(data_type), (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_div: num=%lld, data_type=%s -> hip_elementwise_div\n",
      (long long)num_elements, hipdnn_ep_datatype_name(data_type));
  return hip_elementwise_div(stream, lhs, rhs, output, num_elements, hip_dtype);
}
