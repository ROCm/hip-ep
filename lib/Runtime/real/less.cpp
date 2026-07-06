/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Less: y = (a < b)  -- element-wise, same-shape, bool (1-byte) output.
//
// Source: onnxruntime/core/providers/cuda/math/binary_elementwise_ops_impl.cu
//         @ v1.22.2 (BINARY_OP_NAME_EXPR2(Less, (a < b)))
//
// Same same-shape constraint as Equal / Div / Mod (broadcasting via
// upstream Expand).
#include "../cpu_fallback_invoke.h"
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

static int less_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
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

int wrap_less(RuntimeState *state, void *a, void *b, void *output,
              int64_t num_elements, int64_t data_type) {
  OP_PROFILE(
      "less",
      [&] {
        char buf[48];
        snprintf(buf, sizeof(buf), "%lld:%s", (long long)num_elements,
                 hipdnn_ep_datatype_name(data_type));
        return std::string(buf);
      },
      state);

  if (!state || !a || !b || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_less: null argument\n");
    return -1;
  }
  if (num_elements <= 0) {
    return 0;
  }

  int hip_dtype = less_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_less: unsupported input data_type=%s(%lld) "
            "(supported: f16, f32, i32, i64)\n",
            hipdnn_ep_datatype_name(data_type), (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  {
    const int fb_rc = hipdnn_cpu_fb_try_binary_1d(
        state, stream, "Less", a, b, output, num_elements, data_type,
        HIPDNN_EP_DATATYPE_UINT8);
    if (fb_rc == 0)
      return 0;
    if (fb_rc < 0)
      return -1;
  }
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_less: num=%lld, input_type=%s -> hip_elementwise_less\n",
      (long long)num_elements, hipdnn_ep_datatype_name(data_type));
  return hip_elementwise_less(stream, a, b, output, num_elements, hip_dtype);
}
