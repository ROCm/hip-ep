/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Cos: y = cos(x) (element-wise).
//
// Source: onnxruntime/core/providers/cuda/math/unary_elementwise_ops_impl.cu
//         @ v1.22.2 (UNARY_OP_NAME_EXPR(Cos, _Cos(a)),
//                    SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Cos))
//         + core/providers/cuda/cu_inc/common.cuh _Cos<half> / _Cos<float>.
#include "../cpu_fallback_invoke.h"
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

static int cos_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  default:
    return -1;
  }
}

int wrap_cos(RuntimeState *state, void *input, void *output,
             int64_t num_elements, int64_t data_type) {
  OP_PROFILE(
      "cos",
      [&] {
        char b[48];
        snprintf(b, sizeof(b), "%lld:%s", (long long)num_elements,
                 hipdnn_ep_datatype_name(data_type));
        return std::string(b);
      },
      state);

  if (!state || !input || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_cos: null argument\n");
    return -1;
  }
  if (num_elements <= 0) {
    return 0;
  }

  int hip_dtype = cos_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_cos: unsupported data_type=%s(%lld) "
            "(supported: f16, f32)\n",
            hipdnn_ep_datatype_name(data_type), (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  {
    const int fb_rc = hipdnn_cpu_fb_try_unary_1d(state, stream, "Cos", input,
                                                 output, num_elements, data_type);
    if (fb_rc == 0)
      return 0;
    if (fb_rc < 0)
      return -1;
  }
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_cos: num=%lld, data_type=%s -> hip_elementwise_cos\n",
      (long long)num_elements, hipdnn_ep_datatype_name(data_type));
  return hip_elementwise_cos(stream, input, output, num_elements, hip_dtype);
}
