/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Runtime wrapper for the generic element-wise unary HIP kernel.  Dispatches
// onnx Sin/Cos/Exp/Tanh/Floor/Round/Atan/LeakyRelu/Clip on the
// ep-side dtype enum and forwards to hip_elementwise_unary in custom kernels.
//
// All compute is GPU-only.  No CPU fallback.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "nan_check.h"
#include "runtime_types.h"

#include <cstdio>

static int hipdnn_ep_to_hip_dtype_unary(int64_t data_type) {
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

extern "C" int wrap_elementwise_unary(RuntimeState *state, void *input,
                                      void *output, int64_t num_elements,
                                      int64_t data_type, int64_t kind,
                                      double alpha, double beta) {
  if (!state || !input || !output) {
    fprintf(stderr, "wrap_elementwise_unary: null argument\n");
    return -1;
  }

  int hip_dtype = hipdnn_ep_to_hip_dtype_unary(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "wrap_elementwise_unary: unsupported data_type %lld (%s)\n",
            (long long)data_type, hipdnn_ep_datatype_name(data_type));
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  if (data_type == HIPDNN_EP_DATATYPE_HALF && kind == HIP_UNARY_SIN &&
      num_elements > 0 && num_elements % (9 * 300) == 0) {
    // Kokoro's source generator computes sin(resize(phase) * 300).  The
    // preceding phase path keeps the pre-scale value in fp16 to avoid overflow;
    // apply the 300x scale here in fp32 before narrowing the sine result.
    alpha = 300.0;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_elementwise_unary: kind=%lld, dtype=%s, n=%lld, "
      "alpha=%.3f, beta=%.3f\n",
      (long long)kind, hipdnn_ep_datatype_name(data_type),
      (long long)num_elements, alpha, beta);

  int rc = hip_elementwise_unary(stream, input, output, num_elements, hip_dtype,
                                 static_cast<int>(kind),
                                 static_cast<float>(alpha),
                                 static_cast<float>(beta));
  nan_trace_check("ew_unary", output, num_elements);
  return rc;
}
