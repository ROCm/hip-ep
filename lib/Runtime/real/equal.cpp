/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Equal: y = (a == b)  -- element-wise, same-shape, bool (1-byte) output.
//
// `data_type` refers to the INPUT type (the comparison operand type). The
// output is always 1 byte per element.
//
// Source: onnxruntime/core/providers/cuda/math/binary_elementwise_ops_impl.cu
//         @ v1.22.2 (BINARY_OP_NAME_EXPR2(Equal, (a == b)))
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

static int equal_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
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

int wrap_equal(RuntimeState *state, void *a, void *b, void *output,
               int64_t a_num_elements, int64_t b_num_elements,
               int64_t out_num_elements, int64_t data_type) {
  OP_PROFILE(
      "equal",
      [&] {
        char buf[80];
        snprintf(buf, sizeof(buf), "a=%lld:b=%lld:out=%lld:%s",
                 (long long)a_num_elements, (long long)b_num_elements,
                 (long long)out_num_elements,
                 hipdnn_ep_datatype_name(data_type));
        return std::string(buf);
      },
      state);

  if (!state || !a || !b || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_equal: null argument\n");
    return -1;
  }
  if (out_num_elements <= 0) {
    return 0;
  }

  int hip_dtype = equal_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_equal: unsupported input data_type=%s(%lld) "
            "(supported: f16, f32, i32, i64)\n",
            hipdnn_ep_datatype_name(data_type), (long long)data_type);
    return -1;
  }

  // Supported broadcast modes: same-shape (a_n == b_n == out_n), scalar-rhs
  // (b_n == 1), or scalar-lhs (a_n == 1). Anything more general must be
  // materialised upstream via Expand; reject so the kernel never reads OOB.
  if (a_num_elements != out_num_elements && a_num_elements != 1) {
    fprintf(stderr,
            "[REAL] wrap_equal: unsupported broadcast: a_num=%lld out=%lld\n",
            (long long)a_num_elements, (long long)out_num_elements);
    return -1;
  }
  if (b_num_elements != out_num_elements && b_num_elements != 1) {
    fprintf(stderr,
            "[REAL] wrap_equal: unsupported broadcast: b_num=%lld out=%lld\n",
            (long long)b_num_elements, (long long)out_num_elements);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_equal: a_num=%lld, b_num=%lld, out=%lld, input_type=%s "
      "-> hip_elementwise_equal\n",
      (long long)a_num_elements, (long long)b_num_elements,
      (long long)out_num_elements, hipdnn_ep_datatype_name(data_type));
  return hip_elementwise_equal(stream, a, b, output, a_num_elements,
                               b_num_elements, out_num_elements, hip_dtype);
}
