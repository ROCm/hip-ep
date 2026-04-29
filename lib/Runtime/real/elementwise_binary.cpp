/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Runtime wrapper for the generic element-wise binary HIP kernel (Div, Pow).
//
// The compiler (HipToLLVM) produces three i64 arrays per call:
//   out_shape[rank], lhs_strides_elems[rank], rhs_strides_elems[rank]
// already padded to the broadcast-compatible rank, with zero strides where
// an axis is broadcast.  We pass them straight through to the kernel.
//
// All compute is GPU-only.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>
#include <cstdint>
#include <vector>

static int hipdnn_ep_to_hip_dtype_binary(int64_t data_type) {
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

extern "C" int
wrap_elementwise_binary(RuntimeState *state, void *lhs, void *rhs, void *out,
                        int64_t num_elements, int64_t data_type, int64_t kind,
                        int64_t rank, const int64_t *out_shape,
                        const int64_t *lhs_strides_elems,
                        const int64_t *rhs_strides_elems) {
  if (!state || !lhs || !rhs || !out) {
    fprintf(stderr, "wrap_elementwise_binary: null argument\n");
    return -1;
  }
  if (!out_shape || !lhs_strides_elems || !rhs_strides_elems) {
    fprintf(stderr, "wrap_elementwise_binary: null shape/stride array\n");
    return -1;
  }

  int hip_dtype = hipdnn_ep_to_hip_dtype_binary(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "wrap_elementwise_binary: unsupported data_type %lld (%s)\n",
            (long long)data_type, hipdnn_ep_datatype_name(data_type));
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_elementwise_binary: kind=%lld, dtype=%s, n=%lld, rank=%lld\n",
      (long long)kind, hipdnn_ep_datatype_name(data_type),
      (long long)num_elements, (long long)rank);


  // Trace inputs only when the heavyweight NaN scanner is explicitly enabled.

  int rc = hip_elementwise_binary(stream, lhs, rhs, out, num_elements, hip_dtype,
                                  static_cast<int>(kind), static_cast<int>(rank),
                                  out_shape, lhs_strides_elems,
                                  rhs_strides_elems);

  return rc;
}
