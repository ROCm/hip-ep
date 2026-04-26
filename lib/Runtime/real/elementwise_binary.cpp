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
#include "nan_check.h"
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

  // Compute actual rhs element count from strides
  int64_t rhs_max_offset = 0;
  for (int64_t d = 0; d < rank; d++) {
    if (rhs_strides_elems[d] > 0)
      rhs_max_offset += (out_shape[d] - 1) * rhs_strides_elems[d];
  }
  int64_t rhs_actual_n = rhs_max_offset + 1;

  // Trace inputs
  if (!g_nan_first_found) {
    int op_id = g_nan_trace_counter + 1;
    nan_trace_check_input("ew_binary", op_id, "lhs", lhs, num_elements);
    // Check rhs with ACTUAL valid count, not output count
    nan_trace_check_input("ew_binary", op_id, "rhs_actual", rhs, rhs_actual_n);
    fprintf(stderr,
            "[NAN_TRACE] op#%d ew_binary: kind=%lld rank=%lld n=%lld "
            "rhs_actual_n=%lld",
            op_id, (long long)kind, (long long)rank,
            (long long)num_elements, (long long)rhs_actual_n);
    fprintf(stderr, " shape=[");
    for (int64_t d = 0; d < rank; d++)
      fprintf(stderr, "%s%lld", d ? "," : "", (long long)out_shape[d]);
    fprintf(stderr, "] rhs_str=[");
    for (int64_t d = 0; d < rank; d++)
      fprintf(stderr, "%s%lld", d ? "," : "", (long long)rhs_strides_elems[d]);
    fprintf(stderr, "]\n");
    fflush(stderr);
  }

  int rc = hip_elementwise_binary(stream, lhs, rhs, out, num_elements, hip_dtype,
                                  static_cast<int>(kind), static_cast<int>(rank),
                                  out_shape, lhs_strides_elems,
                                  rhs_strides_elems);

  nan_trace_check("ew_binary", out, num_elements);
  return rc;
}
