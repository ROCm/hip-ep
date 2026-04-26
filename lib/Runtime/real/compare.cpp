/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Runtime wrapper for hip_compare (Equal/Greater/Less/GreaterOrEqual/And).

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "nan_check.h"
#include "runtime_types.h"

#include <cstdio>
#include <cstdint>

static int hipdnn_ep_to_hip_dtype_compare(int64_t data_type) {
  switch (data_type) {
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

extern "C" int wrap_compare(RuntimeState *state, void *lhs, void *rhs, void *out,
                            int64_t num_elements, int64_t data_type, int64_t kind,
                            int64_t rank, const int64_t *out_shape,
                            const int64_t *lhs_strides_elems,
                            const int64_t *rhs_strides_elems) {
  if (!state || !lhs || !rhs || !out) {
    fprintf(stderr, "wrap_compare: null argument\n");
    return -1;
  }
  // For And the dtype isn't used, so allow any value.
  int hip_dtype = (kind == 4 /*And*/) ? 0 : hipdnn_ep_to_hip_dtype_compare(data_type);
  if (kind != 4 && hip_dtype < 0) {
    fprintf(stderr, "wrap_compare: unsupported data_type %lld (%s)\n",
            (long long)data_type, hipdnn_ep_datatype_name(data_type));
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_compare: kind=%lld, dtype=%s, n=%lld, rank=%lld\n",
      (long long)kind, hipdnn_ep_datatype_name(data_type),
      (long long)num_elements, (long long)rank);

  int rc = hip_compare(stream, lhs, rhs, out, num_elements, hip_dtype,
                       static_cast<int>(kind), static_cast<int>(rank), out_shape,
                       lhs_strides_elems, rhs_strides_elems);
  if (rc == 0)
    nan_trace_check("compare", out, num_elements);
  return rc;
}
