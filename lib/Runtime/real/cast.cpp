/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

// Map HIPDNN_EP_DATATYPE_* → hip_dtype_t for custom kernels.
// The two enum systems use different orderings (e.g. bf16=2 vs 5, i64=4 vs 2).
static int hipdnn_to_hip_dtype(int64_t hipdnn_type) {
  switch (hipdnn_type) {
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
  case HIPDNN_EP_DATATYPE_UINT8:
    return HIP_DTYPE_UINT8;
  case HIPDNN_EP_DATATYPE_INT8:
    return HIP_DTYPE_INT8;
  default:
    return -1;
  }
}

int wrap_cast(RuntimeState *state, void *input, void *output,
              int64_t num_elements, int64_t src_data_type,
              int64_t dst_data_type) {
  OP_PROFILE(
      "cast",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "n=%lld", (long long)num_elements);
        return std::string(b);
      },
      state);
  if (!state || !input || !output) {
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_cast: null argument (state=%p input=%p output=%p "
        "n=%lld src=%lld dst=%lld)\n",
        (void *)state, input, output, (long long)num_elements,
        (long long)src_data_type, (long long)dst_data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  int src_hip_dtype = hipdnn_to_hip_dtype(src_data_type);
  int dst_hip_dtype = hipdnn_to_hip_dtype(dst_data_type);

  if (src_hip_dtype < 0 || dst_hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_cast: unsupported data type src=%lld dst=%lld\n",
            (long long)src_data_type, (long long)dst_data_type);
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_cast: num_elements=%lld, src=%s(%lld), "
      "dst=%s(%lld)\n",
      (long long)num_elements, hipdnn_ep_datatype_name(src_data_type),
      (long long)src_data_type, hipdnn_ep_datatype_name(dst_data_type),
      (long long)dst_data_type);

  return hip_cast(stream, input, output, num_elements, src_hip_dtype,
                  dst_hip_dtype);
}
