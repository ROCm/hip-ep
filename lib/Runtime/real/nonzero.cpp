/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

static int nonzero_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_INT32:
    return HIP_DTYPE_INT32;
  case HIPDNN_EP_DATATYPE_INT64:
    return HIP_DTYPE_INT64;
  case HIPDNN_EP_DATATYPE_INT8:
    return HIP_DTYPE_INT8;
  case HIPDNN_EP_DATATYPE_UINT8:
    return HIP_DTYPE_UINT8;
  default:
    return -1;
  }
}

int wrap_nonzero(RuntimeState *state, void *input, void *output,
                 int32_t *count_ptr, int64_t input_num_elements,
                 int64_t input_rank, const int64_t *input_dims,
                 int64_t output_capacity, int64_t input_data_type) {
  OP_PROFILE(
      "nonzero",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "numel=%lld:rank=%lld:%s",
                 (long long)input_num_elements, (long long)input_rank,
                 hipdnn_ep_datatype_name(input_data_type));
        return std::string(b);
      },
      state);

  if (!state || !input || !output || !count_ptr || !input_dims) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_nonzero: null required argument\n");
    return -1;
  }
  if (input_num_elements <= 0 || input_rank <= 0) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_nonzero: invalid dimensions "
                      "(input_num_elements=%lld, input_rank=%lld)\n",
                      (long long)input_num_elements, (long long)input_rank);
    return -1;
  }

  int hip_dtype = nonzero_hipdnn_to_hip_dtype(input_data_type);
  if (hip_dtype < 0) {
    fprintf(stderr, "[REAL] wrap_nonzero: unsupported data_type=%s(%lld)\n",
            hipdnn_ep_datatype_name(input_data_type),
            (long long)input_data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_nonzero: numel=%lld, rank=%lld, capacity=%lld, "
      "dtype=%s -> hip_nonzero\n",
      (long long)input_num_elements, (long long)input_rank,
      (long long)output_capacity, hipdnn_ep_datatype_name(input_data_type));

  return hip_nonzero(stream, input, output, count_ptr, input_num_elements,
                     input_rank, input_dims, output_capacity, hip_dtype);
}
