/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>
#include <string>

static int hipdnn_ep_to_hip_dtype_trilu(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  case HIPDNN_EP_DATATYPE_DOUBLE:
    return HIP_DTYPE_FLOAT64;
  default:
    return -1;
  }
}

int wrap_trilu(RuntimeState *state, void *input, void *output,
               int64_t input_elements, int64_t num_elements, int64_t rows,
               int64_t cols, int64_t k, int64_t upper, int64_t data_type) {
  OP_PROFILE(
      "trilu",
      [&] {
        char b[96];
        snprintf(b, sizeof(b), "n=%lld %lldx%lld", (long long)num_elements,
                 (long long)rows, (long long)cols);
        return std::string(b);
      },
      state);
  if (!state || !input || !output) {
    fprintf(stderr, "[REAL] wrap_trilu: null argument\n");
    return -1;
  }

  int hip_dtype = hipdnn_ep_to_hip_dtype_trilu(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr, "[REAL] wrap_trilu: unsupported data_type %lld\n",
            (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  int result = hip_trilu(stream, input, output, input_elements, num_elements,
                         rows, cols, k, upper, hip_dtype);
  if (result != 0) {
    fprintf(stderr, "[REAL] wrap_trilu: kernel launch failed (%d)\n", result);
    return -1;
  }
  return 0;
}
