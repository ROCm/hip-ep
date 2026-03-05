/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"
#include "runtime_types.h"
#include "udna_custom_kernels.h"

#include <cstdio>

int wrap_reduce_sum(RuntimeState* state, void* data, void* axes,
                    void* output, int64_t data_num_elements,
                    int64_t output_num_elements, int64_t element_size_bytes,
                    int64_t keepdims) {
  if (!state || !data || !output) {
    fprintf(stderr, "[REAL] wrap_reduce_sum: null argument\n");
    return -1;
  }

  void* stream = hipdnn_ep_state_get_stream(state);

  int udna_dtype;
  switch (element_size_bytes) {
    case 8: udna_dtype = UDNA_DTYPE_INT64; break;
    case 4: udna_dtype = UDNA_DTYPE_INT32; break;
    default:
      fprintf(stderr,
              "[REAL] wrap_reduce_sum: unsupported element_size=%lld\n",
              (long long)element_size_bytes);
      return -1;
  }

  fprintf(stderr,
          "[REAL] wrap_reduce_sum: data_num=%lld, output_num=%lld, "
          "elem_size=%lld, keepdims=%lld, dtype=%d -> calling udna_reduce_sum\n",
          (long long)data_num_elements, (long long)output_num_elements,
          (long long)element_size_bytes, (long long)keepdims, udna_dtype);

  return udna_reduce_sum(stream, data, output, data_num_elements,
                         output_num_elements, udna_dtype);
}
