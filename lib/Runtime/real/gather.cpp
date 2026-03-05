/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"
#include "runtime_types.h"
#include "udna_custom_kernels.h"

#include <cstdio>

int wrap_gather(RuntimeState* state, void* data, void* indices,
                void* output, int64_t axis, int64_t data_num_elements,
                int64_t output_num_elements, int64_t element_size_bytes) {
  if (!state || !data || !indices || !output) {
    fprintf(stderr, "[REAL] wrap_gather: null argument\n");
    return -1;
  }

  void* stream = hipdnn_ep_state_get_stream(state);

  fprintf(stderr,
          "[REAL] wrap_gather: axis=%lld, data_num=%lld, output_num=%lld, "
          "elem_size=%lld -> calling udna_gather\n",
          (long long)axis, (long long)data_num_elements,
          (long long)output_num_elements, (long long)element_size_bytes);

  return udna_gather(stream, data, indices, output, axis,
                     data_num_elements, output_num_elements,
                     static_cast<int>(element_size_bytes));
}
