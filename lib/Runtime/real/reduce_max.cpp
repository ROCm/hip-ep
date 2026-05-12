/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"
#include "runtime_types.h"

#include <cstdio>

// TODO: Implement actual ReduceMax kernel (similar to reduce_sum).
// Currently a stub that returns success without computing.
int wrap_reduce_max(RuntimeState *state, void *data, void *axes, void *output,
                    int64_t data_num_elements, int64_t output_num_elements,
                    int64_t axes_num_elements, int64_t data_type,
                    int64_t keepdims, int64_t noop_with_empty_axes) {
  if (!state || !data || !output) {
    fprintf(stderr, "[REAL] wrap_reduce_max: null argument\n");
    return -1;
  }

  fprintf(stderr,
          "[REAL] wrap_reduce_max: STUB - not yet implemented "
          "(data_num=%lld, output_num=%lld, data_type=%s)\n",
          (long long)data_num_elements, (long long)output_num_elements,
          hipdnn_ep_datatype_name(data_type));

  return 0;
}
