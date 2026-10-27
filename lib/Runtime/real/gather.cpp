/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../runtime_state_internal.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>

int wrap_gather(RuntimeState *state, void *data, void *indices, void *output,
                int64_t axis, int64_t data_num_elements,
                int64_t indices_num_elements, int64_t output_num_elements,
                int64_t axis_size, int64_t inner_size,
                int64_t element_size_bytes,
                int64_t indices_element_size_bytes) {
  OP_PROFILE(
      "gather",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "n=%lld", (long long)output_num_elements);
        return std::string(b);
      },
      state);
  if (!state || !data || !indices || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_gather: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_gather: axis=%lld, data_num=%lld, indices_num=%lld, "
      "output_num=%lld, axis_size=%lld, inner=%lld, elem_size=%lld, "
      "idx_elem_size=%lld -> calling hip_gather\n",
      (long long)axis, (long long)data_num_elements,
      (long long)indices_num_elements, (long long)output_num_elements,
      (long long)axis_size, (long long)inner_size,
      (long long)element_size_bytes, (long long)indices_element_size_bytes);

  return hip_gather(stream, data, indices, output, axis, data_num_elements,
                    indices_num_elements, output_num_elements, axis_size,
                    inner_size, static_cast<int>(element_size_bytes),
                    static_cast<int>(indices_element_size_bytes));
}

int wrap_gather_from_file(RuntimeState *state, int64_t constant_index,
                          void *indices, void *output, int64_t axis,
                          int64_t data_num_elements,
                          int64_t indices_num_elements,
                          int64_t output_num_elements, int64_t axis_size,
                          int64_t inner_size, int64_t element_size_bytes,
                          int64_t indices_element_size_bytes) {
  OP_PROFILE(
      "gather_from_file",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "n=%lld", (long long)indices_num_elements);
        return std::string(b);
      },
      state);
  if (!state || !output || axis != 0 ||
      data_num_elements != axis_size * inner_size ||
      output_num_elements != indices_num_elements * inner_size) {
    fprintf(stderr,
            "[REAL] wrap_gather_from_file: constant %lld does not match an "
            "axis-0 gather of a disk-resident table\n",
            (long long)constant_index);
    return -1;
  }
  return hipdnn_ep_disk_constant_gather_rows(
      state, constant_index, indices, indices_num_elements,
      indices_element_size_bytes, axis_size, inner_size * element_size_bytes,
      output);
}
