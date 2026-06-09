/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>

// ONNX NonZero (GPU). Fills the over-allocated output buffer [R, capacity]
// (capacity == input_num_elements worst case) with the row-major indices of
// the non-zero input elements, in increasing flat-index order. The kernel
// zero-fills the buffer first so the undefined tail [N, capacity) is
// deterministic; N is not separately reported (see hipdnn_ep_runtime.h).
//
// The input_data_type (HIPDNN_EP_DATATYPE_*) is decomposed into
// (element_size_bytes, is_float), which fully determines the non-zero
// predicate the kernel needs -- this keeps the .hip kernel decoupled from the
// runtime's dtype enum.
int wrap_nonzero(RuntimeState *state, void *input, void *output,
                 int64_t input_num_elements, int64_t input_rank,
                 int64_t output_capacity, int64_t input_data_type,
                 const int64_t *input_shape) {
  OP_PROFILE(
      "nonzero",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "n=%lld rank=%lld",
                 (long long)input_num_elements, (long long)input_rank);
        return std::string(b);
      },
      state);

  if (!state || !input || !output || !input_shape) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_nonzero: null argument\n");
    return -1;
  }
  if (input_rank <= 0) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_nonzero: invalid rank=%lld\n",
                      (long long)input_rank);
    return -1;
  }

  int element_size_bytes = 0;
  int is_float = 0;
  switch (input_data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    element_size_bytes = 4;
    is_float = 1;
    break;
  case HIPDNN_EP_DATATYPE_HALF:
    element_size_bytes = 2;
    is_float = 1;
    break;
  case HIPDNN_EP_DATATYPE_INT32:
    element_size_bytes = 4;
    break;
  case HIPDNN_EP_DATATYPE_INT64:
    element_size_bytes = 8;
    break;
  case HIPDNN_EP_DATATYPE_INT8:  // i1 / signed i8
  case HIPDNN_EP_DATATYPE_UINT8: // ui8 (ORT bool)
    element_size_bytes = 1;
    break;
  default:
    RUNTIME_DEBUG_LOG("[REAL] wrap_nonzero: unsupported input_data_type=%lld\n",
                      (long long)input_data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_nonzero: num_elements=%lld, rank=%lld, capacity=%lld, "
      "elem_size=%d, is_float=%d\n",
      (long long)input_num_elements, (long long)input_rank,
      (long long)output_capacity, element_size_bytes, is_float);

  return hip_nonzero(stream, input, output, input_shape, input_rank,
                     input_num_elements, output_capacity, element_size_bytes,
                     is_float);
}
