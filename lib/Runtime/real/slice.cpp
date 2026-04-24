/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>

// ONNX Slice (opset 13+) -> hip_slice custom kernel.
//
// `axes` and `steps` are optional ONNX operands: when they are absent the
// lowering passes nullptr and the kernel falls back to axes=[0..rank) and
// steps=all-1. All tensor pointers other than `state`, `input_shape` and
// `output_shape` point into device memory; `input_shape` and `output_shape`
// are host arrays populated by the HipToLLVM lowering from the static
// memref dims.
int wrap_slice(RuntimeState *state, const void *data, const void *starts,
               const void *ends, const void *axes, const void *steps,
               void *output, const int64_t *input_shape,
               const int64_t *output_shape, int64_t rank,
               int64_t num_slice_entries, int64_t output_num_elements,
               int64_t element_size_bytes, int64_t data_type) {
  if (!state || !data || !starts || !ends || !output || !input_shape ||
      !output_shape) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_slice: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_slice: rank=%lld, slice_entries=%lld, output_num=%lld, "
      "elem_size=%lld, dtype=%lld, axes=%s, steps=%s -> calling hip_slice\n",
      (long long)rank, (long long)num_slice_entries,
      (long long)output_num_elements, (long long)element_size_bytes,
      (long long)data_type, axes ? "yes" : "no", steps ? "yes" : "no");

  return hip_slice(stream, data, starts, ends, axes, steps, output, input_shape,
                   output_shape, rank, num_slice_entries, output_num_elements,
                   static_cast<int>(element_size_bytes));
}
