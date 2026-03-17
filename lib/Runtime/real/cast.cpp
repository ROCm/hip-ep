/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"
#include "../debug_log.h"
#include "runtime_types.h"
#include "hip_custom_kernels.h"

#include <cstdio>

// ONNX TensorProto element type enum values used by the `to` parameter
static constexpr int64_t ONNX_FLOAT = 1;
static constexpr int64_t ONNX_INT32 = 6;
static constexpr int64_t ONNX_INT64 = 7;
static constexpr int64_t ONNX_FLOAT16 = 10;

static int onnx_type_to_hip_dtype(int64_t onnx_type) {
  switch (onnx_type) {
    case ONNX_FLOAT:   return hip_DTYPE_FLOAT32;
    case ONNX_INT32:   return hip_DTYPE_INT32;
    case ONNX_INT64:   return hip_DTYPE_INT64;
    case ONNX_FLOAT16: return hip_DTYPE_FLOAT16;
    default:           return -1;
  }
}

static int element_size_to_hip_dtype(int64_t elem_size) {
  switch (elem_size) {
    case 8: return hip_DTYPE_INT64;
    case 4: return hip_DTYPE_INT32;
    case 2: return hip_DTYPE_FLOAT16;
    default: return -1;
  }
}

int wrap_cast(RuntimeState* state, void* input, void* output,
              int64_t num_elements, int64_t input_element_size,
              int64_t output_element_size, int64_t to) {
  if (!state || !input || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_cast: null argument\n");
    return -1;
  }

  void* stream = hipdnn_ep_state_get_stream(state);

  int output_dtype = onnx_type_to_hip_dtype(to);
  if (output_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_cast: unsupported ONNX output type to=%lld\n",
            (long long)to);
    return -1;
  }

  int input_dtype = element_size_to_hip_dtype(input_element_size);
  if (input_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_cast: unsupported input element_size=%lld\n",
            (long long)input_element_size);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_cast: num_elements=%lld, input_size=%lld, "
                    "output_size=%lld, to=%lld, input_dtype=%d, output_dtype=%d\n",
                    (long long)num_elements, (long long)input_element_size,
                    (long long)output_element_size, (long long)to,
                    input_dtype, output_dtype);

  return hip_cast(stream, input, output, num_elements,
                   input_dtype, output_dtype);
}
