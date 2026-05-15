/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Expand: copy `input` to `output`, broadcasting any size-1 input dim. ONNX
// Expand uses numpy-style right-aligned broadcasting; the kernel handles
// the alignment internally so this wrapper just forwards shapes and ptrs.
//
// Source: onnxruntime/core/providers/cuda/tensor/expand.cu @ v1.22.2.
//
// The GPU `shape` tensor pointer is not used here -- the lowering already
// gives us host-side output_shape, which is the broadcast result the shape
// tensor would have produced. Avoids a per-call D2H of the shape input.
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

static int expand_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_INT32:
    return HIP_DTYPE_INT32;
  case HIPDNN_EP_DATATYPE_INT64:
    return HIP_DTYPE_INT64;
  default:
    return -1;
  }
}

int wrap_expand(RuntimeState *state, void *input, void *shape, void *output,
                const int64_t *input_shape, int64_t input_rank,
                const int64_t *output_shape, int64_t output_rank,
                int64_t data_type) {
  OP_PROFILE(
      "expand",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lld->%lld:%s", (long long)input_rank,
                 (long long)output_rank, hipdnn_ep_datatype_name(data_type));
        return std::string(b);
      },
      state);

  (void)shape;

  if (!state || !input || !output || !output_shape) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_expand: null argument\n");
    return -1;
  }
  if (output_rank == 0) {
    return 0;
  }
  if (input_rank > 0 && !input_shape) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_expand: input_shape null with rank>0\n");
    return -1;
  }

  int hip_dtype = expand_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_expand: unsupported data_type=%s(%lld) "
            "(supported: f16, f32, i32, i64)\n",
            hipdnn_ep_datatype_name(data_type), (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_expand: in_rank=%lld, out_rank=%lld, data_type=%s "
      "-> hip_expand\n",
      (long long)input_rank, (long long)output_rank,
      hipdnn_ep_datatype_name(data_type));

  return hip_expand(stream, input, output, input_shape,
                    static_cast<int>(input_rank), output_shape,
                    static_cast<int>(output_rank), hip_dtype);
}
