/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Expand: copy `input` to `output`, broadcasting any size-1 input dim. ONNX
// Expand uses numpy-style right-aligned broadcasting; the kernel handles
// the alignment internally.
//
// Source: onnxruntime/core/providers/cuda/tensor/expand.cu @ v1.22.2.
//
// HIP conversion calls the checked adapter below. Both entry points pass the
// destination descriptor to the kernel for an independent safety check. The
// original GPU `shape` pointer is unused.
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
  // Equal→Expand on embedding masks uses ui8; broadcast is bitwise-identical
  // to i8 (no arithmetic), so reuse the int8 kernel instantiation.
  case HIPDNN_EP_DATATYPE_UINT8:
    return HIP_DTYPE_INT8;
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

  if (!state) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_expand: null state\n");
    return -1;
  }
  if (input_rank < 0 || output_rank < 0) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_expand: negative rank\n");
    return -1;
  }
  if ((input_rank > 0 && !input_shape) || (output_rank > 0 && !output_shape)) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_expand: missing shape descriptor\n");
    return -1;
  }

  int hip_dtype = expand_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_expand: unsupported data_type=%s(%lld) "
            "(supported: f16, f32, i32, i64, u8)\n",
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
int wrap_expand_checked(RuntimeState *state, void *input, void *shape,
                        void *output, const int64_t *input_shape,
                        int64_t input_rank, const int64_t *output_shape,
                        int64_t output_rank, int64_t shape_valid,
                        int64_t data_type) {
  if (!shape_valid) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_expand_checked: invalid broadcast shape\n");
    return -1;
  }
  return wrap_expand(state, input, shape, output, input_shape, input_rank,
                     output_shape, output_rank, data_type);
}
