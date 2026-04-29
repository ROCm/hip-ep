/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Runtime wrapper for hip_pad (ONNX Pad opset 18).

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>
#include <cstdint>

static int hipdnn_ep_to_hip_dtype_pad(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  default:
    return -1;
  }
}

extern "C" int wrap_pad(RuntimeState *state, void *input, void *output,
                        const int64_t *in_shape,
                        const int64_t *in_strides_elems,
                        const int64_t *out_shape,
                        const int64_t *out_strides_elems, int64_t rank,
                        const int64_t *pads_begin, int64_t pads_begin_len,
                        int64_t data_type, int64_t mode, float value) {
  if (!state || !input || !output || !in_shape || !in_strides_elems ||
      !out_shape || !pads_begin) {
    fprintf(stderr, "wrap_pad: null argument\n");
    return -1;
  }

  int hip_dtype = hipdnn_ep_to_hip_dtype_pad(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr, "wrap_pad: unsupported data_type %lld (%s)\n",
            (long long)data_type, hipdnn_ep_datatype_name(data_type));
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_pad: dtype=%s, rank=%lld, mode=%lld, value=%g\n",
      hipdnn_ep_datatype_name(data_type), (long long)rank, (long long)mode,
      (double)value);

  int64_t num_out = 1;
  for (int64_t d = 0; d < rank; ++d)
    num_out *= out_shape[d];
  int rc = hip_pad(stream, input, output, in_shape, in_strides_elems, out_shape,
                   out_strides_elems, rank, pads_begin, pads_begin_len, hip_dtype,
                   static_cast<int>(mode), value);
  return rc;
}
