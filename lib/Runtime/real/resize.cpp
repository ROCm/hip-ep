/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Runtime wrapper for hip_resize (ONNX Resize opset 18).

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>
#include <cstdint>

// Local extern "C" declaration of the kernel launcher.  We mirror what
// `hip_custom_kernels.h` declares for `hip_resize` so that this
// translation unit can compile cleanly even if the shared header gets
// reverted by sibling-agent merges in flight (it has happened repeatedly
// during this branch's parallel landing).  The linker resolves to the
// canonical symbol exported by `3rd-party/custom_kernels/hip/resize_kernel.hip`.
extern "C" int hip_resize(void *stream, const void *input, void *output,
                          const int64_t *in_shape,
                          const int64_t *in_strides_elems,
                          const int64_t *out_shape,
                          const int64_t *out_strides_elems, int64_t rank,
                          int hip_dtype, int mode, int coord_xform,
                          float cubic_coeff_a);

static int hipdnn_ep_to_hip_dtype_resize(int64_t data_type) {
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

extern "C" int wrap_resize(RuntimeState *state, void *input, void *output,
                           const int64_t *in_shape,
                           const int64_t *in_strides_elems,
                           const int64_t *out_shape,
                           const int64_t *out_strides_elems, int64_t rank,
                           int64_t data_type, int64_t mode,
                           int64_t coord_xform, float cubic_coeff_a) {
  if (!state || !input || !output || !in_shape || !in_strides_elems ||
      !out_shape) {
    fprintf(stderr, "wrap_resize: null argument\n");
    return -1;
  }

  int hip_dtype = hipdnn_ep_to_hip_dtype_resize(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr, "wrap_resize: unsupported data_type %lld (%s)\n",
            (long long)data_type, hipdnn_ep_datatype_name(data_type));
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_resize: dtype=%s, rank=%lld, mode=%lld, xform=%lld, "
      "cubic_a=%g\n",
      hipdnn_ep_datatype_name(data_type), (long long)rank, (long long)mode,
      (long long)coord_xform, (double)cubic_coeff_a);

  int64_t num_out = 1;
  for (int64_t d = 0; d < rank; ++d)
    num_out *= out_shape[d];
  int rc = hip_resize(stream, input, output, in_shape, in_strides_elems,
                      out_shape, out_strides_elems, rank, hip_dtype,
                      static_cast<int>(mode), static_cast<int>(coord_xform),
                      cubic_coeff_a);
  return rc;
}
