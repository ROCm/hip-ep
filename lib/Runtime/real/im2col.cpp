// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "op_profile.h"
#include "runtime_types.h"

#include <hip/hip_runtime.h>
#include <string>

#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t error = (cmd);                                                  \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,          \
              hipGetErrorString(error));                                       \
      return -1;                                                               \
    }                                                                          \
  } while (0)

static int hipdnn_to_hip_dtype(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  default:
    return -1;
  }
}

int wrap_im2d2col(RuntimeState *state, const void *input, int64_t data_type,
                  int64_t C, int64_t H, int64_t W, int64_t kh, int64_t kw,
                  int64_t pad_top, int64_t pad_bottom, int64_t pad_left,
                  int64_t pad_right, int64_t stride_h, int64_t stride_w,
                  int64_t dilation_h, int64_t dilation_w, void *output,
                  int64_t out_h, int64_t out_w) {
  OP_PROFILE(
      "im2d2col",
      [&] {
        char b[80];
        const char *dt = (data_type == HIPDNN_EP_DATATYPE_HALF)       ? "f16"
                         : (data_type == HIPDNN_EP_DATATYPE_BFLOAT16) ? "bf16"
                                                                      : "f32";
        snprintf(b, sizeof(b), "%lldx%lldx%lld,k=%lldx%lld,%s",
                 (long long)C, (long long)H, (long long)W,
                 (long long)kh, (long long)kw, dt);
        return std::string(b);
      },
      state);

  if (!state || !input || !output) {
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  return hip_im2d2col(stream, input, data_type, C, H, W, kh, kw, pad_top,
                      pad_bottom, pad_left, pad_right, stride_h, stride_w,
                      output, out_h, out_w);
}
