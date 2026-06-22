/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "error_check_macros.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>

//===----------------------------------------------------------------------===//
// Resize — custom HIP kernel
//===----------------------------------------------------------------------===//
//
// Lowering signature (matches ResizeLowering.cpp):
//   wrap_resize(state, input, output,
//               data_type, spatial_rank,
//               N, C, in0..2, out0..2,
//               mode, coord_transform, nearest_mode)
//
// `mode`:           0 = nearest, 1 = linear (N-linear)
// `coord_transform`: 0 = half_pixel, 1 = asymmetric, 2 = align_corners
// `nearest_mode`:    0 = round_prefer_floor (only used when mode=nearest)

static int hipdnn_ep_to_hip_dtype(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  case HIPDNN_EP_DATATYPE_DOUBLE:
    return HIP_DTYPE_FLOAT64;
  default:
    return -1;
  }
}

int wrap_resize(RuntimeState *state, void *input, void *output,
                int64_t data_type, int64_t spatial_rank, int64_t N, int64_t C,
                int64_t in0, int64_t in1, int64_t in2, int64_t out0,
                int64_t out1, int64_t out2, int64_t mode,
                int64_t coord_transform, int64_t nearest_mode) {
  OP_PROFILE(
      "resize",
      [&] {
        char b[160];
        snprintf(b, sizeof(b),
                 "rank=%lld,in=[%lld,%lld,%lld],out=[%lld,%lld,%lld],mode=%lld",
                 (long long)spatial_rank, (long long)in0, (long long)in1,
                 (long long)in2, (long long)out0, (long long)out1,
                 (long long)out2, (long long)mode);
        return std::string(b);
      },
      state);

  if (!state || !input || !output) {
    hipdnn_ep_log_emit("[REAL] wrap_resize: null argument\n");
    return -1;
  }
  int hip_dtype = hipdnn_ep_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    hipdnn_ep_log_emit("[REAL] wrap_resize: unsupported data_type %lld\n",
                       (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_resize: dtype=%s(%lld) rank=%lld N=%lld C=%lld "
      "in=[%lld,%lld,%lld] out=[%lld,%lld,%lld] mode=%lld "
      "ct=%lld nm=%lld\n",
      hipdnn_ep_datatype_name(data_type), (long long)data_type,
      (long long)spatial_rank, (long long)N, (long long)C, (long long)in0,
      (long long)in1, (long long)in2, (long long)out0, (long long)out1,
      (long long)out2, (long long)mode, (long long)coord_transform,
      (long long)nearest_mode);

  int rc = hip_resize(
      stream, input, output, hip_dtype, static_cast<int>(spatial_rank), N, C,
      in0, in1, in2, out0, out1, out2, static_cast<int>(mode),
      static_cast<int>(coord_transform), static_cast<int>(nearest_mode));
  if (rc != 0) {
    hipdnn_ep_log_emit("[REAL] wrap_resize: kernel launch failed (%d)\n", rc);
    return -1;
  }
  RUNTIME_DEBUG_LOG("[REAL] wrap_resize: completed successfully\n");
  return 0;
}
