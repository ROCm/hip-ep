/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// com.microsoft GroupNorm:
//   y = gamma * (x - mean) / sqrt(var + epsilon) + beta
// Mean/var are per (N, group). activation 1 applies SiLU afterwards.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>
#include <string>

static int hipdnn_ep_to_hip_dtype_group_norm(int64_t data_type) {
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

HIPDNN_EP_RT_EXPORT int wrap_group_norm(RuntimeState *state, void *input,
                                        void *scale, void *bias, void *output,
                                        int64_t n, int64_t c, int64_t spatial,
                                        int64_t groups, int64_t channels_last,
                                        int64_t activation, int64_t data_type,
                                        float epsilon) {
  OP_PROFILE(
      "groupnorm",
      [&] {
        char b[96];
        snprintf(b, sizeof(b), "%lldx%lldx%lld g=%lld", (long long)n,
                 (long long)c, (long long)spatial, (long long)groups);
        return std::string(b);
      },
      state);

  if (!state || !input || !scale || !bias || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_group_norm: null required arg\n");
    return -1;
  }
  if (n <= 0 || c <= 0 || spatial <= 0) {
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_group_norm: empty shape n=%lld c=%lld spatial=%lld\n",
        (long long)n, (long long)c, (long long)spatial);
    return 0;
  }
  if (groups <= 0 || c % groups != 0 ||
      (channels_last != 0 && channels_last != 1) ||
      (activation != 0 && activation != 1)) {
    fprintf(stderr,
            "[REAL] wrap_group_norm: invalid groups=%lld channels_last=%lld "
            "activation=%lld c=%lld\n",
            (long long)groups, (long long)channels_last, (long long)activation,
            (long long)c);
    return -1;
  }

  int hip_dtype = hipdnn_ep_to_hip_dtype_group_norm(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr, "[REAL] wrap_group_norm: unsupported data_type=%lld\n",
            (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_group_norm: n=%lld c=%lld spatial=%lld groups=%lld "
      "channels_last=%lld activation=%lld dtype=%lld eps=%e\n",
      (long long)n, (long long)c, (long long)spatial, (long long)groups,
      (long long)channels_last, (long long)activation, (long long)data_type,
      (double)epsilon);

  return hip_group_norm(stream, input, scale, bias, output, n, c, spatial,
                        groups, static_cast<int>(channels_last),
                        static_cast<int>(activation), epsilon, hip_dtype);
}
