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

// Map HIPDNN_EP_DATATYPE_* to hip_dtype_t for elementwise HIP kernels
// (reciprocal, sqrt). Values match hip_dtype_t in hip_custom_kernels.h.
static int hipdnn_ep_to_hip_dtype_elementwise_unary(int64_t data_type) {
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

// HipToLLVM lowers hip.reciprocal and hip.sqrt to @wrap_power with
// (alpha, beta, gamma):
// - Reciprocal (0, 1, -1): ONNX 1/x via hip_elementwise_reciprocal.
// - Sqrt (0, 1, 0.5): ONNX sqrt via hip_elementwise_sqrt (negative → NaN).
// - Other (alpha, beta, gamma): unsupported.

namespace {

int launchReciprocalHip(RuntimeState *state, void *input, void *output,
                        int64_t num_elements, int64_t data_type) {
  if (!state || !input || !output) {
    fprintf(stderr, "launchReciprocalHip: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  int hip_dtype = hipdnn_ep_to_hip_dtype_elementwise_unary(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_power (reciprocal HIP): unsupported data_type %lld "
            "(%s)\n",
            (long long)data_type, hipdnn_ep_datatype_name(data_type));
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_power (reciprocal HIP 1/x): num_elements=%lld, dtype=%s\n",
      (long long)num_elements, hipdnn_ep_datatype_name(data_type));

  return hip_elementwise_reciprocal(stream, input, output, num_elements,
                                    hip_dtype);
}

int launchSqrtHip(RuntimeState *state, void *input, void *output,
                  int64_t num_elements, int64_t data_type) {
  if (!state || !input || !output) {
    fprintf(stderr, "launchSqrtHip: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  int hip_dtype = hipdnn_ep_to_hip_dtype_elementwise_unary(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_power (sqrt HIP): unsupported data_type %lld (%s)\n",
            (long long)data_type, hipdnn_ep_datatype_name(data_type));
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_power (sqrt HIP): num_elements=%lld, dtype=%s\n",
      (long long)num_elements, hipdnn_ep_datatype_name(data_type));

  return hip_elementwise_sqrt(stream, input, output, num_elements, hip_dtype);
}

} // namespace

int wrap_power(RuntimeState *state, void *input, void *output,
               int64_t num_elements, int64_t data_type, double alpha,
               double beta, double gamma) {
  OP_PROFILE(
      "power",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "n=%lld", (long long)num_elements);
        return std::string(b);
      },
      state);
  if (!state || !input || !output) {
    fprintf(stderr, "wrap_power: null argument\n");
    return -1;
  }

  // hip.reciprocal lowers to wrap_power(…, 0, 1, -1).
  if (alpha == 0.0 && beta == 1.0 && gamma == -1.0)
    return launchReciprocalHip(state, input, output, num_elements, data_type);

  // hip.sqrt lowers to wrap_power(…, 0, 1, 0.5).
  if (alpha == 0.0 && beta == 1.0 && gamma == 0.5)
    return launchSqrtHip(state, input, output, num_elements, data_type);

  fprintf(stderr,
          "wrap_power: unsupported (alpha=%.2f, beta=%.2f, gamma=%.2f)\n",
          alpha, beta, gamma);
  std::abort();
  return -1;
}
