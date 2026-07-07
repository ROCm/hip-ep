/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "runtime_types.h"

#include "hip_custom_kernels.h"

static int hipdnn_ep_to_hip_dtype_bias_gelu(int64_t data_type) {
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

// Fused com.microsoft.BiasGelu: output = Gelu_erf(data + broadcast(bias)).
// bias_len is the length of the 1D bias vector (last dim of data).
HIPDNN_EP_RT_EXPORT int wrap_bias_gelu(RuntimeState *state, void *data,
                                       void *bias, void *output,
                                       int64_t num_elements, int64_t bias_len,
                                       int64_t data_type) {
  OP_PROFILE(
      "bias_gelu",
      [&] {
        char b[96];
        snprintf(b, sizeof(b), "n=%lld,bias=%lld", (long long)num_elements,
                 (long long)bias_len);
        return std::string(b);
      },
      state);

  if (!state || !data || !bias || !output) {
    fprintf(stderr, "[REAL] wrap_bias_gelu: null argument\n");
    return -1;
  }
  if (num_elements <= 0 || bias_len <= 0) {
    fprintf(stderr,
            "[REAL] wrap_bias_gelu: invalid shape num_elements=%lld "
            "bias_len=%lld\n",
            (long long)num_elements, (long long)bias_len);
    return -1;
  }
  if (num_elements % bias_len != 0) {
    fprintf(stderr,
            "[REAL] wrap_bias_gelu: num_elements (%lld) must be a "
            "multiple of bias_len (%lld)\n",
            (long long)num_elements, (long long)bias_len);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  int hip_dtype = hipdnn_ep_to_hip_dtype_bias_gelu(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr, "[REAL] wrap_bias_gelu: unsupported data_type %lld\n",
            (long long)data_type);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_bias_gelu: num_elements=%lld, bias_len=%lld, "
                    "data_type=%s(%lld)\n",
                    (long long)num_elements, (long long)bias_len,
                    hipdnn_ep_datatype_name(data_type), (long long)data_type);

  int result = hip_bias_gelu(stream, data, bias, output, num_elements, bias_len,
                             hip_dtype);
  if (result != 0) {
    fprintf(stderr, "[REAL] wrap_bias_gelu: kernel launch failed (%d)\n",
            result);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_bias_gelu: completed successfully\n");
  return 0;
}
