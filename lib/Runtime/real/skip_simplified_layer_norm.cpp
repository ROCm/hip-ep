/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// SkipSimplifiedLayerNormalization (com.microsoft), full MS spec:
//
//   input_skip_bias_sum = input + skip [+ bias]
//   output              = RMSNorm(input_skip_bias_sum) * gamma
//
// The residual add, the optional bias add, and the norm are fused into a
// single block-per-row HIP kernel (see skip_rms_norm_kernel.hip) with FP32
// internal math. Because the kernel can recompute the sum in its second pass,
// it needs no scratch workspace when input_skip_bias_sum is not requested.
//
// Tensor layout:
//   input / skip / input_skip_bias_sum:  [num_rows, hidden_dim]
//   gamma / bias:                        [hidden_dim], broadcast across rows
//   output:                              [num_rows, hidden_dim]

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>

int wrap_skip_simplified_layer_norm(RuntimeState *state, void *input,
                                    void *skip, void *gamma, void *bias,
                                    void *output, void *input_skip_bias_sum,
                                    int64_t input_num_elements,
                                    int64_t gamma_num_elements,
                                    int64_t element_size_bytes, float epsilon) {
  OP_PROFILE(
      "skip_layernorm",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lldx%lld",
                 (long long)(gamma_num_elements > 0
                                 ? input_num_elements / gamma_num_elements
                                 : 0),
                 (long long)gamma_num_elements);
        return std::string(b);
      },
      state);

  if (!state || !input || !skip || !gamma || !output) {
    fprintf(stderr,
            "wrap_skip_simplified_layer_norm: null required argument\n");
    return -1;
  }
  if (gamma_num_elements <= 0) {
    fprintf(stderr,
            "wrap_skip_simplified_layer_norm: gamma_num_elements=%lld\n",
            (long long)gamma_num_elements);
    return -1;
  }

  int hip_dtype;
  if (element_size_bytes == 2)
    hip_dtype = HIP_DTYPE_FLOAT16;
  else if (element_size_bytes == 4)
    hip_dtype = HIP_DTYPE_FLOAT32;
  else {
    fprintf(stderr,
            "wrap_skip_simplified_layer_norm: unsupported element_size %lld "
            "(supported: 2=fp16, 4=fp32)\n",
            (long long)element_size_bytes);
    return -1;
  }

  int64_t hidden_dim = gamma_num_elements;
  int64_t num_rows = input_num_elements / hidden_dim;

  RUNTIME_DEBUG_LOG("[REAL] wrap_skip_simplified_layer_norm: num_rows=%lld, "
                    "hidden_dim=%lld, data_type=%s, epsilon=%e, "
                    "bias=%s, input_skip_bias_sum=%s\n",
                    (long long)num_rows, (long long)hidden_dim,
                    (element_size_bytes == 2) ? "f16" : "f32", (double)epsilon,
                    bias ? "yes" : "no", input_skip_bias_sum ? "yes" : "no");

  return hip_skip_rms_norm(hipdnn_ep_state_get_stream(state), input, skip,
                           gamma, bias, output, input_skip_bias_sum, num_rows,
                           hidden_dim, epsilon, hip_dtype);
}
