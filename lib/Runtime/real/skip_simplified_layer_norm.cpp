/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"

#include <cstdio>

//===----------------------------------------------------------------------===//
// SkipSimplifiedLayerNormalization via fused HIP custom kernel
//===----------------------------------------------------------------------===//
//
// ONNX SkipSimplifiedLayerNormalization (com.microsoft):
//   input_skip_bias_sum = input + skip [+ bias]
//   output              = RMSNorm(input_skip_bias_sum) * gamma
//
// Implementation: dispatches to hip_skip_simplified_layer_norm, which fuses
// the add and RMS-normalize steps into a single GPU kernel (one block per
// row, block-wide reduction of sum-of-squares). No MIOpen descriptors,
// no scratch workspace, no intermediate add buffer -- the running sum lives
// in registers and is only spilled to memory when input_skip_bias_sum is a
// requested output.
//
// Tensor layout:
//   input / skip / input_skip_bias_sum:  [num_rows, hidden_dim]
//   gamma / bias:                        [hidden_dim]
//   output:                              [num_rows, hidden_dim]
//===----------------------------------------------------------------------===//

int wrap_skip_simplified_layer_norm(RuntimeState *state, void *input,
                                    void *skip, void *gamma, void *bias,
                                    void *output, void *input_skip_bias_sum,
                                    int64_t input_num_elements,
                                    int64_t gamma_num_elements,
                                    int64_t element_size_bytes, float epsilon) {
  if (!state || !input || !skip || !gamma || !output) {
    fprintf(stderr,
            "wrap_skip_simplified_layer_norm: null required argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    fprintf(stderr, "wrap_skip_simplified_layer_norm: null stream\n");
    return -1;
  }

  int64_t hidden_dim = gamma_num_elements;
  int64_t num_rows = input_num_elements / hidden_dim;

  int hip_dtype;
  const char *type_name;
  if (element_size_bytes == 2) {
    hip_dtype = HIP_DTYPE_FLOAT16;
    type_name = "f16";
  } else if (element_size_bytes == 4) {
    hip_dtype = HIP_DTYPE_FLOAT32;
    type_name = "f32";
  } else {
    fprintf(stderr,
            "wrap_skip_simplified_layer_norm: unsupported element_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_skip_simplified_layer_norm: num_rows=%lld, "
                    "hidden_dim=%lld, data_type=%s, epsilon=%e, "
                    "bias=%s, input_skip_bias_sum=%s -> "
                    "calling hip_skip_simplified_layer_norm\n",
                    (long long)num_rows, (long long)hidden_dim, type_name,
                    (double)epsilon, bias ? "yes" : "no",
                    input_skip_bias_sum ? "yes" : "no");

  return hip_skip_simplified_layer_norm(stream, input, skip, gamma, bias,
                                        output, input_skip_bias_sum, num_rows,
                                        hidden_dim, hip_dtype, epsilon);
}
