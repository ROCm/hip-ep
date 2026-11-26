/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// LayerNormalization (ONNX-17): full LN with optional bias / mean /
// inv_std outputs.
//
//   y = (x - mean) * rsqrt(var + epsilon) * scale + bias
//
// Row-per-group HIP kernel with a block-per-row fallback (see
// layer_norm_kernel.hip). Internal math is FP32 regardless of I/O dtype. The
// optional `mean` and `inv_std` outputs are written in the dtype implied by
// `stash_type` (the ONNX attribute, raw TensorProto enum).
//
// Source: onnxruntime/core/providers/cuda/nn/layer_norm_impl.cu @ v1.22.2
//         (small-N branch -- block-per-row two-pass reduction).

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>

// ONNX TensorProto.DataType enum values for stash_type. The lowering
// passes these through unchanged (NOT mapped through getHipdnnDataType).
static constexpr int64_t kOnnxStashFloat = 1; // FP32 (ONNX default)
static constexpr int64_t kOnnxStashFloat16 = 10;

int wrap_layer_normalization(RuntimeState *state, void *input, void *scale,
                             void *bias, void *output, void *mean,
                             void *inv_std, int64_t input_num_elements,
                             int64_t scale_num_elements,
                             int64_t element_size_bytes, int64_t axis,
                             float epsilon, int64_t stash_type) {
  OP_PROFILE(
      "layernorm",
      [&] {
        char b[64];
        int64_t outer = scale_num_elements > 0
                            ? input_num_elements / scale_num_elements
                            : 0;
        snprintf(b, sizeof(b), "%lldx%lld:%s%s%s", (long long)outer,
                 (long long)scale_num_elements,
                 (element_size_bytes == 2) ? "f16" : "f32", bias ? ":b" : "",
                 (mean || inv_std) ? ":stats" : "");
        return std::string(b);
      },
      state);

  (void)axis;

  if (!state || !input || !scale || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_layer_normalization: null required arg\n");
    return -1;
  }
  if (scale_num_elements <= 0) {
    fprintf(stderr,
            "[REAL] wrap_layer_normalization: scale_num_elements=%lld\n",
            (long long)scale_num_elements);
    return -1;
  }
  if (input_num_elements % scale_num_elements != 0) {
    fprintf(stderr,
            "[REAL] wrap_layer_normalization: input_num(%lld) not divisible "
            "by scale_num(%lld)\n",
            (long long)input_num_elements, (long long)scale_num_elements);
    return -1;
  }

  int hip_dtype;
  if (element_size_bytes == 2) {
    hip_dtype = HIP_DTYPE_FLOAT16;
  } else if (element_size_bytes == 4) {
    hip_dtype = HIP_DTYPE_FLOAT32;
  } else {
    fprintf(stderr,
            "[REAL] wrap_layer_normalization: unsupported element_size=%lld "
            "(supported: 2=fp16, 4=fp32)\n",
            (long long)element_size_bytes);
    return -1;
  }

  // stash_type uses raw ONNX TensorProto.DataType. Map to our hip_dtype
  // enum. Per ONNX spec, default is FLOAT (1).
  int mean_dtype;
  if (stash_type == kOnnxStashFloat || stash_type == 0) {
    mean_dtype = HIP_DTYPE_FLOAT32;
  } else if (stash_type == kOnnxStashFloat16) {
    mean_dtype = HIP_DTYPE_FLOAT16;
  } else {
    fprintf(stderr,
            "[REAL] wrap_layer_normalization: unsupported stash_type=%lld "
            "(supported: 1=fp32, 10=fp16)\n",
            (long long)stash_type);
    return -1;
  }

  int64_t outer = input_num_elements / scale_num_elements;
  int64_t norm_size = scale_num_elements;

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_layer_normalization: outer=%lld, norm_size=%lld, "
      "elem_size=%lld, stash=%lld, eps=%e, bias=%s, mean=%s, inv_std=%s\n",
      (long long)outer, (long long)norm_size, (long long)element_size_bytes,
      (long long)stash_type, (double)epsilon, bias ? "yes" : "null",
      mean ? "yes" : "null", inv_std ? "yes" : "null");

  return hip_layer_norm(stream, input, scale, bias, output, mean, inv_std,
                        outer, norm_size, epsilon, hip_dtype, mean_dtype);
}

// Residual add fused with LayerNormalization over the last axis:
//   sum = input + skip [+ bias];  y = LN(sum) * gamma [+ beta]
// `input_skip_bias_sum` receives the sum when non-null. A skip with fewer
// elements than input holds the trailing rows and repeats over the input.
int wrap_skip_layer_norm(RuntimeState *state, void *input, void *skip,
                         void *gamma, void *beta, void *bias, void *output,
                         void *input_skip_bias_sum, int64_t input_num_elements,
                         int64_t skip_num_elements, int64_t gamma_num_elements,
                         int64_t element_size_bytes, float epsilon,
                         int32_t round_sum) {
  OP_PROFILE(
      "skip_layernorm",
      [&] {
        char b[64];
        int64_t outer = gamma_num_elements > 0
                            ? input_num_elements / gamma_num_elements
                            : 0;
        snprintf(b, sizeof(b), "%lldx%lld:%s%s%s", (long long)outer,
                 (long long)gamma_num_elements,
                 (element_size_bytes == 2) ? "f16" : "f32", bias ? ":b" : "",
                 input_skip_bias_sum ? ":sum" : "");
        return std::string(b);
      },
      state);

  if (!state || !input || !skip || !gamma || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_skip_layer_norm: null required arg\n");
    return -1;
  }
  if (gamma_num_elements <= 0 || input_num_elements % gamma_num_elements != 0) {
    fprintf(stderr,
            "[REAL] wrap_skip_layer_norm: input_num(%lld) not a multiple of "
            "gamma_num(%lld)\n",
            (long long)input_num_elements, (long long)gamma_num_elements);
    return -1;
  }
  int hip_dtype;
  if (element_size_bytes == 2) {
    hip_dtype = HIP_DTYPE_FLOAT16;
  } else if (element_size_bytes == 4) {
    hip_dtype = HIP_DTYPE_FLOAT32;
  } else {
    fprintf(stderr,
            "[REAL] wrap_skip_layer_norm: unsupported element_size=%lld "
            "(supported: 2=fp16, 4=fp32)\n",
            (long long)element_size_bytes);
    return -1;
  }

  if (input_num_elements == 0)
    return 0;
  if (skip_num_elements <= 0 || skip_num_elements % gamma_num_elements != 0 ||
      input_num_elements % skip_num_elements != 0) {
    fprintf(stderr,
            "[REAL] wrap_skip_layer_norm: skip_num(%lld) does not tile "
            "input_num(%lld) in rows of %lld\n",
            (long long)skip_num_elements, (long long)input_num_elements,
            (long long)gamma_num_elements);
    return -1;
  }

  int64_t outer = input_num_elements / gamma_num_elements;
  int64_t skip_rows = skip_num_elements / gamma_num_elements;
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_skip_layer_norm: outer=%lld, skip_rows=%lld, "
      "norm_size=%lld, elem_size=%lld, eps=%e, beta=%s, bias=%s, "
      "sum=%s\n",
      (long long)outer, (long long)skip_rows, (long long)gamma_num_elements,
      (long long)element_size_bytes, (double)epsilon, beta ? "yes" : "null",
      bias ? "yes" : "null", input_skip_bias_sum ? "yes" : "null");

  void *stream = hipdnn_ep_state_get_stream(state);
  return hip_skip_layer_norm(stream, input, skip, gamma, beta, bias, output,
                             input_skip_bias_sum, outer, skip_rows,
                             gamma_num_elements, epsilon, round_sum, hip_dtype);
}
