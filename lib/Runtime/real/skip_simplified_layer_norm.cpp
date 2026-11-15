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

  int rc = hip_skip_rms_norm(hipdnn_ep_state_get_stream(state), input, skip,
                             gamma, bias, output, input_skip_bias_sum, num_rows,
                             hidden_dim, epsilon, hip_dtype);
  if (rc == 0 && num_rows > 0 && input_num_elements % hidden_dim == 0 &&
      _opProf && op_profile_call_capture_enabled(_opProf->ps)) {
    using Json = CallCaptureJson;
    const char *dtype = element_size_bytes == 2 ? "float16" : "float32";
    auto tensor = [dtype, num_rows, hidden_dim](int slot, const char *role,
                                                const void *pointer,
                                                bool vector = false) {
      return Json::object(
          {{"slot", slot},
           {"role", role},
           {"present", pointer != nullptr},
           {"dtype", pointer ? dtype : nullptr},
           {"shape", pointer
                         ? (vector ? Json::dimensions({hidden_dim})
                                   : Json::dimensions({num_rows, hidden_dim}))
                         : Json(nullptr)},
           {"layout",
            pointer ? Json::object({{"contiguous", true},
                                    {"view", "production_abi_view"},
                                    {"order", vector ? "flat" : "row_major"}})
                    : Json(nullptr)}});
    };
    _opProf->capture = op_profile_capture_call(
        _opProf->ps,
        Json::object(
            {{"op_type", "SkipSimplifiedLayerNormalization"},
             {"execution",
              Json::object(
                  {{"backend", "HIP-EP"},
                   {"entry", "wrap_skip_simplified_layer_norm"},
                   {"variant", "fused_skip_rms"},
                   {"config",
                    Json::object({{"hip_dtype", hip_dtype},
                                  {"kernel_entry", "hip_skip_rms_norm"},
                                  {"accumulation_dtype", "float32"},
                                  {"internal_config", nullptr},
                                  {"internal_config_missing",
                                   "not_exposed_at_wrapper_abi"}})}})},
             {"inputs",
              Json::array({tensor(0, "input", input), tensor(1, "skip", skip),
                           tensor(2, "gamma", gamma, true),
                           tensor(3, "bias", bias, true)})},
             {"outputs", Json::array({tensor(0, "output", output),
                                      tensor(1, "input_skip_bias_sum",
                                             input_skip_bias_sum)})},
             {"attributes", Json::object({{"epsilon", epsilon}})},
             {"parameters",
              Json::object({{"input_num_elements", input_num_elements},
                            {"gamma_num_elements", gamma_num_elements},
                            {"element_size_bytes", element_size_bytes},
                            {"num_rows", num_rows},
                            {"hidden_dim", hidden_dim},
                            {"epsilon", epsilon},
                            {"has_bias", bias != nullptr},
                            {"has_sum_output", input_skip_bias_sum != nullptr},
                            {"original_logical_rank", nullptr}})}}));
  }
  return rc;
}

static int floatDtypeForSize(int64_t element_size_bytes) {
  if (element_size_bytes == 2)
    return HIP_DTYPE_FLOAT16;
  if (element_size_bytes == 4)
    return HIP_DTYPE_FLOAT32;
  return -1;
}

int wrap_skip_simplified_layer_norm_mixed(
    RuntimeState *state, void *input, void *skip, void *gamma, void *bias,
    void *output, void *input_skip_bias_sum, int64_t input_num_elements,
    int64_t gamma_num_elements, int64_t input_element_size,
    int64_t skip_element_size, int64_t gamma_element_size,
    int64_t output_element_size, int64_t sum_element_size, float epsilon) {
  OP_PROFILE(
      "skip_layernorm",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lldx%lld,mixed",
                 (long long)(gamma_num_elements > 0
                                 ? input_num_elements / gamma_num_elements
                                 : 0),
                 (long long)gamma_num_elements);
        return std::string(b);
      },
      state);

  if (!state || !input || !skip || !gamma || !output) {
    fprintf(stderr,
            "wrap_skip_simplified_layer_norm_mixed: null required argument\n");
    return -1;
  }
  if (gamma_num_elements <= 0) {
    fprintf(stderr,
            "wrap_skip_simplified_layer_norm_mixed: gamma_num_elements=%lld\n",
            (long long)gamma_num_elements);
    return -1;
  }

  const int in_dt = floatDtypeForSize(input_element_size);
  const int skip_dt = floatDtypeForSize(skip_element_size);
  const int gamma_dt = floatDtypeForSize(gamma_element_size);
  const int out_dt = floatDtypeForSize(output_element_size);
  const int sum_dt =
      input_skip_bias_sum ? floatDtypeForSize(sum_element_size) : in_dt;
  if (in_dt < 0 || skip_dt < 0 || gamma_dt < 0 || out_dt < 0 || sum_dt < 0) {
    fprintf(stderr,
            "wrap_skip_simplified_layer_norm_mixed: unsupported element sizes "
            "in/skip/gamma/out/sum=%lld/%lld/%lld/%lld/%lld\n",
            (long long)input_element_size, (long long)skip_element_size,
            (long long)gamma_element_size, (long long)output_element_size,
            (long long)sum_element_size);
    return -1;
  }

  int64_t hidden_dim = gamma_num_elements;
  int64_t num_rows = input_num_elements / hidden_dim;

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_skip_simplified_layer_norm_mixed: num_rows=%lld, "
      "hidden_dim=%lld, elem bytes in/skip/gamma/out/sum=%lld/%lld/%lld/%lld/"
      "%lld, bias=%s, input_skip_bias_sum=%s\n",
      (long long)num_rows, (long long)hidden_dim, (long long)input_element_size,
      (long long)skip_element_size, (long long)gamma_element_size,
      (long long)output_element_size, (long long)sum_element_size,
      bias ? "yes" : "no", input_skip_bias_sum ? "yes" : "no");

  return hip_skip_rms_norm_mixed(hipdnn_ep_state_get_stream(state), input, skip,
                                 gamma, bias, output, input_skip_bias_sum,
                                 num_rows, hidden_dim, epsilon, in_dt, skip_dt,
                                 gamma_dt, out_dt, sum_dt);
}
