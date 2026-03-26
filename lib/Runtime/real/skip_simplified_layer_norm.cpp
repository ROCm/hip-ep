/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Enable MIOpen beta APIs (miopenT5LayerNormForward, miopenNormMode_t)
#define MIOPEN_BETA_API

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "error_check_macros.h"
#include "runtime_types.h"

#include <cstdio>

// Use the shared MIOPEN_CHECK_GOTO macro for goto cleanup pattern
#undef MIOPEN_CHECK
#define MIOPEN_CHECK(cmd) MIOPEN_CHECK_GOTO(cmd, cleanup)

// =============================================================================
// SkipSimplifiedLayerNormalization via MIOpen (Full MS spec)
// =============================================================================
//
// ONNX SkipSimplifiedLayerNormalization (com.microsoft):
//   input_skip_bias_sum = input + skip [+ bias]
//   output              = RMSNorm(input_skip_bias_sum) * gamma
//
// MIOpen has no fused "add + T5 norm" API, so we compose calls:
//   1. miopenOpTensor(ADD):           tmp = input + skip
//   2. miopenOpTensor(ADD):           tmp = tmp + bias   (if bias != nullptr)
//   3. miopenT5LayerNormForward:      output = RMSNorm(tmp) * gamma
//
// All execute on the same GPU stream via the MIOpen handle — no host-device
// round trips.
//
// If input_skip_bias_sum is nullptr (optional output not requested), a
// temporary GPU buffer is allocated for the intermediate result and freed
// before return.
//
// Tensor layout:
//   input / skip / input_skip_bias_sum:  [num_rows, hidden_dim]
//   gamma / bias:                        [hidden_dim]
//   output:                              [num_rows, hidden_dim]
//   rstd (scratch):                      [num_rows] (f32)
// =============================================================================

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

  miopenHandle_t handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  if (!handle) {
    fprintf(stderr, "wrap_skip_simplified_layer_norm: null MIOpen handle\n");
    return -1;
  }

  int64_t hidden_dim = gamma_num_elements;
  int64_t num_rows = input_num_elements / hidden_dim;

  const char *type_name = (element_size_bytes == 2)   ? "f16"
                          : (element_size_bytes == 4) ? "f32"
                                                      : "?";
  RUNTIME_DEBUG_LOG("[REAL] wrap_skip_simplified_layer_norm: num_rows=%lld, "
                    "hidden_dim=%lld, data_type=%s, epsilon=%e, "
                    "bias=%s, input_skip_bias_sum=%s\n",
                    (long long)num_rows, (long long)hidden_dim, type_name,
                    (double)epsilon, bias ? "yes" : "no",
                    input_skip_bias_sum ? "yes" : "no");

  miopenDataType_t data_type;
  if (element_size_bytes == 2)
    data_type = miopenHalf;
  else if (element_size_bytes == 4)
    data_type = miopenFloat;
  else {
    fprintf(stderr,
            "wrap_skip_simplified_layer_norm: unsupported element_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  int result = 0;
  void *rstd_buf = nullptr;
  void *tmp_skip_buf = nullptr;
  bool owns_skip_buf = false;

  // If caller doesn't want input_skip_bias_sum, allocate a temp buffer
  void *skip_buf = input_skip_bias_sum;
  if (!skip_buf) {
    hipError_t hip_err =
        hipMalloc(&tmp_skip_buf, input_num_elements * element_size_bytes);
    if (hip_err != hipSuccess) {
      fprintf(stderr,
              "wrap_skip_simplified_layer_norm: hipMalloc tmp failed: %s\n",
              hipGetErrorString(hip_err));
      return -1;
    }
    skip_buf = tmp_skip_buf;
    owns_skip_buf = true;
  }

  // Descriptors for miopenOpTensor (ADD)
  miopenTensorDescriptor_t addADesc = nullptr;
  miopenTensorDescriptor_t addBDesc = nullptr;
  miopenTensorDescriptor_t addCDesc = nullptr;
  miopenTensorDescriptor_t biasDesc = nullptr;

  // Descriptors for miopenT5LayerNormForward
  miopenTensorDescriptor_t xDesc = nullptr;
  miopenTensorDescriptor_t weightDesc = nullptr;
  miopenTensorDescriptor_t yDesc = nullptr;
  miopenTensorDescriptor_t rstdDesc = nullptr;

  // =========================================================================
  // Step 1: Element-wise add — skip_buf = input + skip
  // =========================================================================
  RUNTIME_DEBUG_LOG("[REAL] wrap_skip_simplified_layer_norm: step 1 — "
                    "miopenOpTensor(ADD) for %lld elements\n",
                    (long long)input_num_elements);

  MIOPEN_CHECK(miopenCreateTensorDescriptor(&addADesc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&addBDesc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&addCDesc));

  {
    int n = static_cast<int>(input_num_elements);
    MIOPEN_CHECK(miopenSet4dTensorDescriptor(addADesc, data_type, 1, 1, 1, n));
    MIOPEN_CHECK(miopenSet4dTensorDescriptor(addBDesc, data_type, 1, 1, 1, n));
    MIOPEN_CHECK(miopenSet4dTensorDescriptor(addCDesc, data_type, 1, 1, 1, n));
  }

  {
    float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;
    MIOPEN_CHECK(miopenOpTensor(handle, miopenTensorOpAdd, &alpha1, addADesc,
                                input, &alpha2, addBDesc, skip, &beta, addCDesc,
                                skip_buf));
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_skip_simplified_layer_norm: step 1 completed "
                    "(add)\n");

  // =========================================================================
  // Step 1b (optional): Add bias — skip_buf = skip_buf + bias
  // =========================================================================
  if (bias) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_skip_simplified_layer_norm: step 1b — "
                      "adding bias (%lld elements, broadcast)\n",
                      (long long)hidden_dim);

    MIOPEN_CHECK(miopenCreateTensorDescriptor(&biasDesc));
    {
      int n = static_cast<int>(hidden_dim);
      MIOPEN_CHECK(
          miopenSet4dTensorDescriptor(biasDesc, data_type, 1, 1, 1, n));
    }

    {
      float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;
      MIOPEN_CHECK(miopenOpTensor(handle, miopenTensorOpAdd, &alpha1, addCDesc,
                                  skip_buf, &alpha2, biasDesc, bias, &beta,
                                  addCDesc, skip_buf));
    }

    RUNTIME_DEBUG_LOG("[REAL] wrap_skip_simplified_layer_norm: step 1b "
                      "completed (bias add)\n");
  }

  // =========================================================================
  // Step 2: T5 RMS norm — output = RMSNorm(skip_buf) * gamma
  // =========================================================================
  RUNTIME_DEBUG_LOG("[REAL] wrap_skip_simplified_layer_norm: step 2 — "
                    "miopenT5LayerNormForward(eps=%e)\n",
                    (double)epsilon);

  // rstd scratch buffer (always f32)
  {
    hipError_t hip_err = hipMalloc(&rstd_buf, num_rows * sizeof(float));
    if (hip_err != hipSuccess) {
      RUNTIME_DEBUG_LOG(
          "[REAL] wrap_skip_simplified_layer_norm: hipMalloc rstd "
          "failed: %s\n",
          hipGetErrorString(hip_err));
      goto cleanup;
    }
  }

  MIOPEN_CHECK(miopenCreateTensorDescriptor(&xDesc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&weightDesc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&yDesc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&rstdDesc));

  {
    int x_dims[] = {static_cast<int>(num_rows), static_cast<int>(hidden_dim)};
    int x_strides[] = {static_cast<int>(hidden_dim), 1};
    MIOPEN_CHECK(
        miopenSetTensorDescriptor(xDesc, data_type, 2, x_dims, x_strides));
    MIOPEN_CHECK(
        miopenSetTensorDescriptor(yDesc, data_type, 2, x_dims, x_strides));

    int w_dims[] = {static_cast<int>(hidden_dim)};
    int w_strides[] = {1};
    MIOPEN_CHECK(
        miopenSetTensorDescriptor(weightDesc, data_type, 1, w_dims, w_strides));

    int rstd_dims[] = {static_cast<int>(num_rows)};
    int rstd_strides[] = {1};
    MIOPEN_CHECK(miopenSetTensorDescriptor(rstdDesc, miopenFloat, 1, rstd_dims,
                                           rstd_strides));
  }

  MIOPEN_CHECK(miopenT5LayerNormForward(
      handle, MIOPEN_ELEMENTWISE_AFFINE_T5, xDesc, skip_buf, weightDesc, gamma,
      epsilon, yDesc, output, rstdDesc, rstd_buf));

  RUNTIME_DEBUG_LOG("[REAL] wrap_skip_simplified_layer_norm: completed "
                    "successfully\n");
  result = 0;

cleanup:
  if (addADesc)
    miopenDestroyTensorDescriptor(addADesc);
  if (addBDesc)
    miopenDestroyTensorDescriptor(addBDesc);
  if (addCDesc)
    miopenDestroyTensorDescriptor(addCDesc);
  if (biasDesc)
    miopenDestroyTensorDescriptor(biasDesc);
  if (xDesc)
    miopenDestroyTensorDescriptor(xDesc);
  if (weightDesc)
    miopenDestroyTensorDescriptor(weightDesc);
  if (yDesc)
    miopenDestroyTensorDescriptor(yDesc);
  if (rstdDesc)
    miopenDestroyTensorDescriptor(rstdDesc);
  if (rstd_buf)
    hipFree(rstd_buf);
  if (owns_skip_buf && tmp_skip_buf)
    hipFree(tmp_skip_buf);

  return result;
}
