/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Enable MIOpen beta APIs (miopenT5LayerNormForward, miopenNormMode_t)
#define MIOPEN_BETA_API

#include "../hipdnn_ep_runtime.h"
#include "../debug_log.h"
#include "runtime_types.h"

#include <cstdio>

#define MIOPEN_CHECK(cmd)                                                      \
  do {                                                                         \
    miopenStatus_t status = (cmd);                                             \
    if (status != miopenStatusSuccess) {                                       \
      RUNTIME_DEBUG_LOG("[REAL] MIOpen error %d at %s:%d\n", status, __FILE__,   \
              __LINE__);                                                        \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

// =============================================================================
// SkipSimplifiedLayerNormalization via MIOpen
// =============================================================================
//
// ONNX SkipSimplifiedLayerNormalization (com.microsoft):
//   skip_output = input + skip
//   output      = RMSNorm(skip_output) * gamma
//
// MIOpen has no fused "add + T5 norm" API, so we compose two calls:
//   1. miopenOpTensor(ADD):           skip_output = input + skip
//   2. miopenT5LayerNormForward:      output = RMSNorm(skip_output) * gamma
//
// Both execute on the same GPU stream via the MIOpen handle — no host-device
// round trips. The only overhead vs a hypothetical fused kernel is one extra
// kernel launch.
//
// Tensor layout:
//   input / skip / skip_output:  [num_rows, hidden_dim]  (flat total = input_num_elements)
//   gamma:                       [hidden_dim]             (gamma_num_elements)
//   output:                      [num_rows, hidden_dim]
//   rstd (scratch):              [num_rows]               (f32, not exposed to caller)
// =============================================================================

int wrap_skip_simplified_layer_norm(RuntimeState* state,
                                    void* input, void* skip, void* gamma,
                                    void* output, void* skip_output,
                                    int64_t input_num_elements,
                                    int64_t gamma_num_elements,
                                    int64_t element_size_bytes,
                                    float epsilon) {
  if (!state || !input || !skip || !gamma || !output || !skip_output) {
    fprintf(stderr,
            "[REAL] wrap_skip_simplified_layer_norm: null argument\n");
    return -1;
  }

  miopenHandle_t handle = static_cast<miopenHandle_t>(
      hipdnn_ep_state_get_miopen_handle(state));
  if (!handle) {
    fprintf(stderr,
            "[REAL] wrap_skip_simplified_layer_norm: null MIOpen handle\n");
    return -1;
  }

  int64_t hidden_dim = gamma_num_elements;
  int64_t num_rows = input_num_elements / hidden_dim;

  const char* type_name = (element_size_bytes == 2)   ? "f16"
                          : (element_size_bytes == 4)  ? "f32"
                                                       : "?";
  fprintf(stderr,
          "[REAL] wrap_skip_simplified_layer_norm: num_rows=%lld, "
          "hidden_dim=%lld, data_type=%s, epsilon=%e, "
          "total_bytes=%lld\n",
          (long long)num_rows, (long long)hidden_dim,
          type_name, (double)epsilon,
          (long long)(input_num_elements * element_size_bytes));

  miopenDataType_t data_type;
  if (element_size_bytes == 2)
    data_type = miopenHalf;
  else if (element_size_bytes == 4)
    data_type = miopenFloat;
  else {
    fprintf(stderr,
            "[REAL] wrap_skip_simplified_layer_norm: unsupported "
            "element_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  int rc = -1;
  void* rstd_buf = nullptr;

  // Descriptors for miopenOpTensor (ADD)
  miopenTensorDescriptor_t addADesc = nullptr;
  miopenTensorDescriptor_t addBDesc = nullptr;
  miopenTensorDescriptor_t addCDesc = nullptr;

  // Descriptors for miopenT5LayerNormForward
  miopenTensorDescriptor_t xDesc = nullptr;
  miopenTensorDescriptor_t weightDesc = nullptr;
  miopenTensorDescriptor_t yDesc = nullptr;
  miopenTensorDescriptor_t rstdDesc = nullptr;

  // =========================================================================
  // Step 1: Element-wise add — skip_output = input + skip
  // =========================================================================
  fprintf(stderr,
          "[REAL] wrap_skip_simplified_layer_norm: step 1 — "
          "miopenOpTensor(ADD) for %lld elements\n",
          (long long)input_num_elements);

  MIOPEN_CHECK(miopenCreateTensorDescriptor(&addADesc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&addBDesc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&addCDesc));

  {
    int n = static_cast<int>(input_num_elements);
    MIOPEN_CHECK(
        miopenSet4dTensorDescriptor(addADesc, data_type, 1, 1, 1, n));
    MIOPEN_CHECK(
        miopenSet4dTensorDescriptor(addBDesc, data_type, 1, 1, 1, n));
    MIOPEN_CHECK(
        miopenSet4dTensorDescriptor(addCDesc, data_type, 1, 1, 1, n));
  }

  {
    float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;
    MIOPEN_CHECK(miopenOpTensor(handle, miopenTensorOpAdd,
                                &alpha1, addADesc, input,
                                &alpha2, addBDesc, skip,
                                &beta, addCDesc, skip_output));
  }

  fprintf(stderr,
          "[REAL] wrap_skip_simplified_layer_norm: step 1 completed "
          "(add)\n");

  // =========================================================================
  // Step 2: T5 RMS norm — output = RMSNorm(skip_output) * gamma
  // =========================================================================
  fprintf(stderr,
          "[REAL] wrap_skip_simplified_layer_norm: step 2 — "
          "miopenT5LayerNormForward(eps=%e)\n",
          (double)epsilon);

  // rstd scratch buffer (always f32)
  {
    hipError_t hip_err = hipMalloc(&rstd_buf, num_rows * sizeof(float));
    if (hip_err != hipSuccess) {
      fprintf(stderr,
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
        miopenSetTensorDescriptor(weightDesc, data_type, 1, w_dims,
                                  w_strides));

    int rstd_dims[] = {static_cast<int>(num_rows)};
    int rstd_strides[] = {1};
    MIOPEN_CHECK(miopenSetTensorDescriptor(rstdDesc, miopenFloat, 1,
                                           rstd_dims, rstd_strides));
  }

  MIOPEN_CHECK(miopenT5LayerNormForward(handle, MIOPEN_ELEMENTWISE_AFFINE_T5,
                                        xDesc, skip_output,
                                        weightDesc, gamma,
                                        epsilon,
                                        yDesc, output,
                                        rstdDesc, rstd_buf));

  fprintf(stderr,
          "[REAL] wrap_skip_simplified_layer_norm: completed "
          "successfully\n");
  rc = 0;

cleanup:
  if (addADesc)
    miopenDestroyTensorDescriptor(addADesc);
  if (addBDesc)
    miopenDestroyTensorDescriptor(addBDesc);
  if (addCDesc)
    miopenDestroyTensorDescriptor(addCDesc);
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

  return rc;
}
