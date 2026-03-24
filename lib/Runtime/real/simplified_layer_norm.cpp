/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Enable MIOpen beta APIs (miopenT5LayerNormForward, miopenNormMode_t)
#define MIOPEN_BETA_API

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "runtime_types.h"
#include "error_check_macros.h"

#include <cstdio>

// Use the shared macros from error_check_macros.h with goto cleanup pattern
#define MIOPEN_CHECK(cmd) MIOPEN_CHECK_GOTO(cmd, cleanup)

// =============================================================================
// SimplifiedLayerNormalization via MIOpen T5LayerNorm
// =============================================================================
//
// ONNX SimplifiedLayerNormalization (RMS Norm):
//   rms   = sqrt(mean(input^2, axis) + epsilon)
//   output = (input / rms) * scale
//
// This maps directly to MIOpen's T5LayerNorm with MIOPEN_ELEMENTWISE_AFFINE_T5.
//
// Tensor layout (row-major):
//   input:  [num_rows, hidden_dim]  where num_rows = input_num_elements /
//   scale_num_elements scale:  [hidden_dim] output: [num_rows, hidden_dim]
//   rstd:   [num_rows]              (scratch — not exposed to caller)
// =============================================================================

int wrap_miopenT5LayerNormForward(RuntimeState *state, void *input, void *scale,
                                  void *output, int64_t input_num_elements,
                                  int64_t scale_num_elements,
                                  int64_t element_size_bytes, int64_t axis,
                                  float epsilon, int64_t stash_type) {
  if (!state || !input || !scale || !output) {
    fprintf(stderr, "Invalid arguments to wrap_miopenT5LayerNormForward\n");
    return -1;
  }

  miopenHandle_t handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  if (!handle) {
    fprintf(stderr, "wrap_miopenT5LayerNormForward: null MIOpen handle\n");
    return -1;
  }

  int64_t hidden_dim = scale_num_elements;
  int64_t num_rows = input_num_elements / hidden_dim;

  const char *type_name = (element_size_bytes == 2)   ? "f16"
                          : (element_size_bytes == 4) ? "f32"
                                                      : "?";
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_miopenT5LayerNormForward: num_rows=%lld, hidden_dim=%lld, "
      "data_type=%s, epsilon=%e, "
      "total_bytes=%lld\n",
      (long long)num_rows, (long long)hidden_dim, type_name, (double)epsilon,
      (long long)(input_num_elements * element_size_bytes));

  miopenDataType_t data_type;
  if (element_size_bytes == 2)
    data_type = miopenHalf;
  else if (element_size_bytes == 4)
    data_type = miopenFloat;
  else {
    fprintf(stderr,
            "wrap_miopenT5LayerNormForward: unsupported element_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  int result = 0;
  int rc = -1;
  void *rstd_buf = nullptr;
  miopenTensorDescriptor_t xDesc = nullptr, weightDesc = nullptr,
                           yDesc = nullptr, rstdDesc = nullptr;

  // rstd is always f32 regardless of input type
  HIP_CHECK_GOTO(hipMalloc(&rstd_buf, num_rows * sizeof(float)), cleanup);

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

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_miopenT5LayerNormForward: calling miopenT5LayerNormForward"
      "(mode=ELEMENTWISE_AFFINE_T5, eps=%e)\n",
      (double)epsilon);

  MIOPEN_CHECK(miopenT5LayerNormForward(
      handle, MIOPEN_ELEMENTWISE_AFFINE_T5, xDesc, input, weightDesc, scale,
      epsilon, yDesc, output, rstdDesc, rstd_buf));

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_miopenT5LayerNormForward: completed successfully\n");
  rc = 0;

cleanup:
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
