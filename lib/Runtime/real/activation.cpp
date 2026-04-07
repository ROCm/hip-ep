/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "error_check_macros.h"
#include "runtime_types.h"

#include <cstdio>

// Convenience wrappers for goto cleanup pattern (all functions use 'cleanup'
// label)
#define MIOPEN_CHECK(cmd) MIOPEN_CHECK_GOTO(cmd, cleanup)

static miopenDataType_t hipdnn_ep_to_miopen_type(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return miopenFloat;
  case HIPDNN_EP_DATATYPE_HALF:
    return miopenHalf;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return miopenBFloat16;
  default:
    fprintf(stderr, "[REAL] unsupported data_type %lld for MIOpen\n",
            (long long)data_type);
    return miopenFloat;
  }
}

// Maps HIPDNN_EP_ACTIVATION_* to miopenActivationMode_t.
// MIOpen calls sigmoid "logistic" (miopenActivationLOGISTIC).
static miopenActivationMode_t hipdnn_ep_to_miopen_activation(int64_t mode) {
  switch (mode) {
  case HIPDNN_EP_ACTIVATION_SIGMOID:
    return miopenActivationLOGISTIC;
  case HIPDNN_EP_ACTIVATION_RELU:
    return miopenActivationRELU;
  case HIPDNN_EP_ACTIVATION_TANH:
    return miopenActivationTANH;
  default:
    fprintf(stderr, "[REAL] unsupported activation_mode %lld for MIOpen\n",
            (long long)mode);
    return miopenActivationLOGISTIC;
  }
}

// =============================================================================
// Generic MIOpen Activation Forward
// =============================================================================
//
// Applies activation_mode element-wise using miopenActivationForward.
// Tensor is represented as flat 1D [1, 1, 1, num_elements] to satisfy
// MIOpen's 4D tensor descriptor requirement.
// =============================================================================

int wrap_miopenActivationForward(RuntimeState *state, void *input, void *output,
                                 int64_t num_elements, int64_t data_type,
                                 int64_t activation_mode) {
  if (!state || !input || !output) {
    fprintf(stderr, "wrap_miopenActivationForward: null argument\n");
    return -1;
  }

  const char *act_name = hipdnn_ep_activation_name(activation_mode);
  const char *type_name = hipdnn_ep_datatype_name(data_type);
  int64_t elem_size = hipdnn_ep_datatype_size(data_type);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_miopenActivationForward: activation=%s, "
      "num_elements=%lld, data_type=%s(%lld), element_size=%lld bytes, "
      "total_size=%lld bytes\n",
      act_name, (long long)num_elements, type_name, (long long)data_type,
      (long long)elem_size, (long long)(num_elements * elem_size));

  miopenHandle_t handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  if (!handle) {
    fprintf(stderr, "wrap_miopenActivationForward: null MIOpen handle\n");
    return -1;
  }

  miopenDataType_t miopen_type = hipdnn_ep_to_miopen_type(data_type);
  miopenActivationMode_t miopen_act =
      hipdnn_ep_to_miopen_activation(activation_mode);

  // Initialize all resource pointers to nullptr for safe cleanup
  miopenTensorDescriptor_t inDesc = nullptr;
  miopenTensorDescriptor_t outDesc = nullptr;
  miopenActivationDescriptor_t actDesc = nullptr;
  int result = 0;
  float alpha = 1.0f, beta = 0.0f;

  int n = static_cast<int>(num_elements);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_miopenActivationForward: creating tensor descriptors "
      "[1,1,1,%d] with type %s\n",
      n, type_name);

  MIOPEN_CHECK(miopenCreateTensorDescriptor(&inDesc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&outDesc));
  // Use miopenSetNdTensorDescriptorWithLayout to set NCHW layout explicitly;
  // miopenSet4dTensorDescriptor leaves the layout as 'UNKNOWN' which triggers
  // warnings in MIOpen 7.12+.
  {
    int dims[] = {1, 1, 1, n};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        inDesc, miopen_type, miopenTensorNCHW, dims, 4));
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        outDesc, miopen_type, miopenTensorNCHW, dims, 4));
  }
  MIOPEN_CHECK(miopenCreateActivationDescriptor(&actDesc));
  MIOPEN_CHECK(
      miopenSetActivationDescriptor(actDesc, miopen_act, 0.0, 0.0, 0.0));

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_miopenActivationForward: calling miopenActivationForward"
      "(%s, alpha=%.1f, beta=%.1f)\n",
      act_name, alpha, beta);

  MIOPEN_CHECK(miopenActivationForward(handle, actDesc, &alpha, inDesc, input,
                                       &beta, outDesc, output));

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_miopenActivationForward: completed successfully\n");

cleanup:
  // Best-effort cleanup: free all allocated resources
  // Continue cleanup even if individual operations fail
  if (actDesc) {
    miopenDestroyActivationDescriptor(actDesc);
  }
  if (inDesc) {
    miopenDestroyTensorDescriptor(inDesc);
  }
  if (outDesc) {
    miopenDestroyTensorDescriptor(outDesc);
  }

  return result;
}
