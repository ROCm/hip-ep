/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"
#include "runtime_types.h"

#include <cstdio>

#define MIOPEN_CHECK(cmd)                                                      \
  do {                                                                         \
    miopenStatus_t status = (cmd);                                             \
    if (status != miopenStatusSuccess) {                                       \
      fprintf(stderr, "[REAL] MIOpen error %d at %s:%d\n", status, __FILE__,   \
              __LINE__);                                                       \
      return -1;                                                               \
    }                                                                          \
  } while (0)

static miopenDataType_t hipdnn_ep_to_miopen_type(int64_t data_type) {
  switch (data_type) {
    case HIPDNN_EP_DATATYPE_FLOAT:    return miopenFloat;
    case HIPDNN_EP_DATATYPE_HALF:     return miopenHalf;
    case HIPDNN_EP_DATATYPE_BFLOAT16: return miopenBFloat16;
    default:
      fprintf(stderr,
              "[REAL] unsupported data_type %lld for MIOpen\n",
              (long long)data_type);
      return miopenFloat;
  }
}

// Maps HIPDNN_EP_ACTIVATION_* to miopenActivationMode_t.
// MIOpen calls sigmoid "logistic" (miopenActivationLOGISTIC).
static miopenActivationMode_t hipdnn_ep_to_miopen_activation(int64_t mode) {
  switch (mode) {
    case HIPDNN_EP_ACTIVATION_SIGMOID: return miopenActivationLOGISTIC;
    case HIPDNN_EP_ACTIVATION_RELU:    return miopenActivationRELU;
    case HIPDNN_EP_ACTIVATION_TANH:    return miopenActivationTANH;
    default:
      fprintf(stderr,
              "[REAL] unsupported activation_mode %lld for MIOpen\n",
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
// miopenSet4dTensorDescriptor's 4D requirement.
// =============================================================================

int wrap_miopenActivationForward(RuntimeState* state, void* input, void* output,
                                 int64_t num_elements, int64_t data_type,
                                 int64_t activation_mode) {
  if (!state || !input || !output) {
    fprintf(stderr, "[REAL] wrap_miopenActivationForward: null argument\n");
    return -1;
  }

  const char* act_name = hipdnn_ep_activation_name(activation_mode);
  const char* type_name = hipdnn_ep_datatype_name(data_type);
  int64_t elem_size = hipdnn_ep_datatype_size(data_type);
  fprintf(stderr,
          "[REAL] wrap_miopenActivationForward: activation=%s, "
          "num_elements=%lld, data_type=%s(%lld), element_size=%lld bytes, "
          "total_size=%lld bytes\n",
          act_name, (long long)num_elements, type_name, (long long)data_type,
          (long long)elem_size, (long long)(num_elements * elem_size));

  miopenHandle_t handle = static_cast<miopenHandle_t>(
      hipdnn_ep_state_get_miopen_handle(state));
  if (!handle) {
    fprintf(stderr, "[REAL] wrap_miopenActivationForward: null MIOpen handle\n");
    return -1;
  }

  miopenDataType_t miopen_type = hipdnn_ep_to_miopen_type(data_type);
  miopenActivationMode_t miopen_act = hipdnn_ep_to_miopen_activation(activation_mode);

  int n = static_cast<int>(num_elements);
  fprintf(stderr,
          "[REAL] wrap_miopenActivationForward: creating tensor descriptors "
          "[1,1,1,%d] with type %s\n",
          n, type_name);

  miopenTensorDescriptor_t inDesc, outDesc;
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&inDesc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&outDesc));

  MIOPEN_CHECK(miopenSet4dTensorDescriptor(inDesc, miopen_type, 1, 1, 1, n));
  MIOPEN_CHECK(miopenSet4dTensorDescriptor(outDesc, miopen_type, 1, 1, 1, n));

  miopenActivationDescriptor_t actDesc;
  MIOPEN_CHECK(miopenCreateActivationDescriptor(&actDesc));
  MIOPEN_CHECK(miopenSetActivationDescriptor(actDesc, miopen_act,
                                             0.0, 0.0, 0.0));

  float alpha = 1.0f, beta = 0.0f;
  fprintf(stderr,
          "[REAL] wrap_miopenActivationForward: calling miopenActivationForward"
          "(%s, alpha=%.1f, beta=%.1f)\n",
          act_name, alpha, beta);

  MIOPEN_CHECK(miopenActivationForward(handle, actDesc, &alpha,
                                       inDesc, input, &beta,
                                       outDesc, output));

  fprintf(stderr, "[REAL] wrap_miopenActivationForward: completed successfully\n");

  miopenDestroyActivationDescriptor(actDesc);
  miopenDestroyTensorDescriptor(inDesc);
  miopenDestroyTensorDescriptor(outDesc);
  return 0;
}
