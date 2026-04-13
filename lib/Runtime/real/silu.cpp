/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"
#include "error_check_macros.h"
#include "runtime_types.h"

#include <cstdio>

/// SiLU activation: silu(x) = x * sigmoid(x)
///
/// Implemented as two MIOpen operations:
///   1. sigmoid(x) -> output  (via miopenActivationForward)
///   2. x * output -> output  (via miopenOpTensor with OpMul)
///
/// Output buffer is used as temp storage for the sigmoid result,
/// then overwritten with the final x * sigmoid(x) product.
extern "C" int wrap_hip_silu(RuntimeState *state, const void *input,
                             void *output, int64_t n, int64_t data_type) {
  if (!state || !input || !output) {
    fprintf(stderr, "wrap_hip_silu: null argument\n");
    return -1;
  }
  if (n <= 0)
    return 0;

  miopenHandle_t handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  if (!handle) {
    fprintf(stderr, "wrap_hip_silu: null MIOpen handle\n");
    return -1;
  }

  // Map data type
  miopenDataType_t dt;
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    dt = miopenFloat;
    break;
  case HIPDNN_EP_DATATYPE_HALF:
    dt = miopenHalf;
    break;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    dt = miopenBFloat16;
    break;
  default:
    fprintf(stderr, "wrap_hip_silu: unsupported data_type %lld\n",
            (long long)data_type);
    return -1;
  }

  int ni = static_cast<int>(n);
  int result = 0;

  // Create descriptors
  miopenTensorDescriptor_t desc = nullptr;
  miopenActivationDescriptor_t actDesc = nullptr;

  MIOPEN_CHECK_GOTO(miopenCreateTensorDescriptor(&desc), fail);
  MIOPEN_CHECK_GOTO(miopenSet4dTensorDescriptor(desc, dt, 1, 1, 1, ni), fail);
  MIOPEN_CHECK_GOTO(miopenCreateActivationDescriptor(&actDesc), fail);
  MIOPEN_CHECK_GOTO(
      miopenSetActivationDescriptor(actDesc, miopenActivationLOGISTIC, 0, 0, 0),
      fail);

  {
    float alpha = 1.0f, beta = 0.0f;

    // Step 1: sigmoid(input) -> output
    MIOPEN_CHECK_GOTO(miopenActivationForward(handle, actDesc, &alpha, desc,
                                              input, &beta, desc, output),
                      fail);

    // Step 2: input * output -> output (elementwise mul, in-place on output)
    alpha = 1.0f;
    float alpha2 = 1.0f;
    beta = 0.0f;
    MIOPEN_CHECK_GOTO(miopenOpTensor(handle, miopenTensorOpMul, &alpha, desc,
                                     input, &alpha2, desc, output, &beta, desc,
                                     output),
                      fail);
  }

  goto done;

fail:
  fprintf(stderr, "wrap_hip_silu: MIOpen operation failed\n");
  result = -1;

done:
  if (actDesc)
    miopenDestroyActivationDescriptor(actDesc);
  if (desc)
    miopenDestroyTensorDescriptor(desc);
  return result;
}
