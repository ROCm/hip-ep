/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- miopen_rms_norm.cpp - hip.miopen.rms_norm runtime
//-------------------===//
//
// RMS normalization via miopenT5LayerNormForward with
// MIOPEN_ELEMENTWISE_AFFINE_T5.
//
// output[n,d] = input[n,d] / rms(input[n,:]) * weight[d]
// where rms(x) = sqrt(mean(x^2) + epsilon)
//
// Signature from MLIR lowering:
//   hip_miopen_rms_norm(handle, input_ptr, weight_ptr, output_ptr, N, D)
//
// input is [N,D], weight is [D], output is [N,D].
//
// Compile (requires MIOPEN_BETA_API for LayerNorm/T5LayerNorm APIs):
//   cl /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /DMIOPEN_BETA_API
//      /I%THEROCK_DIST%\include miopen_rms_norm.cpp
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime_api.h>
#include <miopen/miopen.h>

#define MIOPEN_CHECK(call)                                                     \
  do {                                                                         \
    miopenStatus_t status = (call);                                            \
    if (status != miopenStatusSuccess) {                                       \
      fprintf(stderr, "MIOpen error at %s:%d (status=%d)\n", __FILE__,         \
              __LINE__, status);                                               \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

extern "C" void hip_miopen_rms_norm(void * /*handle*/, void *input,
                                    void *weight, void *output, int64_t N,
                                    int64_t D) {
  fprintf(
      stderr,
      "[miopen.rms_norm] output[%lld,%lld] = RMSNorm(input, weight[%lld])\n",
      (long long)N, (long long)D, (long long)D);

  miopenHandle_t handle = nullptr;
  miopenTensorDescriptor_t inputDesc = nullptr, weightDesc = nullptr;
  miopenTensorDescriptor_t outputDesc = nullptr, rstdDesc = nullptr;
  void *rstd = nullptr;
  float epsilon = 1e-5f;

  miopenCreate(&handle);

  // input / output: [N, D] row-major
  miopenCreateTensorDescriptor(&inputDesc);
  int iDims[] = {(int)N, (int)D};
  int iStrides[] = {(int)D, 1};
  MIOPEN_CHECK(
      miopenSetTensorDescriptor(inputDesc, miopenFloat, 2, iDims, iStrides));

  miopenCreateTensorDescriptor(&outputDesc);
  MIOPEN_CHECK(
      miopenSetTensorDescriptor(outputDesc, miopenFloat, 2, iDims, iStrides));

  // weight: [D]
  miopenCreateTensorDescriptor(&weightDesc);
  int wDims[] = {(int)D};
  int wStrides[] = {1};
  MIOPEN_CHECK(
      miopenSetTensorDescriptor(weightDesc, miopenFloat, 1, wDims, wStrides));

  // rstd (inverse RMS per row): [N]
  miopenCreateTensorDescriptor(&rstdDesc);
  int rDims[] = {(int)N};
  int rStrides[] = {1};
  MIOPEN_CHECK(
      miopenSetTensorDescriptor(rstdDesc, miopenFloat, 1, rDims, rStrides));
  hipMalloc(&rstd, N * sizeof(float));

  MIOPEN_CHECK(miopenT5LayerNormForward(
      handle, MIOPEN_ELEMENTWISE_AFFINE_T5, inputDesc, input, weightDesc,
      weight, epsilon, outputDesc, output, rstdDesc, rstd));
  hipDeviceSynchronize();

cleanup:
  if (rstd)
    hipFree(rstd);
  if (rstdDesc)
    miopenDestroyTensorDescriptor(rstdDesc);
  if (weightDesc)
    miopenDestroyTensorDescriptor(weightDesc);
  if (outputDesc)
    miopenDestroyTensorDescriptor(outputDesc);
  if (inputDesc)
    miopenDestroyTensorDescriptor(inputDesc);
  if (handle)
    miopenDestroy(handle);
}
