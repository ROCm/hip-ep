/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- miopen_softmax.cpp - hip.miopen.softmax runtime
//---------------------===//
//
// Row-wise softmax via miopenSoftmaxForward_V2.
// output[n,:] = softmax(input[n,:])  for each row n.
//
// Signature from MLIR lowering:
//   hip_miopen_softmax(handle, input_ptr, output_ptr, N, D)
//
// input is [N,D], output is [N,D].
// We describe the tensor as 4D [N, D, 1, 1] and use SOFTMAX_MODE_CHANNEL
// to normalize over the D dimension (dim 1).
//
// Compile:
//   cl /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I%THEROCK_DIST%\include
//      miopen_softmax.cpp
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

extern "C" void hip_miopen_softmax(void * /*handle*/, void *input, void *output,
                                   int64_t N, int64_t D) {
  fprintf(stderr, "[miopen.softmax] softmax [%lld, %lld] over last dim\n",
          (long long)N, (long long)D);

  miopenHandle_t handle = nullptr;
  miopenTensorDescriptor_t desc = nullptr;

  miopenCreate(&handle);

  // 4D descriptor [N, D, 1, 1] -- MODE_CHANNEL normalizes over dim 1 (= D)
  miopenCreateTensorDescriptor(&desc);
  int dims[] = {(int)N, (int)D, 1, 1};
  int strides[] = {(int)D, 1, 1, 1};
  MIOPEN_CHECK(miopenSetTensorDescriptor(desc, miopenFloat, 4, dims, strides));

  float alpha = 1.0f, beta = 0.0f;
  MIOPEN_CHECK(miopenSoftmaxForward_V2(handle, &alpha, desc, input, &beta, desc,
                                       output, MIOPEN_SOFTMAX_ACCURATE,
                                       MIOPEN_SOFTMAX_MODE_CHANNEL));
  hipDeviceSynchronize();

cleanup:
  if (desc)
    miopenDestroyTensorDescriptor(desc);
  if (handle)
    miopenDestroy(handle);
}
