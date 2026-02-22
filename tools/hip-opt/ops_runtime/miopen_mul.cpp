/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- miopen_mul.cpp - hip.miopen.mul runtime
//-----------------------------===//
//
// Element-wise multiplication C = A * B via miopenOpTensor(miopenTensorOpMul).
//
// Signature from MLIR lowering:
//   hip_miopen_mul(handle, A_ptr, B_ptr, C_ptr, numElements)
//
// The lowering computes numElements as the product of all memref dimensions.
// The runtime treats the data as a flat 1D tensor for miopenOpTensor.
//
// Compile:
//   cl /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I%THEROCK_DIST%\include
//      miopen_mul.cpp
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime_api.h>
#include <miopen/miopen.h>

#define MIOPEN_CHECK(call)                                             \
  do {                                                                 \
    miopenStatus_t status = (call);                                    \
    if (status != miopenStatusSuccess) {                               \
      fprintf(stderr, "MIOpen error at %s:%d (status=%d)\n", __FILE__, \
              __LINE__, status);                                       \
      return;                                                          \
    }                                                                  \
  } while (0)

extern "C" void hip_miopen_mul(void* /*handle*/, void* A, void* B, void* C,
                               int64_t numElements) {
  fprintf(stderr, "[miopen.mul] C = A * B  (%lld elements)\n",
          (long long)numElements);

  miopenHandle_t handle = nullptr;
  MIOPEN_CHECK(miopenCreate(&handle));

  miopenTensorDescriptor_t desc = nullptr;
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&desc));
  int dims[] = {(int)numElements};
  int strides[] = {1};
  MIOPEN_CHECK(miopenSetTensorDescriptor(desc, miopenFloat, 1, dims, strides));

  float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;
  MIOPEN_CHECK(miopenOpTensor(handle, miopenTensorOpMul, &alpha1, desc, A,
                              &alpha2, desc, B, &beta, desc, C));
  hipDeviceSynchronize();

  miopenDestroyTensorDescriptor(desc);
  miopenDestroy(handle);
}
