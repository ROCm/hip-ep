/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- miopen_add.cpp - hip.miopen.add runtime ----------------------------===//
//
// Element-wise addition C = A + B via miopenOpTensor(miopenTensorOpAdd).
// Supports scalar broadcast: when numB == 1 the single B value is broadcast
// across all elements of A.
//
// Signature from MLIR lowering:
//   hip_miopen_add(handle, A_ptr, B_ptr, C_ptr, numA, numB)
//
// Host pointers (e.g. from memref.global constants) are transparently copied
// to device memory before calling MIOpen.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static void* ensure_device(void* ptr, size_t bytes) {
  if (!ptr || bytes == 0) return ptr;
  hipPointerAttribute_t attrs = {};
  if (hipPointerGetAttributes(&attrs, ptr) == hipSuccess &&
      (attrs.type == hipMemoryTypeDevice ||
       attrs.type == hipMemoryTypeUnified))
    return ptr;
  void* staging = malloc(bytes);
  if (!staging) return ptr;
  memcpy(staging, ptr, bytes);
  void* d = nullptr;
  hipMalloc(&d, bytes);
  hipMemcpy(d, staging, bytes, hipMemcpyHostToDevice);
  free(staging);
  return d;
}

extern "C" void hip_miopen_add(void* /*handle*/, void* A, void* B, void* C,
                               int64_t numA, int64_t numB) {
  fprintf(stderr, "[miopen.add] C = A + B  (numA=%lld numB=%lld)\n",
          (long long)numA, (long long)numB);

  void* devA = ensure_device(A, numA * sizeof(float));
  void* devB = ensure_device(B, numB * sizeof(float));
  void* devC = ensure_device(C, numA * sizeof(float));
  bool allocA = (devA != A), allocB = (devB != B), allocC = (devC != C);

  miopenHandle_t handle = nullptr;
  MIOPEN_CHECK(miopenCreate(&handle));

  miopenTensorDescriptor_t descA = nullptr, descB = nullptr, descC = nullptr;
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&descA));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&descB));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&descC));

  int dimsA[] = {1, (int)numA};
  int stridesA[] = {(int)numA, 1};
  int dimsB[] = {1, (int)numB};
  int stridesB[] = {(int)numB, 1};
  MIOPEN_CHECK(miopenSetTensorDescriptor(descA, miopenFloat, 2, dimsA, stridesA));
  MIOPEN_CHECK(miopenSetTensorDescriptor(descB, miopenFloat, 2, dimsB, stridesB));
  MIOPEN_CHECK(miopenSetTensorDescriptor(descC, miopenFloat, 2, dimsA, stridesA));

  float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;
  MIOPEN_CHECK(miopenOpTensor(handle, miopenTensorOpAdd, &alpha1, descA, devA,
                              &alpha2, descB, devB, &beta, descC, devC));
  hipDeviceSynchronize();

  if (allocC)
    hipMemcpy(C, devC, numA * sizeof(float), hipMemcpyDeviceToHost);

  miopenDestroyTensorDescriptor(descC);
  miopenDestroyTensorDescriptor(descB);
  miopenDestroyTensorDescriptor(descA);
  miopenDestroy(handle);

  if (allocA) hipFree(devA);
  if (allocB) hipFree(devB);
  if (allocC) hipFree(devC);
}
