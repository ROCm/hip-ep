//===- miopen_add.cpp - hip.miopen.add runtime -----------------------------===//
//
// Element-wise addition C = A + B via miopenOpTensor(miopenTensorOpAdd).
//
// Signature from MLIR lowering:
//   hip_miopen_add(handle, A_ptr, B_ptr, C_ptr, numElements)
//
// The lowering computes numElements as the product of all memref dimensions.
// The runtime treats the data as a flat 1D tensor for miopenOpTensor.
//
// Compile:
//   cl /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I%THEROCK_DIST%\include
//      miopen_add.cpp
//
//===----------------------------------------------------------------------===//

#include <miopen/miopen.h>
#include <hip/hip_runtime_api.h>
#include <cstdint>
#include <cstdio>

#define MIOPEN_CHECK(call)                                                \
  do {                                                                    \
    miopenStatus_t status = (call);                                       \
    if (status != miopenStatusSuccess) {                                  \
      fprintf(stderr, "MIOpen error at %s:%d (status=%d)\n",              \
              __FILE__, __LINE__, status);                                \
      return;                                                             \
    }                                                                     \
  } while (0)

extern "C" void hip_miopen_add(void * /*handle*/,
                                void *A, void *B, void *C,
                                int64_t numElements) {
  fprintf(stderr, "[miopen.add] C = A + B  (%lld elements)\n",
          (long long)numElements);

  miopenHandle_t handle = nullptr;
  MIOPEN_CHECK(miopenCreate(&handle));

  // Flat 1D tensor descriptor: [numElements] with stride [1]
  miopenTensorDescriptor_t desc = nullptr;
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&desc));
  int dims[] = {(int)numElements};
  int strides[] = {1};
  MIOPEN_CHECK(miopenSetTensorDescriptor(desc, miopenFloat, 1, dims, strides));

  float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;
  MIOPEN_CHECK(miopenOpTensor(handle, miopenTensorOpAdd,
                               &alpha1, desc, A,
                               &alpha2, desc, B,
                               &beta,   desc, C));
  hipDeviceSynchronize();

  miopenDestroyTensorDescriptor(desc);
  miopenDestroy(handle);
}
