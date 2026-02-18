//===- miopen_mul.cpp - hip.miopen.mul runtime -----------------------------===//
//
// Element-wise multiplication via miopenOpTensor(miopenTensorOpMul).
// C = A * B
//
// Signature from MLIR lowering:
//   hip_miopen_mul(handle, A, B, C)
//
//===----------------------------------------------------------------------===//

#include <miopen/miopen.h>
#include <hip/hip_runtime_api.h>
#include <cstdint>
#include <cstdio>

extern "C" void hip_miopen_mul(void * /*handle*/, void *A, void *B, void *C) {
  // Structural stub -- full implementation would call:
  //   float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;
  //   miopenOpTensor(miopenHandle, miopenTensorOpMul,
  //       &alpha1, aDesc, A, &alpha2, bDesc, B, &beta, cDesc, C);
  fprintf(stderr, "[hip_miopen_mul] called (A=%p, B=%p, C=%p)\n", A, B, C);
}
