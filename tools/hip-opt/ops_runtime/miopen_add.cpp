//===- miopen_add.cpp - hip.miopen.add runtime -----------------------------===//
//
// Element-wise addition via miopenOpTensor(miopenTensorOpAdd).
// C = A + B
//
// Signature from MLIR lowering:
//   hip_miopen_add(handle, A, B, C)
//
//===----------------------------------------------------------------------===//

#include <miopen/miopen.h>
#include <hip/hip_runtime_api.h>
#include <cstdint>
#include <cstdio>

extern "C" void hip_miopen_add(void * /*handle*/, void *A, void *B, void *C) {
  // Structural stub -- full implementation would call:
  //   float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;
  //   miopenOpTensor(miopenHandle, miopenTensorOpAdd,
  //       &alpha1, aDesc, A, &alpha2, bDesc, B, &beta, cDesc, C);
  fprintf(stderr, "[hip_miopen_add] called (A=%p, B=%p, C=%p)\n", A, B, C);
}
