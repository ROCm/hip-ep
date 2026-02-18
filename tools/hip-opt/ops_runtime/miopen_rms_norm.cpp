//===- miopen_rms_norm.cpp - hip.miopen.rms_norm runtime -------------------===//
//
// RMS normalization via miopenT5LayerNormForward.
// Mode: MIOPEN_ELEMENTWISE_AFFINE_T5
//
// Signature from MLIR lowering:
//   hip_miopen_rms_norm(handle, input, weight, output)
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

extern "C" void hip_miopen_rms_norm(void * /*handle*/, void *input,
                                     void *weight, void *output) {
  // Without shape metadata passed from the lowering layer, we cannot set up
  // the MIOpen tensor descriptors.  This is a structural stub showing the
  // correct API call sequence.
  //
  // A full implementation would:
  //   1. Create tensor descriptors for input (NxD), weight (D), output (NxD)
  //   2. Call miopenT5LayerNormForward(miopenHandle, MIOPEN_ELEMENTWISE_AFFINE_T5,
  //        inputDesc, input, weightDesc, weight, epsilon,
  //        outputDesc, output, rstdDesc, rstd)
  //   3. Destroy descriptors
  fprintf(stderr, "[hip_miopen_rms_norm] called (input=%p, weight=%p, output=%p)\n",
          input, weight, output);
}
