//===- miopen_skip_rms_norm.cpp - hip.miopen.skip_rms_norm runtime ---------===//
//
// Fused Add + RMS normalization via miopenAddLayerNormForward (T5 mode).
//   residual = x + skip
//   output   = RMSNorm(residual) * weight
//
// Signature from MLIR lowering:
//   hip_miopen_skip_rms_norm(handle, x, skip, weight, output, residual)
//
//===----------------------------------------------------------------------===//

#include <miopen/miopen.h>
#include <hip/hip_runtime_api.h>
#include <cstdint>
#include <cstdio>

extern "C" void hip_miopen_skip_rms_norm(void * /*handle*/, void *x,
                                          void *skip, void *weight,
                                          void *output, void *residual) {
  // Structural stub -- full implementation would call:
  //   miopenAddLayerNormForward(miopenHandle, MIOPEN_ELEMENTWISE_AFFINE_T5,
  //       xDesc, x, x2Desc, skip, weightDesc, weight, biasDesc, nullptr,
  //       epsilon, normalized_dim, yDesc, output, meanDesc, nullptr,
  //       rstdDesc, nullptr)
  // and also write the residual output.
  fprintf(stderr,
          "[hip_miopen_skip_rms_norm] called (x=%p, skip=%p, weight=%p, "
          "output=%p, residual=%p)\n",
          x, skip, weight, output, residual);
}
