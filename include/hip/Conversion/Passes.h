//===- Passes.h - HIP conversion pass umbrella header -----------*- C++ -*-===//
//
// Aggregates per-subsystem conversion-pass declarations and exposes the
// TableGen-generated `registerHipConversionPasses()` that registers all of
// them at once. Mirrors `mlir/include/mlir/Conversion/Passes.h` upstream.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_CONVERSION_PASSES_H
#define HIP_CONVERSION_PASSES_H

#include "hip/Conversion/HipToLLVM/Passes.h"
#include "hip/Conversion/OnnxToHip/Passes.h"
#include "hip/Conversion/OnnxToHipDNN/Passes.h"

namespace mlir {
namespace hip {

#define GEN_PASS_REGISTRATION
#include "hip/Conversion/Passes.h.inc"

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_PASSES_H
