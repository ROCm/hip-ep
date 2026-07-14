/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_COMPILER_CONVERSION_PASSES_H
#define HIP_COMPILER_CONVERSION_PASSES_H

#include "hip/Conversion/HipToLLVM/Passes.h"
#include "hip/Conversion/OnnxToHip/Passes.h"
#include "hip/Conversion/OnnxToHipDNN/Passes.h"
#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"

namespace mlir {
namespace hipsr {

// Registration hooks for the TableGen-based HIP conversions (currently the
// ONNX->hipsr conversion). Mirrors upstream mlir/Conversion/Passes.h; the
// per-conversion declarations are pulled in by the includes above.
#define GEN_PASS_REGISTRATION
#include "hip/Conversion/Passes.h.inc"

} // namespace hipsr
} // namespace mlir

namespace hip::compiler {

/// Register all conversion passes (ONNX->HIP, HIP->LLVM).
void registerConversionPasses();

} // namespace hip::compiler

#endif // HIP_COMPILER_CONVERSION_PASSES_H
