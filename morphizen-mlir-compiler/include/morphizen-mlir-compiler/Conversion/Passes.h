/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef MORPHIZEN_MLIR_COMPILER_CONVERSION_PASSES_H
#define MORPHIZEN_MLIR_COMPILER_CONVERSION_PASSES_H

#include "morphizen-mlir-compiler/Conversion/HipToLLVM/Passes.h"
#include "morphizen-mlir-compiler/Conversion/OnnxToHip/Passes.h"

namespace morphizen {

/// Register all conversion passes (ONNX→HIP, HIP→LLVM).
void registerConversionPasses();

} // namespace morphizen

#endif // MORPHIZEN_MLIR_COMPILER_CONVERSION_PASSES_H
