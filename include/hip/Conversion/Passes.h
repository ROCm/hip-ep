/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef UDNA_COMPILER_CONVERSION_PASSES_H
#define UDNA_COMPILER_CONVERSION_PASSES_H

#include "hip/Conversion/HipToLLVM/Passes.h"
#include "hip/Conversion/OnnxToHip/Passes.h"

namespace udna::compiler {

/// Register all conversion passes (ONNX→HIP, HIP→LLVM).
void registerConversionPasses();

} // namespace udna::compiler

#endif // UDNA_COMPILER_CONVERSION_PASSES_H
