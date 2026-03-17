/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef hip_COMPILER_CONVERSION_PASSES_H
#define hip_COMPILER_CONVERSION_PASSES_H

#include "hip/Conversion/HipToLLVM/Passes.h"
#include "hip/Conversion/OnnxToHip/Passes.h"

namespace hip::compiler {

/// Register all conversion passes (ONNX→HIP, HIP→LLVM).
void registerConversionPasses();

} // namespace hip::compiler

#endif // hip_COMPILER_CONVERSION_PASSES_H
