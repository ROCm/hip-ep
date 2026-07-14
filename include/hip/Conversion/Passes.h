/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_COMPILER_CONVERSION_PASSES_H
#define HIP_COMPILER_CONVERSION_PASSES_H

#include "hip/Conversion/HipToLLVM/Passes.h"
#include "hip/Conversion/OnnxToHip/Passes.h"
#include "hip/Conversion/OnnxToHipDNN/Passes.h"
#include "hip/Conversion/OnnxToHipsr/Passes.h"

namespace hip::compiler {

/// Register all conversion passes (ONNX->HIP, HIP->LLVM).
void registerConversionPasses();

} // namespace hip::compiler

#endif // HIP_COMPILER_CONVERSION_PASSES_H
