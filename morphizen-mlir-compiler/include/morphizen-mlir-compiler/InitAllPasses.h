/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef MORPHIZEN_MLIR_COMPILER_INITALLPASSES_H
#define MORPHIZEN_MLIR_COMPILER_INITALLPASSES_H

#include "morphizen-mlir-compiler/Compiler/Passes/Passes.h"
#include "morphizen-mlir-compiler/Compiler/Pipeline.h"
#include "morphizen-mlir-compiler/Conversion/Passes.h"
#include "morphizen-mlir-compiler/Dialect/Hip/Transforms/Passes.h"

namespace morphizen {

/// Register all passes and pipelines for use with morphizen-opt.
/// This is a convenience function that registers:
/// - HIP transform passes (memory pooling, buffer deallocation)
/// - Conversion passes (ONNX→HIP, HIP→LLVM)
/// - Compiler passes (GenerateInterface)
/// - Morphizen pipeline (complete ONNX→HIP→LLVM→Interface)
inline void registerAllPasses() {
  mlir::hip::registerHipTransformPasses();
  registerConversionPasses();
  compiler::registerCompilerPasses();
  compiler::registerMorphizenPipeline();
}

} // namespace morphizen

#endif // MORPHIZEN_MLIR_COMPILER_INITALLPASSES_H
