/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef hip_COMPILER_COMPILER_PIPELINE_H
#define hip_COMPILER_COMPILER_PIPELINE_H

#include "compilation_options_generated.h"
#include "mlir/Pass/PassManager.h"
#include <cstdint>

namespace morphizen {
class FileSystem;
} // namespace morphizen

namespace hip::compiler {
namespace compiler {

/// Populates the HIP→LLVM sub-pipeline (stages after ONNX conversion).
/// Shared between the full compiler pipeline and hip-mlir-opt.
///
/// Pipeline stages:
/// 1. Bufferization (one-shot + buffer-results-to-out-params)
/// 2. Canonicalization
/// 3. Memory pooling
/// 4. HIP → LLVM conversion
/// 5. Interface generation
void populateHipPipeline(mlir::OpPassManager& pm,
                         const hip::compiler::CompilationOptionsT& options);

/// Populates the complete Morphizen compilation pipeline.
/// This is the SINGLE SOURCE OF TRUTH for the ONNX→HIP→LLVM→Interface pipeline.
///
/// Pipeline stages:
/// 1. HipAddContextArg
/// 2. ONNX → HIP conversion
/// 3. populateHipPipeline (bufferization, canonicalize, pool, HIP→LLVM, interface)
void populateMorphizenPipeline(mlir::OpPassManager& pm,
                               const hip::compiler::CompilationOptionsT& options,
                               morphizen::FileSystem* fs);

} // namespace compiler
} // namespace hip::compiler

#endif // hip_COMPILER_COMPILER_PIPELINE_H
