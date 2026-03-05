/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef UDNA_COMPILER_COMPILER_PIPELINE_H
#define UDNA_COMPILER_COMPILER_PIPELINE_H

#include "compilation_options_generated.h"
#include "mlir/Pass/PassManager.h"
#include <cstdint>

namespace morphizen {
class FileSystem;
} // namespace morphizen

namespace udna::compiler {
namespace compiler {

/// Populates the complete Morphizen compilation pipeline.
/// This is the SINGLE SOURCE OF TRUTH for the ONNX→HIP→LLVM→Interface pipeline.
///
/// Pipeline stages:
/// 1. ONNX → HIP conversion
/// 2. Bufferization (one-shot + buffer-results-to-out-params)
/// 3. Canonicalization
/// 4. Memory pooling
/// 5. HIP → LLVM conversion
/// 6. Interface generation
void populateMorphizenPipeline(mlir::OpPassManager& pm,
                               const udna::compiler::CompilationOptionsT& options,
                               morphizen::FileSystem* fs);

} // namespace compiler
} // namespace udna::compiler

#endif // UDNA_COMPILER_COMPILER_PIPELINE_H
