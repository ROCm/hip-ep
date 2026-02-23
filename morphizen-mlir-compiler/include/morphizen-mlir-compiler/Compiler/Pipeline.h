/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef MORPHIZEN_MLIR_COMPILER_COMPILER_PIPELINE_H
#define MORPHIZEN_MLIR_COMPILER_COMPILER_PIPELINE_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include <cstdint>

namespace morphizen {
namespace compiler {

/// Options for the Morphizen compilation pipeline.
struct PipelineOptions : public mlir::PassPipelineOptions<PipelineOptions> {
  Option<bool> verbose{*this, "verbose",
                       llvm::cl::desc("Enable verbose output"),
                       llvm::cl::init(false)};
  Option<bool> enableMemoryPooling{
      *this, "enable-memory-pooling",
      llvm::cl::desc("Enable memory pooling optimization"),
      llvm::cl::init(true)};
  Option<int32_t> optLevel{*this, "opt-level",
                           llvm::cl::desc("Optimization level (0-3)"),
                           llvm::cl::init(2)};
};

/// Populates the buffer deallocation sub-pipeline (runs nested on func.func).
/// This is a helper used by the main pipeline.
void populateBufferDeallocationPipeline(mlir::OpPassManager& funcPM);

/// Populates the complete Morphizen compilation pipeline.
/// This is the SINGLE SOURCE OF TRUTH for the ONNX→HIP→LLVM→Interface pipeline.
///
/// Pipeline stages:
/// 1. ONNX → HIP conversion
/// 2. Buffer deallocation (nested on func.func)
/// 3. Canonicalization
/// 4. Memory pooling (optional, controlled by options.enableMemoryPooling)
/// 5. HIP → LLVM conversion
/// 6. Interface generation
void populateMorphizenPipeline(mlir::OpPassManager& pm,
                               const PipelineOptions& options);

/// Registers the Morphizen pipeline for use with morphizen-opt.
/// Registered as: --morphizen-pipeline
void registerMorphizenPipeline();

} // namespace compiler
} // namespace morphizen

#endif // MORPHIZEN_MLIR_COMPILER_COMPILER_PIPELINE_H
