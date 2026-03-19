/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_COMPILER_COMPILER_PIPELINE_H
#define HIP_COMPILER_COMPILER_PIPELINE_H

#include "compilation_options_generated.h"
#include "mlir/Pass/PassManager.h"

namespace hip::compiler {

/// Populates the complete Morphizen compilation pipeline.
/// This is the SINGLE SOURCE OF TRUTH for the ONNX→HIP→LLVM→Interface pipeline.
///
/// Pipeline stages:
/// 1. HipAddContextArg
/// 2. ONNX → HIP conversion
/// 3. One-shot bufferization + buffer-results-to-out-params
/// 4. Buffer deallocation
/// 5. Buffer optimizations (OptimizeMemRefs, PoolAllocs)
/// 6. LowerAllocs, ResolveExternConstants
/// 7. HIP → LLVM conversion
/// 8. Interface generation
void populateMorphizenPipeline(mlir::OpPassManager &pm,
                               const mlir::hip::CompilationOptionsT &options);

} // namespace hip::compiler

#endif // HIP_COMPILER_COMPILER_PIPELINE_H
