/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "morphizen-mlir-compiler/Compiler/Pipeline.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "morphizen-mlir-compiler/Compiler/Passes/Passes.h"
#include "morphizen-mlir-compiler/Conversion/Passes.h"
#include "morphizen-mlir-compiler/Dialect/Hip/Transforms/Passes.h"

namespace morphizen {
namespace compiler {

void populateBufferDeallocationPipeline(mlir::OpPassManager& funcPM) {
  // Buffer deallocation pipeline (runs nested on func.func)
  // These passes manage buffer lifetime and insert deallocation operations
  funcPM.addPass(mlir::bufferization::createBufferLoopHoistingPass());
  funcPM.addPass(
      mlir::bufferization::createOwnershipBasedBufferDeallocationPass());
  funcPM.addPass(mlir::bufferization::createOptimizeAllocationLivenessPass());
}

void populateMorphizenPipeline(mlir::OpPassManager& pm,
                               const PipelineOptions& options) {
  // Stage 1: ONNX → HIP conversion
  // Lowers high-level ONNX operations to GPU-accelerated HIP operations
  pm.addPass(mlir::hip::createConvertOnnxToHipPass());

  // Stage 2: Buffer deallocation (nested under func.func)
  // Manages buffer lifetime and inserts deallocation operations
  auto& funcPM = pm.nest<mlir::func::FuncOp>();
  populateBufferDeallocationPipeline(funcPM);

  // Stage 3: Canonicalization (back at module level)
  // Simplifies IR and applies standard optimizations
  pm.addPass(mlir::createCanonicalizerPass());

  // Stage 4: Memory pooling optimization
  // Merges multiple allocations into a single pool (optional)
  if (options.enableMemoryPooling) {
    pm.addPass(mlir::hip::createMemoryPoolingPass());
  }

  // Stage 5: HIP → LLVM conversion
  // Lowers HIP dialect to LLVM dialect (calls to MIOpen/HIP runtime)
  pm.addPass(mlir::hip::createConvertHipToLLVMPass());

  // Stage 6: Interface generation
  // Generates C-ABI compatible wrapper functions for ONNX Runtime
  pm.addPass(createGenerateInterfacePass());
}

void registerMorphizenPipeline() {
  // Register the complete pipeline for use with morphizen-opt
  // Usage: morphizen-opt input.mlir --morphizen-pipeline
  mlir::PassPipelineRegistration<PipelineOptions>(
      "morphizen-pipeline",
      "Complete ONNX→HIP→LLVM→Interface compilation pipeline",
      populateMorphizenPipeline);
}

} // namespace compiler
} // namespace morphizen
