/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Compiler/Pipeline.h"
#include "hip/Compiler/Passes/Passes.h"
#include "hip/Conversion/Passes.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace udna::compiler {
namespace compiler {

void populateHipPipeline(mlir::OpPassManager &pm,
                         const udna::compiler::CompilationOptionsT &options) {
  // Stage 1: One-shot bufferization
  // Converts tensor-mode HIP ops to memref-mode via BufferizableOpInterface
  mlir::bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));

  // Stage 2: Convert function results to out-params (destination-passing style)
  // Transforms func return values into pointer output arguments for C ABI.
  // modifyPublicFunctions=true is required because @main_graph is public.
  mlir::bufferization::BufferResultsToOutParamsPassOptions outParamOpts;
  outParamOpts.modifyPublicFunctions = true;
  pm.addPass(
      mlir::bufferization::createBufferResultsToOutParamsPass(outParamOpts));

  // Stage 3: Canonicalization
  pm.addPass(mlir::createCanonicalizerPass());

  // Stage 4: Memory pooling optimization
  // Packs multiple memref.alloc ops into a single byte pool
  pm.addPass(mlir::hip::createPoolAllocsPass());

  // Stage 5: HIP → LLVM conversion
  pm.addPass(mlir::hip::createConvertHipToLLVMPass());

  // Stage 6: Interface generation
  pm.addPass(createGenerateInterfacePass(options));
}

void populateMorphizenPipeline(mlir::OpPassManager &pm,
                               const udna::compiler::CompilationOptionsT &options,
                               morphizen::FileSystem *fs) {
  // Stage 0: Insert !hip.context argument into all functions
  pm.addPass(mlir::hip::createHipAddContextArgPass());

  // Stage 1: ONNX → HIP conversion (tensor-first, no allocation)
  pm.addPass(mlir::hip::createConvertOnnxToHipPass(fs, options));

  // Stages 2-6: shared HIP→LLVM sub-pipeline
  populateHipPipeline(pm, options);
}

} // namespace compiler
} // namespace udna::compiler
