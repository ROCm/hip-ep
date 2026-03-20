/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Compiler/Pipeline.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace hip::compiler {

void populateMorphizenPipeline(mlir::OpPassManager &pm,
                               const mlir::hip::CompilationOptionsT &options) {
  // Stage 0: Insert !hip.context argument into all functions
  pm.addPass(mlir::hip::createHipAddContextArgPass());

  // Stage 1: ONNX → HIP conversion (tensor-first, no allocation)
  pm.addPass(mlir::hip::createConvertOnnxToHipPass());

  // Stage 2: One-shot bufferization
  mlir::bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));

  // Stage 3: Convert function results to out-params (destination-passing style)
  mlir::bufferization::BufferResultsToOutParamsPassOptions outParamOpts;
  outParamOpts.modifyPublicFunctions = true;
  outParamOpts.hoistStaticAllocs = true;
  outParamOpts.hoistDynamicAllocs = true;
  outParamOpts.addResultAttribute = true;
  pm.addPass(
      mlir::bufferization::createBufferResultsToOutParamsPass(outParamOpts));

  // Stage 4: Buffer deallocation
  mlir::bufferization::BufferDeallocationPipelineOptions deallocOpts;
  mlir::bufferization::buildBufferDeallocationPipeline(pm, deallocOpts);

  // Stage 5: Cleanup after bufferization
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Stage 6: HIP-specific buffer optimizations
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hip::createOptimizeMemRefsPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hip::createPoolAllocsPass());

  // Stage 7: Lower remaining bufferization ops to memref
  pm.addPass(mlir::createConvertBufferizationToMemRefPass());

  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Stage 8: Replace memref.alloc with hip.alloc/hip.free
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hip::createLowerAllocsPass());

  // Stage 9: Resolve extern constants
  pm.addPass(mlir::hip::createResolveExternConstantsPass());

  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Stage 10: HIP → LLVM conversion
  pm.addPass(mlir::hip::createConvertHipToLLVMPass());

  // Stage 11: Interface generation
  mlir::hip::CompilationOptionsT compOpts;
  compOpts.constants_file = options.constants_file;
  pm.addPass(mlir::hip::createGenerateInterfacePass(compOpts));
}

} // namespace hip::compiler
