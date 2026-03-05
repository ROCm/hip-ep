/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Transforms/Pipelines.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

void mlir::hip::buildOnnxToHipPipeline(
    OpPassManager &pm, const OnnxToHipPipelineOptions &options) {
  // 1. ONNX → HIP tensor DPS + constant externalization
  ConvertOnnxToHipPassOptions onnxToHipOpts;
  onnxToHipOpts.externalizeOutputDir = options.externalizeOutputDir;
  onnxToHipOpts.externalizeMinNumElements = options.externalizeMinNumElements;
  pm.addPass(createConvertOnnxToHipPass(std::move(onnxToHipOpts)));

  // 2. Bufferize tensor IR to memref IR
  bufferization::OneShotBufferizePassOptions bufferizeOpts;
  bufferizeOpts.bufferizeFunctionBoundaries = true;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOpts));

  // 3. Convert tensor function results to out-params (memref)
  bufferization::BufferResultsToOutParamsPassOptions outParamsOpts;
  outParamsOpts.hoistStaticAllocs = true;
  outParamsOpts.hoistDynamicAllocs = true;
  outParamsOpts.addResultAttribute = true;
  outParamsOpts.modifyPublicFunctions = true;
  pm.addPass(
      bufferization::createBufferResultsToOutParamsPass(outParamsOpts));

  // 4. Insert ownership-based buffer deallocation
  bufferization::BufferDeallocationPipelineOptions deallocOpts;
  bufferization::buildBufferDeallocationPipeline(pm, deallocOpts);

  // 5. Clean up after bufferization
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());

  // 6. HIP-specific buffer optimizations
  pm.addNestedPass<func::FuncOp>(createOptimizeMemRefsPass());
  pm.addNestedPass<func::FuncOp>(createPoolAllocsPass());

  // 7. Lower remaining bufferization ops to memref
  pm.addPass(createConvertBufferizationToMemRefPass());

  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());

  // 8. Replace memref.alloc with hip.alloc/hip.free
  pm.addNestedPass<func::FuncOp>(createLowerAllocsPass());

  // 9. Resolve extern constants → memref.view into constants blob argument
  pm.addPass(createResolveExternConstantsPass());
}

void mlir::hip::buildHipToLLVMPipeline(OpPassManager &pm) {
  pm.addPass(createConvertHipToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

void mlir::hip::registerHipPipelines() {
  PassPipelineRegistration<OnnxToHipPipelineOptions>(
      "onnx-to-hip-pipeline",
      "Lower ONNX IR to bufferized HIP memref IR with optional constant "
      "externalization",
      buildOnnxToHipPipeline);

  PassPipelineRegistration<>(
      "hip-to-llvm-pipeline",
      "Lower HIP memref IR to LLVM dialect",
      [](OpPassManager &pm) { buildHipToLLVMPipeline(pm); });
}
