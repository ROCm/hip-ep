/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Transforms/Pipelines.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "compilation_options_generated.h"

using namespace mlir;

void mlir::hip::buildOnnxToHipPipeline(
    OpPassManager &pm, const OnnxToHipPipelineOptions &options) {
  // 0. Insert !hip.context as arg 0 into every func.func
  pm.addPass(createHipAddContextArgPass());

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
  outParamsOpts.modifyPublicFunctions = true;
  pm.addPass(bufferization::createBufferResultsToOutParamsPass(outParamsOpts));

  // 4. Canonicalize to eliminate dead/residual onnx.* ops and simplify IR
  pm.addPass(createCanonicalizerPass());

  // 5. Memory pooling: pack intermediate memref.alloc into a single pool
  pm.addNestedPass<func::FuncOp>(createPoolAllocsPass());
}

void mlir::hip::buildHipToLLVMPipeline(
    OpPassManager &pm, const HipToLLVMPipelineOptions &options) {
  pm.addPass(createConvertHipToLLVMPass());

  mlir::hip::CompilationOptionsT compOpts;
  compOpts.constants_file = options.constantsFile;
  pm.addPass(createGenerateInterfacePass(compOpts));
}

void mlir::hip::registerHipPipelines() {
  PassPipelineRegistration<OnnxToHipPipelineOptions>(
      "onnx-to-hip-pipeline",
      "Lower ONNX IR to bufferized HIP memref IR with optional constant "
      "externalization",
      buildOnnxToHipPipeline);

  PassPipelineRegistration<HipToLLVMPipelineOptions>(
      "hip-to-llvm-pipeline",
      "Lower HIP memref IR to LLVM dialect and generate C interface",
      buildHipToLLVMPipeline);
}
