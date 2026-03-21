/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Transforms/Pipelines.h"
#include "hip/Conversion/OnnxToHip/Passes.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "compilation_options_generated.h"

using namespace mlir;

/// Common tail of the ONNX-to-HIP pipeline after the OnnxToHip pass.
static void buildOnnxToHipPipelineTail(OpPassManager &pm) {
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
  pm.addPass(bufferization::createBufferResultsToOutParamsPass(outParamsOpts));

  // 4. Insert ownership-based buffer deallocation
  bufferization::BufferDeallocationPipelineOptions deallocOpts;
  bufferization::buildBufferDeallocationPipeline(pm, deallocOpts);

  // 5. Clean up after bufferization
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());

  // 6. HIP-specific buffer optimizations
  pm.addNestedPass<func::FuncOp>(hip::createOptimizeMemRefsPass());
  pm.addNestedPass<func::FuncOp>(hip::createPoolAllocsPass());

  // 7. Lower remaining bufferization ops to memref
  pm.addPass(createConvertBufferizationToMemRefPass());

  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());

  // 8. Replace memref.alloc with hip.alloc/hip.free
  pm.addNestedPass<func::FuncOp>(hip::createLowerAllocsPass());

  // 9. Resolve extern constants → memref.view into constants blob argument
  pm.addPass(hip::createResolveExternConstantsPass());

  // 10. Final cleanup: LowerAllocs and ResolveExternConstants both introduce
  //     new constants and ops that benefit from deduplication and hoisting.
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
}

void mlir::hip::buildOnnxToHipPipeline(
    OpPassManager &pm, const OnnxToHipPipelineOptions &options) {
  pm.addPass(createHipAddContextArgPass());

  ConvertOnnxToHipPassOptions onnxToHipOpts;
  onnxToHipOpts.externalizeOutputDir = options.externalizeOutputDir;
  onnxToHipOpts.externalizeMinNumElements = options.externalizeMinNumElements;
  pm.addPass(createConvertOnnxToHipPass(std::move(onnxToHipOpts)));

  buildOnnxToHipPipelineTail(pm);
}

void mlir::hip::buildOnnxToHipPipeline(
    OpPassManager &pm, const OnnxToHipPipelineOptions &options,
    morphizen::FileSystem *fs) {
  pm.addPass(createHipAddContextArgPass());

  if (fs) {
    pm.addPass(mlir::hip::createConvertOnnxToHipPass(
        fs, options.externalizeMinNumElements));
  } else {
    ConvertOnnxToHipPassOptions onnxToHipOpts;
    onnxToHipOpts.externalizeOutputDir = options.externalizeOutputDir;
    onnxToHipOpts.externalizeMinNumElements = options.externalizeMinNumElements;
    pm.addPass(createConvertOnnxToHipPass(std::move(onnxToHipOpts)));
  }

  buildOnnxToHipPipelineTail(pm);
}

void mlir::hip::buildHipToLLVMPipeline(
    OpPassManager &pm, const HipToLLVMPipelineOptions &options) {
  pm.addPass(createConvertHipToLLVMPass());

  mlir::hip::CompilationOptionsT compOpts;
  compOpts.constants_file = options.constantsFile;
  pm.addPass(createGenerateInterfacePass(compOpts));
}

void mlir::hip::buildHipdnnPipeline(
    OpPassManager &pm, const HipdnnPipelineOptions &options) {
  OnnxToHipPipelineOptions onnxOpts;
  onnxOpts.externalizeOutputDir = options.constantsDir;
  onnxOpts.externalizeMinNumElements = options.externalizeMinNumElements;
  buildOnnxToHipPipeline(pm, onnxOpts);

  HipToLLVMPipelineOptions llvmOpts;
  llvmOpts.constantsFile = options.constantsFile;
  buildHipToLLVMPipeline(pm, llvmOpts);
}

void mlir::hip::registerHipPipelines() {
  PassPipelineRegistration<OnnxToHipPipelineOptions>(
      "onnx-to-hip-pipeline",
      "Lower ONNX IR to bufferized HIP memref IR with optional constant "
      "externalization",
      [](OpPassManager &pm, const OnnxToHipPipelineOptions &opts) {
        buildOnnxToHipPipeline(pm, opts);
      });

  PassPipelineRegistration<HipToLLVMPipelineOptions>(
      "hip-to-llvm-pipeline",
      "Lower HIP memref IR to LLVM dialect and generate C interface",
      buildHipToLLVMPipeline);

  PassPipelineRegistration<HipdnnPipelineOptions>(
      "hipdnn-pipeline",
      "Complete HIPDNN ONNX→HIP→LLVM→Interface pipeline",
      buildHipdnnPipeline);
}
