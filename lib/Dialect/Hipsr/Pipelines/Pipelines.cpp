/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/Pipelines/Pipelines.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"

#include "compilation_options_generated.h"

// --hipsr-pipeline is equivalent to running these in order:
//   --hipsr-add-context-arg
//   --convert-onnx-to-hipsr
//   --hipsr-populate-shape-region
//   --hipsr-partition-pool-domains
//   --hipsr-materialize-init-tensors
//   --remove-shape-constraints
//   --hipsr-convert-shape-to-extent
//   --shape-to-shape-lowering
//   --convert-shape-to-std
//   --one-shot-bufferize
//   --convert-linalg-to-loops
//   --hipsr-use-output-allocator
//   --hipsr-pool-alloc
//   --hipsr-inline-regions
//   --buffer-deallocation-pipeline
//   --optimize-allocation-liveness
//   --hipsr-externalize-constants
//   --assign-op-state-slots
//   --generate-op-state-init
//   --expand-strided-metadata
//   --lower-affine
//   --convert-scf-to-cf
//   --reconcile-unrealized-casts
//   --hipsr-record-graph-io
//   --convert-to-llvm
//   --hipsr-main-graph-abi
//   --generate-interface
void mlir::hipsr::buildHipsrPipeline(OpPassManager &pm,
                                     const HipsrPipelineOptions &options) {
  pm.addPass(createAddContextArgPass());
  pm.addPass(createConvertOnnxToHipsrPass());
  pm.addNestedPass<func::FuncOp>(createPopulateShapeRegionPass());
  pm.addNestedPass<func::FuncOp>(createPartitionPoolDomainsPass());
  pm.addPass(createMaterializeInitTensorsPass());
  pm.addNestedPass<func::FuncOp>(createRemoveShapeConstraintsPass());
  pm.addNestedPass<func::FuncOp>(createConvertShapeToExtentPass());

  // convert-shape-to-std cannot lower shape.num_elements, so rewrite it as a
  // shape.reduce first. This must follow hipsr-convert-shape-to-extent, since
  // ReduceOpConverter rejects a !shape.shape operand.
  pm.addNestedPass<func::FuncOp>(createShapeToShapeLoweringPass());
  pm.addPass(createConvertShapeToStandardPass());

  // useEncodingForMemorySpace turns the #hipsr.mem<...> tensor encoding into
  // the memref memory space.
  bufferization::OneShotBufferizePassOptions bufferizeOptions;
  bufferizeOptions.bufferizeFunctionBoundaries = true;
  bufferizeOptions.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  bufferizeOptions.useEncodingForMemorySpace = true;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOptions));

  // shape.broadcast becomes a tensor.generate, which bufferizes to a
  // linalg.map. That is the only linalg op this pipeline produces.
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());

  pm.addNestedPass<func::FuncOp>(createHipsrUseOutputAllocatorPass());
  pm.addPass(createHipsrPoolAllocPass());
  pm.addPass(createHipsrInlineRegionsPass());

  bufferization::BufferDeallocationPipelineOptions deallocOptions;
  bufferization::buildBufferDeallocationPipeline(pm, deallocOptions);
  pm.addNestedPass<func::FuncOp>(
      bufferization::createOptimizeAllocationLivenessPass());

  HipsrExternalizeConstantsPassOptions extOpts;
  extOpts.constantsFile = options.constantsFile;
  pm.addPass(createHipsrExternalizeConstantsPass(extOpts));

  pm.addPass(hip::createAssignOpStateSlotsPass());
  pm.addPass(hip::createGenerateOpStateInitPass());

  // collapse_shape / expand_shape have no LLVM pattern.
  pm.addPass(memref::createExpandStridedMetadataPass());
  // expand-strided-metadata emits affine.apply for stride products.
  pm.addPass(createLowerAffinePass());
  // convert-linalg-to-loops leaves scf.for / scf.if.
  pm.addPass(createSCFToControlFlowPass());
  // Drop leftover unrealized_conversion_cast from earlier conversions.
  pm.addPass(createReconcileUnrealizedCastsPass());

  // Save the ranked memref shapes before LLVM expands the graph signature.
  pm.addPass(createRecordGraphIOMetadataPass());
  pm.addPass(createConvertToLLVMPass());

  // Wrap the expanded graph in the fixed ABI used by inference_compute.
  pm.addPass(createMainGraphAbiPass());

  mlir::hip::CompilationOptionsT compOpts;
  compOpts.constants_file = options.constantsFile;
  pm.addPass(hip::createGenerateInterfacePass(compOpts));
}

void mlir::hipsr::registerHipsrPipelines() {
  PassPipelineRegistration<HipsrPipelineOptions>(
      "hipsr-pipeline", "Run the hipsr dialect lowering pipeline",
      buildHipsrPipeline);
}
