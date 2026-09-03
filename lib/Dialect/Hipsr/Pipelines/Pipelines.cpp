/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/Pipelines/Pipelines.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"

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
void mlir::hipsr::buildHipsrPipeline(OpPassManager &pm,
                                     const HipsrPipelineOptions & /*options*/) {
  pm.addPass(createAddContextArgPass());
  pm.addPass(createConvertOnnxToHipsrPass());
  pm.addNestedPass<func::FuncOp>(createPopulateShapeRegionPass());
  pm.addNestedPass<func::FuncOp>(createPartitionPoolDomainsPass());
  pm.addPass(createMaterializeInitTensorsPass());
  pm.addNestedPass<func::FuncOp>(createRemoveShapeConstraintsPass());
  pm.addNestedPass<func::FuncOp>(createConvertShapeToExtentPass());

  // convert-shape-to-std cannot lower shape.num_elements, so
  // shape-to-shape-lowering rewrites it as a shape.reduce first.
  // It runs after hipsr-convert-shape-to-extent because ReduceOpConverter
  // rejects a !shape.shape operand.
  pm.addNestedPass<func::FuncOp>(createShapeToShapeLoweringPass());
  pm.addPass(createConvertShapeToStandardPass());

  // useEncodingForMemorySpace keeps the memory space, so
  // tensor<4xf16, #hipsr.mem<device>> bufferizes to
  // memref<4xf16, #hipsr.mem<device>>.
  bufferization::OneShotBufferizePassOptions bufferizeOptions;
  bufferizeOptions.bufferizeFunctionBoundaries = true;
  bufferizeOptions.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  bufferizeOptions.useEncodingForMemorySpace = true;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOptions));

  // shape.broadcast lowers to tensor.generate, which bufferizes to a
  // memref.alloc and a linalg.map. That linalg.map is the only linalg op this
  // pipeline produces.
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
}

void mlir::hipsr::registerHipsrPipelines() {
  PassPipelineRegistration<HipsrPipelineOptions>(
      "hipsr-pipeline", "Run the hipsr dialect lowering pipeline",
      buildHipsrPipeline);
}
