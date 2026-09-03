/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/Pipelines/Pipelines.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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
void mlir::hipsr::buildHipsrPipeline(OpPassManager &pm,
                                     const HipsrPipelineOptions & /*options*/) {
  pm.addPass(createAddContextArgPass());
  pm.addPass(createConvertOnnxToHipsrPass());
  pm.addNestedPass<func::FuncOp>(createPopulateShapeRegionPass());
  pm.addNestedPass<func::FuncOp>(createPartitionPoolDomainsPass());
  pm.addPass(createMaterializeInitTensorsPass());
  pm.addNestedPass<func::FuncOp>(createRemoveShapeConstraintsPass());
  pm.addNestedPass<func::FuncOp>(createConvertShapeToExtentPass());
}

void mlir::hipsr::registerHipsrPipelines() {
  PassPipelineRegistration<HipsrPipelineOptions>(
      "hipsr-pipeline", "Run the hipsr dialect lowering pipeline",
      buildHipsrPipeline);
}
