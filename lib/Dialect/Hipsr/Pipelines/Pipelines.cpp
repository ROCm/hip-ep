/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/Pipelines/Pipelines.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

// --hipsr-pipeline is equivalent to running these in order:
//   --hipsr-add-context-arg
//   --convert-onnx-to-hipsr
//   --hipsr-populate-shape-region
//   --hipsr-partition-pool-domains
//   --hipsr-materialize-init-tensors
//   --remove-shape-constraints
//   --shape-to-shape-lowering
//   --convert-shape-to-std
//   --canonicalize
void mlir::hipsr::buildHipsrPipeline(OpPassManager &pm,
                                     const HipsrPipelineOptions & /*options*/) {
  pm.addPass(createAddContextArgPass());
  pm.addPass(createConvertOnnxToHipsrPass());
  pm.addNestedPass<func::FuncOp>(createPopulateShapeRegionPass());
  pm.addNestedPass<func::FuncOp>(createPartitionPoolDomainsPass());
  pm.addPass(createMaterializeInitTensorsPass());

  // A shape region yields extent tensors, which the upstream shape passes
  // lower. The constraints a recipe records need the other three passes:
  //
  //   - nothing downstream reads them, so remove-shape-constraints drops them;
  //   - the leftover witness always passes and guards a shape.assuming;
  //   - convert-shape-to-std cannot lower that op, so canonicalize inlines it.
  pm.addNestedPass<func::FuncOp>(createRemoveShapeConstraintsPass());
  pm.addNestedPass<func::FuncOp>(createShapeToShapeLoweringPass());
  pm.addPass(createConvertShapeToStandardPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
}

void mlir::hipsr::registerHipsrPipelines() {
  PassPipelineRegistration<HipsrPipelineOptions>(
      "hipsr-pipeline", "Run the hipsr dialect lowering pipeline",
      buildHipsrPipeline);
}
