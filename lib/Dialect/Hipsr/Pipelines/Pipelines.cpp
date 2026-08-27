/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/Pipelines/Pipelines.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"

// --hipsr-pipeline is equivalent to running these in order:
//   --hipsr-add-context-arg
//   --convert-onnx-to-hipsr
//   --hipsr-populate-shape-region
//   --hipsr-partition-pool-domains
void mlir::hipsr::buildHipsrPipeline(OpPassManager &pm,
                                     const HipsrPipelineOptions & /*options*/) {
  pm.addPass(createAddContextArgPass());
  pm.addPass(createConvertOnnxToHipsrPass());
  pm.addNestedPass<func::FuncOp>(createPopulateShapeRegionPass());
  pm.addNestedPass<func::FuncOp>(createPartitionPoolDomainsPass());
}

void mlir::hipsr::registerHipsrPipelines() {
  PassPipelineRegistration<HipsrPipelineOptions>(
      "hipsr-pipeline", "Run the hipsr dialect lowering pipeline",
      buildHipsrPipeline);
}
