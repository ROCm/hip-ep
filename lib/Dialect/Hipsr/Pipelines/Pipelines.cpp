/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/Pipelines/Pipelines.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "mlir/Pass/PassRegistry.h"

void mlir::hipsr::buildHipsrPipeline(OpPassManager &pm,
                                     const HipsrPipelineOptions & /*options*/) {
  // The context argument has to exist before the conversion runs: every
  // conversion pattern reads the runtime context out of function argument 0.
  pm.addPass(createAddContextArgPass());
  pm.addPass(createConvertOnnxToHipsrPass());
}

void mlir::hipsr::registerHipsrPipelines() {
  PassPipelineRegistration<HipsrPipelineOptions>(
      "hipsr-pipeline", "Run the hipsr dialect lowering pipeline",
      buildHipsrPipeline);
}
