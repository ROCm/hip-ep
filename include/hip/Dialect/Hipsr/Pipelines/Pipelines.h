/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_DIALECT_HIPSR_PIPELINES_PIPELINES_H
#define HIP_DIALECT_HIPSR_PIPELINES_PIPELINES_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir {
namespace hipsr {

struct HipsrPipelineOptions : public PassPipelineOptions<HipsrPipelineOptions> {
};

void buildHipsrPipeline(OpPassManager &pm,
                        const HipsrPipelineOptions &options = {});

void registerHipsrPipelines();

} // namespace hipsr
} // namespace mlir

#endif // HIP_DIALECT_HIPSR_PIPELINES_PIPELINES_H
