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

/// Options for `hipsr-pipeline`. Empty for now. Add an option here together
/// with the pass that reads it.
struct HipsrPipelineOptions
    : public PassPipelineOptions<HipsrPipelineOptions> {};

/// Build the hipsr pipeline into `pm`. Adds no passes yet.
///
/// The passes it will add, in order: hipsr-populate-shape-region,
/// hipsr-partition-pool-domains, hipsr-materialize-init-tensors,
/// bufferization, hipsr-pool-alloc, hipsr-externalize-constants. See
/// Dialect/Hipsr/Transforms/Passes.td for the order rules.
void buildHipsrPipeline(OpPassManager &pm,
                        const HipsrPipelineOptions &options = {});

/// Register the `hipsr-pipeline` name so it can be used in `hip-mlir-opt` and
/// in pipeline strings.
void registerHipsrPipelines();

} // namespace hipsr
} // namespace mlir

#endif // HIP_DIALECT_HIPSR_PIPELINES_PIPELINES_H
