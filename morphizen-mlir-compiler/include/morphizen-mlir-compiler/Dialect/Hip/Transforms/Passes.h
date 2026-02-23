/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef MORPHIZEN_MLIR_COMPILER_DIALECT_HIP_TRANSFORMS_PASSES_H
#define MORPHIZEN_MLIR_COMPILER_DIALECT_HIP_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace hip {

/// Creates a pass that performs memory pooling optimization on HIP dialect.
/// Merges multiple allocations into a single pool to reduce allocation
/// overhead.
std::unique_ptr<Pass> createMemoryPoolingPass();

/// Creates a pass that performs buffer deallocation for HIP dialect operations.
/// Inserts hip.free operations for allocated buffers.
std::unique_ptr<Pass> createHipBufferDeallocationPass();

/// Register all HIP transform passes.
void registerHipTransformPasses();

} // namespace hip
} // namespace mlir

#endif // MORPHIZEN_MLIR_COMPILER_DIALECT_HIP_TRANSFORMS_PASSES_H
