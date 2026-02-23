/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef MORPHIZEN_MLIR_COMPILER_CONVERSION_ONNXTOHIP_PASSES_H
#define MORPHIZEN_MLIR_COMPILER_CONVERSION_ONNXTOHIP_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace hip {

/// Creates a pass that converts ONNX operations to HIP dialect.
/// Lowers high-level ONNX tensor operations to GPU-accelerated HIP operations.
std::unique_ptr<Pass> createConvertOnnxToHipPass();

} // namespace hip
} // namespace mlir

#endif // MORPHIZEN_MLIR_COMPILER_CONVERSION_ONNXTOHIP_PASSES_H
