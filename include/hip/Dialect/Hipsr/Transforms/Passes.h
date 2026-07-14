/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_DIALECT_HIPSR_TRANSFORMS_PASSES_H
#define HIP_DIALECT_HIPSR_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace hipsr {

/// Creates a pass that converts ONNX operations to the hipsr dialect.
std::unique_ptr<::mlir::Pass> createConvertOnnxToHipsrPass();

#define GEN_PASS_REGISTRATION
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

} // namespace hipsr
} // namespace mlir

#endif // HIP_DIALECT_HIPSR_TRANSFORMS_PASSES_H
