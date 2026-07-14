/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_CONVERSION_ONNXTOHIPSR_PASSES_H
#define HIP_CONVERSION_ONNXTOHIPSR_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace hipsr {

/// Creates the convert-onnx-to-hipsr pass (definition emitted by GEN_PASS_DEF
/// in OnnxToHipsr.cpp). Declared here so the generated registration hook can
/// reference it.
std::unique_ptr<::mlir::Pass> createConvertOnnxToHipsrPass();

#define GEN_PASS_REGISTRATION
#include "hip/Conversion/OnnxToHipsr/Passes.h.inc"

} // namespace hipsr
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIPSR_PASSES_H
