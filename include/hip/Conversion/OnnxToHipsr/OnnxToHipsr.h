/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_CONVERSION_ONNXTOHIPSR_ONNXTOHIPSR_H
#define HIP_CONVERSION_ONNXTOHIPSR_ONNXTOHIPSR_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace hipsr {

// Per-conversion declarations. createConvertOnnxToHipsrPass() is declared by
// GEN_PASS_DECL (defined by GEN_PASS_DEF in OnnxToHipsr.cpp).
// Pattern-population helpers for individual ONNX ops (added by follow-up
// layers) are declared here too. Registration lives in the aggregate
// hip/Conversion/Passes.h.
#define GEN_PASS_DECL_CONVERTONNXTOHIPSRPASS
#include "hip/Conversion/Passes.h.inc"

} // namespace hipsr
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIPSR_ONNXTOHIPSR_H
