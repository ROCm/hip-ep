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

// createConvertOnnxToHipsrPass() is auto-declared by GEN_PASS_DECL and defined
// by GEN_PASS_DEF (in OnnxToHipsr.cpp); no manual factory is needed since the
// pass has no custom constructor.
#define GEN_PASS_DECL_CONVERTONNXTOHIPSRPASS
#define GEN_PASS_REGISTRATION
#include "hip/Conversion/OnnxToHipsr/Passes.h.inc"

} // namespace hipsr
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIPSR_PASSES_H
