/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_TRANSFORMS_PASSES_H
#define HIPSR_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace hipsr {

#define GEN_PASS_DECL
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

} // namespace hipsr
} // namespace mlir

#endif // HIPSR_TRANSFORMS_PASSES_H
