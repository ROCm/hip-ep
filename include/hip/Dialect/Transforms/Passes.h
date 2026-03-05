/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_PASSES_H
#define HIP_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hip {

#define GEN_PASS_DECL
#include "hip/Dialect/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "hip/Dialect/Transforms/Passes.h.inc"

} // namespace hip
} // namespace mlir

#endif // HIP_PASSES_H
