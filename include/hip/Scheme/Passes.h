/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DECL
#include "hip/Scheme/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "hip/Scheme/Passes.h.inc"

} // namespace hipsr
} // namespace mlir
