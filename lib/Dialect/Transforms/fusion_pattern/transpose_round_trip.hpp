/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void populateTransposeRoundTripPattern(mlir::RewritePatternSet &patterns);

} // namespace fusion_transform
} // namespace hip
