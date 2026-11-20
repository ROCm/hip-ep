/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- transpose_removal.hpp - native helpers for TransposeRemoval.pdll ---===//
#pragma once

#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {

void registerTransposeRemovalHelpers(mlir::PDLPatternModule &pdlPatterns);

} // namespace fusion_transform
} // namespace hip
