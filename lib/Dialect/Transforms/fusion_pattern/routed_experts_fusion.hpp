/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "mlir/IR/PatternMatch.h"
namespace hip::fusion_transform {
void populateRoutedExpertsPattern(mlir::RewritePatternSet &patterns);
}
