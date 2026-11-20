/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_DIALECT_TRANSFORMS_ROCMLIRELEMENTWISE_H
#define HIP_DIALECT_TRANSFORMS_ROCMLIRELEMENTWISE_H

#include "hip/Dialect/IR/HipDialect.h"

#include <mlir/IR/PatternMatch.h>

namespace mlir::hip {

// Replaces `op` with the ops of its body, placed where `op` was.
void inlineRocMlirElementwise(RocMlirElementwiseOp op, RewriterBase &rewriter);

} // namespace mlir::hip

#endif // HIP_DIALECT_TRANSFORMS_ROCMLIRELEMENTWISE_H
