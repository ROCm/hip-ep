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

// Groups `ops`, which sit in one block, into one hip.rocmlir_elementwise op
// after the last of them. A value read outside `ops` becomes a result. Returns
// null and leaves the ops unchanged when they cannot form a module.
RocMlirElementwiseOp groupIntoRocMlirElementwise(ArrayRef<Operation *> ops,
                                                 RewriterBase &rewriter);

} // namespace mlir::hip

#endif // HIP_DIALECT_TRANSFORMS_ROCMLIRELEMENTWISE_H
