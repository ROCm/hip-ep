/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_DIALECT_TRANSFORMS_SINKQUANTIZE_H
#define HIP_DIALECT_TRANSFORMS_SINKQUANTIZE_H

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace mlir::hip {

/// Moves each per-tensor quantize backward through the ElementMoveOpInterface
/// ops that feed it, down to per-tensor dequantizes and empty tensors, so the
/// moved ops run on 8-bit values. A dequantize with the quantize's scale,
/// zero point and type folds to its input; any other dequantize is
/// requantized on its own.
void sinkQuantizeThroughElementMoves(func::FuncOp funcOp);

} // namespace mlir::hip

#endif // HIP_DIALECT_TRANSFORMS_SINKQUANTIZE_H
