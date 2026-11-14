/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_DIALECT_TRANSFORMS_QUANTIZEDEPILOGUE_H
#define HIP_DIALECT_TRANSFORMS_QUANTIZEDEPILOGUE_H

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace mlir::hip {

/// Rewrites each quantize of a dequantized add or mul in an outlined rocMLIR
/// kernel to the arithmetic of ORT CPU's QLinearAdd and QLinearMul, so the
/// kernel rounds at the same points ORT does.
void rewriteQuantizedBinaryOps(func::FuncOp kernel);

} // namespace mlir::hip

#endif // HIP_DIALECT_TRANSFORMS_QUANTIZEDEPILOGUE_H
