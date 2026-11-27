/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- runtime_shape.hpp - prove two tensors share their runtime extents --===//
//
// Identical types do not prove a dynamic axis has one runtime extent:
// tensor<?x?x14336xf16> matches tensor<?x?x14336xf16> when one side is
// [4, S, 14336] and the other is [1, S, 14336], and an elementwise hip op
// would broadcast. Each dynamic axis must trace to the same extent. A block
// argument is its own extent. hip.matmul and hip.qmatmul contribute batch
// and M from the activation (N from the weight); hip.matmul_nbits
// contributes every axis but the last from the activation and the last
// from its N attribute. That is the same source choice as
// reifyMatmulLikeShape and MatMulNBitsOp::reifyResultShapes.
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/Value.h"

namespace hip {
namespace fusion_transform {

/// True when `lhs` and `rhs` have the same ranked type and every dynamic
/// axis traces to a common, proven runtime extent.
bool sameRuntimeShape(mlir::Value lhs, mlir::Value rhs);

/// True when axis `lhsDim` of `lhs` and axis `rhsDim` of `rhs` trace to a
/// common, proven runtime extent (or the same static size).
bool sameRuntimeExtent(mlir::Value lhs, int64_t lhsDim, mlir::Value rhs,
                       int64_t rhsDim);

} // namespace fusion_transform
} // namespace hip
