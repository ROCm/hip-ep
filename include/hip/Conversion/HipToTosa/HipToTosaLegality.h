/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// The view-op rules convert-hip-to-tosa applies, exported for passes that
// decide what reaches it. hip-fuse-rocmlir outlines subgraphs that must lower
// to TOSA in full; keying its view absorption on these same predicates keeps
// it from pulling in an op the lowering would then refuse.
//
// hip.transpose is not listed: HipToTosa converts every one.

#ifndef HIP_CONVERSION_HIPTOTOSA_HIPTOTOSALEGALITY_H
#define HIP_CONVERSION_HIPTOTOSA_HIPTOTOSALEGALITY_H

#include "hip/Dialect/IR/HipDialect.h"

#include <mlir/Dialect/Tensor/IR/Tensor.h>

namespace mlir::hip {

// A tensor.expand_shape or tensor.collapse_shape with static source and
// result shapes. False for any other op.
bool isTosaLowerableReshape(Operation *op);

// A static, unit-stride, same-rank tensor.extract_slice with constant
// offsets.
bool isTosaLowerableSlice(tensor::ExtractSliceOp op);

// A static hip.expand whose input broadcasts to its result.
bool isTosaLowerableExpand(ExpandOp op);

} // namespace mlir::hip

#endif // HIP_CONVERSION_HIPTOTOSA_HIPTOTOSALEGALITY_H
