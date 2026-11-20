/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// The view-op and QDQ rules convert-hip-to-tosa applies, exported for passes
// that decide what reaches it. The rocMLIR fusion outlines subgraphs that must
// lower to TOSA in full; keying its view absorption on these same predicates
// keeps it from pulling in an op the lowering would then refuse.
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

// A static per-tensor or per-axis hip.quantize_linear (saturating) or
// hip.dequantize_linear. False for any other op.
bool isTosaLowerableQdq(Operation *op);

// A pointwise op that convert-hip-to-tosa lowers inside a rocMLIR kernel:
// one that isTosaExpressiblePointwiseOp accepts, a cast to or from any
// integer or float, an 8- or 16-bit unsigned max, min, equal or less, or a
// Where on a byte mask. The rocMLIR fusion takes no other pointwise op; one
// left as a hip op inside a conv or gemm kernel would fail its compile.
bool isTosaLowerableInKernel(Operation *op);

// True when `op` touches no unsigned integer, or when its TOSA form keeps an
// unsigned value's meaning after the kernel is relabeled to signless (views,
// casts, quantize and dequantize, the quantized anchors, a ui8 Where
// condition, and an 8- or 16-bit unsigned max, min, equal or less, which
// compute in i32). Any other op on an unsigned value stops the relabel,
// and rocMLIR then rejects the kernel. A conv or gemm kernel has no HIP
// fallback, so the fusion keeps such an op out of it.
bool isSignlessSafeUnderUnsigned(Operation *op);

// The offset that maps an unsigned integer onto the signed integer of the
// same width, x - offset: 2^(width - 1). 0 for a signless or signed integer,
// which the integer conv and matmul kernels take as is.
int64_t getSignedRebiasOffset(Type elementType);

} // namespace mlir::hip

#endif // HIP_CONVERSION_HIPTOTOSA_HIPTOTOSALEGALITY_H
