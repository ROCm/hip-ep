/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_DIALECT_TRANSFORMS_DECOMPOSEMATMULNBITS_H
#define HIP_DIALECT_TRANSFORMS_DECOMPOSEMATMULNBITS_H

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace mlir::hip {

/// Splits each static 4-bit hip.matmul_nbits into hip.unpack_int4, a blocked
/// hip.dequantize_linear and hip.matmul {transB = 1}, plus a hip.add for the
/// bias, so the rocMLIR fusion treats it as a matmul anchor. An op with a
/// dynamic shape, g_idx or packed zero points keeps the custom kernel.
void decomposeMatMulNBits(func::FuncOp funcOp);

} // namespace mlir::hip

#endif // HIP_DIALECT_TRANSFORMS_DECOMPOSEMATMULNBITS_H
