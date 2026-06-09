/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_IR_HIP_INFER_TYPE_UTILS_H
#define HIP_DIALECT_IR_HIP_INFER_TYPE_UTILS_H

#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hip {

/// Shared body for the hand-written `inferReturnTypes` of HIP DPS ops that
/// need a specialization the auto-generated (`autoInfer=1`) one cannot
/// produce -- e.g. ops with more than one DPS init operand, where the
/// generated inference pushes only the single `outsAccessor` result.
///
/// Push one SSA result type per init operand, in DPS-init order, but only for
/// the ranked-tensor-typed inits. In memref mode (post-bufferize) the inits
/// are memrefs and produce no SSA results, so nothing is pushed -- which is
/// exactly what callers want (the op has no results after bufferization).
///
/// `initTypes` MUST be supplied in DPS-init order so the resulting SSA result
/// order matches what the op's converter builds (e.g. for NonZero:
/// result[0] = y, result[1] = count_buf).
///
/// This lives in a dedicated utils file (rather than inline in HipDialect.cpp)
/// so the set of `inferReturnTypes` specializations stays in one place as more
/// multi-output ops are added.
LogicalResult inferDpsInitReturnTypes(TypeRange initTypes,
                                      SmallVectorImpl<Type> &results);

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_IR_HIP_INFER_TYPE_UTILS_H
