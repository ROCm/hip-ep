/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipDpsOpInterface.cpp - HipDpsOp default reify body ----------------===//
//
// Single shared default `reifyResultShapes` body for HIP DPS ops. Walks
// `DestinationStyleOpInterface::getDpsInits()` and lifts each init's
// runtime shape via `tensor::getMixedSizes` / `memref::getMixedSizes`.
// Ops that need a tighter contract opt out via `autoReify=0` on
// `Hip_DpsOp` and provide a per-op override in
// `HipReifyResultShapesImpl.cpp`.
//
// See `docs/design/hip-shape-inference.md` for the design rationale and
// the recipe for wiring a new op (or a new shape category) into the
// pipeline.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::hip;

namespace mlir::hip {
#include "hip/Dialect/IR/HipDpsOpInterface.cpp.inc"
} // namespace mlir::hip

LogicalResult
HipDpsOp::reifyResultShapes(OpBuilder &b,
                            ReifiedRankedShapedTypeDims &reified) {
  Operation *op = getOperation();
  for (auto [idx, out] :
       llvm::enumerate(cast<DestinationStyleOpInterface>(op).getDpsInits())) {
    SmallVector<OpFoldResult> dims;
    Type outType = out.getType();
    if (isa<RankedTensorType>(outType))
      dims = tensor::getMixedSizes(b, op->getLoc(), out);
    else if (isa<MemRefType>(outType))
      dims = memref::getMixedSizes(b, op->getLoc(), out);
    else
      return op->emitOpError("invalid type for DPS init #")
             << idx << ": expected tensor or memref, got " << outType;
    reified.emplace_back(std::move(dims));
  }
  return success();
}
