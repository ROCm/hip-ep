/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipDpsOpInterface.cpp - HipDpsOp default reify body ----------------===//
//
// Shared reification bodies for HIP DPS ops. Whole-shape reification walks
// `DestinationStyleOpInterface::getDpsInits()` and lifts each init's runtime
// shape. Direct dimension reification follows one result's tied destination
// and materializes only the requested extent. In memref mode there are no SSA
// results, so whole-shape reification returns an empty list.
// Ops that need a tighter contract select a manual-reify family and provide a
// per-op override in `HipReifyResultShapesImpl.cpp`.
//
// See `docs/design/hip-shape-inference.md` for the design rationale and
// the recipe for wiring a new op (or a new shape category) into the
// pipeline.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"

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
  auto dpsOp = cast<DestinationStyleOpInterface>(op);
  Operation::operand_range inits = dpsOp.getDpsInits();

  // Bufferized DPS ops write through destination memrefs and have no SSA
  // results. The upstream interface contract is one vector per op result, so
  // memref mode succeeds with an empty list rather than reporting destination
  // memref shapes as nonexistent result shapes.
  reified.clear();
  if (op->getNumResults() == 0)
    return success();

  if (inits.size() != op->getNumResults())
    return op->emitOpError()
           << "cannot reify tensor result shapes: expected one DPS init per "
              "result, but got "
           << inits.size() << " init(s) for " << op->getNumResults()
           << " result(s)";

  // Validate every result/init pair before getMixedSizes emits tensor.dim.
  // Failure must leave the IR unchanged and the reified output list empty.
  for (auto [idx, result, out] : llvm::enumerate(op->getResults(), inits)) {
    Type resultType = result.getType();
    Type outType = out.getType();
    if (!isa<RankedTensorType>(resultType))
      return op->emitOpError("invalid tensor-mode result #")
             << idx << ": expected ranked tensor, got " << resultType;
    if (!isa<RankedTensorType>(outType))
      return op->emitOpError("invalid tensor-mode DPS init #")
             << idx << ": expected ranked tensor, got " << outType;
    auto resultTensor = cast<RankedTensorType>(resultType);
    auto outTensor = cast<RankedTensorType>(outType);
    if (resultTensor.getRank() != outTensor.getRank())
      return op->emitOpError("invalid tensor-mode result/init pair #")
             << idx << ": result rank " << resultTensor.getRank()
             << " does not match DPS init rank " << outTensor.getRank();
  }

  reified.reserve(op->getNumResults());
  for (Value out : inits) {
    SmallVector<OpFoldResult> dims =
        tensor::getMixedSizes(b, op->getLoc(), out);
    reified.emplace_back(std::move(dims));
  }
  return success();
}

FailureOr<OpFoldResult> HipDpsOp::reifyDimOfResult(OpBuilder &b,
                                                   int resultIndex, int dim) {
  Operation *op = getOperation();
  auto dpsOp = cast<DestinationStyleOpInterface>(op);
  Operation::operand_range inits = dpsOp.getDpsInits();

  // getTiedOpOperand asserts that the result and init indices are valid. Check
  // the complete DPS relationship before asking for the tie or emitting a
  // tensor.dim operation.
  if (resultIndex < 0 || resultIndex >= static_cast<int>(op->getNumResults()) ||
      inits.size() != op->getNumResults())
    return failure();

  OpResult result = op->getResult(resultIndex);
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (!resultType || dim < 0 || dim >= resultType.getRank())
    return failure();

  OpOperand *tiedOperand = dpsOp.getTiedOpOperand(result);
  if (!tiedOperand)
    return failure();
  auto initType = dyn_cast<RankedTensorType>(tiedOperand->get().getType());
  if (!initType || initType != resultType)
    return failure();

  return tensor::getMixedSize(b, op->getLoc(), tiedOperand->get(), dim);
}
