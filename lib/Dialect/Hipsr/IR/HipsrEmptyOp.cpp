/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrEmptyOp.h"

#include "hip/Dialect/Hipsr/IR/HipsrEmptyYieldOp.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h" // tensor::EmptyOp
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::hipsr;

LogicalResult EmptyOp::verify() {
  // RegionBranchOpInterface checks that the yielded values match this op's
  // results. This op additionally requires each value to be a tensor.empty so
  // getMixedSizes() can recover its shape operands.
  auto yieldOp = cast<EmptyYieldOp>(getRegion().front().getTerminator());

  for (auto [idx, tensor] : llvm::enumerate(yieldOp.getTensors())) {
    if (!tensor.getDefiningOp<tensor::EmptyOp>()) {
      return yieldOp.emitOpError("operand #")
             << idx << " must be a tensor.empty result";
    }
  }
  return success();
}

void EmptyOp::getSuccessorRegions(RegionBranchPoint point,
                                  SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.emplace_back(&getRegion());
    return;
  }

  Operation *terminator = point.getTerminatorPredecessorOrNull();
  if (!terminator || terminator->getParentRegion() != &getRegion()) {
    llvm::report_fatal_error("hipsr.empty received an unexpected branch point");
  }
  regions.emplace_back(getOperation(), getResults());
}

SmallVector<SmallVector<OpFoldResult>> EmptyOp::getMixedSizes() {
  assert(!getRegion().empty() &&
         "region must be populated before calling getMixedSizes");
  auto yieldOp = cast<EmptyYieldOp>(getRegion().front().getTerminator());
  SmallVector<SmallVector<OpFoldResult>> sizesPerResult;
  for (Value t : yieldOp.getTensors()) {
    auto emptyTensor = t.getDefiningOp<tensor::EmptyOp>();
    assert(emptyTensor && "hipsr.empty verifier should ensure yielded values "
                          "are tensor.empty results");
    sizesPerResult.push_back(emptyTensor.getMixedSizes());
  }
  return sizesPerResult;
}

SmallVector<RankedTensorType> EmptyOp::getTensorTypes() {
  SmallVector<RankedTensorType> types;
  for (Value result : getResults()) {
    types.push_back(cast<RankedTensorType>(result.getType()));
  }
  return types;
}

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrEmptyOp.cpp.inc"
