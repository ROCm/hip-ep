/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrPoolDomainOp.h"
#include "hip/Dialect/Hipsr/IR/HipsrPoolDomainYieldOp.h"

#include <cassert>

using namespace mlir;
using namespace mlir::hipsr;

OperandRange
PoolDomainOp::getEntrySuccessorOperands(RegionSuccessor successor) {
  assert(successor.getSuccessor() == &getBody() &&
         "expected the pool domain body");
  return getOperands();
}

void PoolDomainOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.emplace_back(&getBody(), getBody().getArguments());
    return;
  }

  assert(point.getTerminatorPredecessorOrNull()->getParentRegion() ==
             &getBody() &&
         "expected the pool domain body");
  regions.emplace_back(getOperation(), getResults());
}

void PoolDomainOp::getRegionInvocationBounds(
    ArrayRef<Attribute>, SmallVectorImpl<InvocationBounds> &invocationBounds) {
  invocationBounds.emplace_back(1, 1);
}

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrPoolDomainOp.cpp.inc"
