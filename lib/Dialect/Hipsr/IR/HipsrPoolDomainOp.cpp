/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::hipsr;

OperandRange
PoolDomainOp::getEntrySuccessorOperands(RegionSuccessor successor) {
  if (successor.getSuccessor() != &getBody()) {
    llvm::report_fatal_error(
        "hipsr.pool_domain received an unexpected entry successor");
  }
  return getOperands();
}

ValueRange PoolDomainOp::getSuccessorInputs(RegionSuccessor successor) {
  return successor.isParent() ? ValueRange(getResults())
                              : ValueRange(getBody().getArguments());
}

void PoolDomainOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.emplace_back(&getBody());
    return;
  }

  Operation *terminator = point.getTerminatorPredecessorOrNull();
  if (!terminator || terminator->getParentRegion() != &getBody()) {
    llvm::report_fatal_error(
        "hipsr.pool_domain received an unexpected branch point");
  }
  regions.push_back(RegionSuccessor::parent());
}
