/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::hipsr;

MutableOperandRange ComputeOp::getDpsInitsMutable() {
  return getInitsMutable();
}

OperandRange ComputeOp::getEntrySuccessorOperands(RegionSuccessor successor) {
  if (successor.getSuccessor() != &getBody()) {
    llvm::report_fatal_error(
        "hipsr.compute received an unexpected entry successor");
  }
  // The body is isolated from above, so every operand crosses the boundary as
  // an entry-block argument: the context first, then the inputs and the inits.
  return getOperands();
}

void ComputeOp::getSuccessorRegions(RegionBranchPoint point,
                                    SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.emplace_back(&getBody(), getBody().getArguments());
    return;
  }

  Operation *terminator = point.getTerminatorPredecessorOrNull();
  if (!terminator || terminator->getParentRegion() != &getBody()) {
    llvm::report_fatal_error(
        "hipsr.compute received an unexpected branch point");
  }
  regions.emplace_back(getOperation(), getResults());
}
