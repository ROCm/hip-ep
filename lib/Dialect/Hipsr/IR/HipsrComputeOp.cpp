/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::hipsr;

OperandRange ComputeOp::getEntrySuccessorOperands(RegionSuccessor successor) {
  if (successor.getSuccessor() != &getBody()) {
    llvm::report_fatal_error(
        "hipsr.compute received an unexpected entry successor");
  }
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

// Only the result-to-output pairing needs checking here.
LogicalResult ComputeOp::verify() {
  ValueRange results = getResults();
  if (!results.empty() && results.size() != getOutputs().size()) {
    return emitOpError("expects one result per output, but got ")
           << results.size() << " results and " << getOutputs().size()
           << " outputs";
  }

  return success();
}
