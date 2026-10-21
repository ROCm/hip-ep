/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "llvm/Config/llvm-config.h"
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
  return !successor.getSuccessor() ? ValueRange(getResults())
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
#if LLVM_VERSION_MAJOR >= 24
  regions.emplace_back(getOperation());
#else
  regions.emplace_back(nullptr);
#endif
}
