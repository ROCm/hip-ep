/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrPoolDomainOp.h"

// The implicit-terminator trait's generated methods need the full terminator
// type, rather than the forward declaration in HipsrPoolDomainOp.h.
#include "hip/Dialect/Hipsr/IR/HipsrPoolDomainYieldOp.h"

#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::hipsr;

// Explicit operands cross the isolation boundary as entry-block arguments.
OperandRange
PoolDomainOp::getEntrySuccessorOperands(RegionSuccessor successor) {
  if (successor.getSuccessor() != &getBody()) {
    emitOpError("expected its body as the entry successor");
    llvm::report_fatal_error(
        "hipsr.pool_domain received an unexpected entry successor");
  }
  return getOperands();
}

// Report how control and values enter and leave the body. MLIR uses these
// mappings to check argument and result counts and types.
void PoolDomainOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.emplace_back(&getBody(), getBody().getArguments());
    return;
  }

  Operation *terminator = point.getTerminatorPredecessorOrNull();
  if (!terminator || terminator->getParentRegion() != &getBody()) {
    emitOpError("expected a branch point in its body");
    llvm::report_fatal_error(
        "hipsr.pool_domain received an unexpected branch point");
  }
  regions.emplace_back(getOperation(), getResults());
}

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrPoolDomainOp.cpp.inc"
