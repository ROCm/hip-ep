/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrPoolDomainOp.h"

// Needs the full PoolDomainYieldOp type: the implicit-terminator trait's
// generated methods and verify() below both use it.
#include "hip/Dialect/Hipsr/IR/HipsrPoolDomainYieldOp.h"

#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::hipsr;

LogicalResult PoolDomainOp::verify() {
  // The trait already guarantees a single block ending in PoolDomainYieldOp.
  // Its operands become this op's results, so just check count and types.
  auto yieldOp = cast<PoolDomainYieldOp>(getBody().front().getTerminator());

  if (yieldOp.getNumOperands() != getNumResults())
    return emitOpError() << "has " << getNumResults()
                         << " result(s) but its pool_domain_yield yields "
                         << yieldOp.getNumOperands() << " value(s)";

  for (unsigned idx : llvm::seq<unsigned>(0, getNumResults())) {
    Type resultType = getResultTypes()[idx];
    Type yieldType = yieldOp.getOperandTypes()[idx];
    if (resultType != yieldType)
      return emitOpError() << "result #" << idx << " type " << resultType
                           << " does not match the yielded value type "
                           << yieldType;
  }

  return success();
}

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrPoolDomainOp.cpp.inc"
