/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrPoolDomainOp.h"
#include "hip/Dialect/Hipsr/IR/HipsrPoolDomainYieldOp.h"

#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::hipsr;

LogicalResult PoolDomainOp::verify() {
  Block &body = getBody().front();
  if (getNumOperands() != body.getNumArguments()) {
    return emitOpError() << "has " << getNumOperands()
                         << " operand(s) but its entry block has "
                         << body.getNumArguments() << " argument(s)";
  }

  for (unsigned idx : llvm::seq<unsigned>(0, getNumOperands())) {
    Type operandType = getOperand(idx).getType();
    Type argumentType = body.getArgument(idx).getType();
    if (operandType != argumentType) {
      return emitOpError() << "operand #" << idx << " type " << operandType
                           << " does not match entry block argument type "
                           << argumentType;
    }
  }

  auto yieldOp = cast<PoolDomainYieldOp>(body.getTerminator());
  if (yieldOp.getNumOperands() != getNumResults()) {
    return emitOpError() << "has " << getNumResults()
                         << " result(s) but its pool_domain_yield yields "
                         << yieldOp.getNumOperands() << " value(s)";
  }

  for (unsigned idx : llvm::seq<unsigned>(0, getNumResults())) {
    Type resultType = getResult(idx).getType();
    Type yieldType = yieldOp.getOperand(idx).getType();
    if (resultType != yieldType) {
      return emitOpError() << "result #" << idx << " type " << resultType
                           << " does not match the yielded value type "
                           << yieldType;
    }
  }

  return success();
}

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrPoolDomainOp.cpp.inc"
