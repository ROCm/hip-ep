/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Interfaces/DestinationStyleOpInterface.h"

using namespace mlir;
using namespace mlir::hipsr;

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrOps.cpp.inc"

OperandRange mlir::hipsr::getHipsrDestinationOperands(Operation *op) {
  OperandRange none = OperandRange(nullptr, 0);
  if (op->getName().getDialectNamespace() !=
      HipsrDialect::getDialectNamespace()) {
    return none;
  }
  if (auto computeOp = dyn_cast<ComputeOp>(op)) {
    return computeOp.getOutputs();
  }
  if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op)) {
    return dpsOp.getDpsInits();
  }
  return none;
}

bool mlir::hipsr::isHipsrDestinationOperand(OpOperand &use) {
  OperandRange destinations = getHipsrDestinationOperands(use.getOwner());
  if (destinations.empty())
    return false;
  unsigned index = use.getOperandNumber();
  unsigned begin = destinations.getBeginOperandIndex();
  return index >= begin && index < begin + destinations.size();
}
