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

// return Outputs if op is a hipsr.compute op, or the DPS inits if op is a
// hipsr DPS op.
OperandRange mlir::hipsr::getHipsrDestinationOperands(Operation *op) {
  OperandRange none = op->getOperands().take_front(0);
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
  Operation *owner = use.getOwner();
  if (owner->getName().getDialectNamespace() !=
      HipsrDialect::getDialectNamespace()) {
    return false;
  }
  if (auto computeOp = dyn_cast<ComputeOp>(owner)) {
    // use is in Outputs
    size_t firstOutput = 1 + computeOp.getInputs().size();
    size_t operandNumber = use.getOperandNumber();
    return operandNumber >= firstOutput &&
           operandNumber < firstOutput + computeOp.getOutputs().size();
  }
  if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(owner)) {
    return dpsOp.isDpsInit(&use);
  }
  return false;
}
