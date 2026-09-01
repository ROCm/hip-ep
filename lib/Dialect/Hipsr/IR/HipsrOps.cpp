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

// A hipsr op orders its operands as context, inputs, then destinations.
OperandRange mlir::hipsr::getHipsrInputOperands(Operation *op) {
  OperandRange destinations = getHipsrDestinationOperands(op);
  if (destinations.empty()) {
    return OperandRange(nullptr, 0);
  }
  unsigned contextAndInputs = destinations.getBeginOperandIndex();
  return op->getOperands().slice(1, contextAndInputs - 1);
}

Value mlir::hipsr::getShapeGraphCounterpart(Value value) {
  if (PlaceholderOp::isAllowedShapeGraphInput(value)) {
    return value;
  }

  // Block arguments are allowed, so anything left is a result.
  auto result = cast<OpResult>(value);
  OperandRange destinations = getHipsrDestinationOperands(result.getOwner());
  if (result.getResultNumber() >= destinations.size()) {
    return value;
  }
  return destinations[result.getResultNumber()];
}

bool mlir::hipsr::isHipsrDestinationOperand(OpOperand &use) {
  OperandRange destinations = getHipsrDestinationOperands(use.getOwner());
  if (destinations.empty())
    return false;
  unsigned index = use.getOperandNumber();
  unsigned begin = destinations.getBeginOperandIndex();
  return index >= begin && index < begin + destinations.size();
}

// A hipsr op holds the result in the destination slot at the same position.
OpResult mlir::hipsr::getHipsrResultHeldIn(OpOperand &use) {
  if (!isHipsrDestinationOperand(use)) {
    return {};
  }
  Operation *op = use.getOwner();
  unsigned slot = use.getOperandNumber() -
                  getHipsrDestinationOperands(op).getBeginOperandIndex();
  // Bufferization keeps the destinations and drops the results they held.
  if (slot >= op->getNumResults()) {
    return {};
  }
  return op->getResult(slot);
}
