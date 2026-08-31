/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/STLExtras.h"

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

RankedTensorType mlir::hipsr::getExtentTensorTypeForRank(MLIRContext *ctx,
                                                         int64_t rank) {
  return RankedTensorType::get({rank}, IndexType::get(ctx));
}

RankedTensorType mlir::hipsr::getExtentTensorTypeOf(Value data) {
  auto tensorType = cast<RankedTensorType>(data.getType());
  return getExtentTensorTypeForRank(data.getContext(), tensorType.getRank());
}

RankedTensorType mlir::hipsr::getBroadcastExtentTensorType(ValueRange shapes) {
  assert(!shapes.empty() && "broadcast needs at least one shape");
  auto extentCount = [](Value shape) {
    int64_t count = cast<RankedTensorType>(shape.getType()).getDimSize(0);
    // kDynamic is the smallest int64, so an unknown count would lose the
    // comparison below and silently understate the broadcast rank.
    assert(!ShapedType::isDynamic(count) &&
           "broadcast operand must state its extent count");
    return count;
  };
  int64_t rank = *llvm::max_element(llvm::map_range(shapes, extentCount));
  return getExtentTensorTypeForRank(shapes.front().getContext(), rank);
}

OpResult mlir::hipsr::getResultForDestination(OpOperand &use) {
  if (!isHipsrDestinationOperand(use)) {
    return {};
  }
  Operation *op = use.getOwner();
  unsigned slot = use.getOperandNumber() -
                  getHipsrDestinationOperands(op).getBeginOperandIndex();
  if (slot >= op->getNumResults()) {
    return {};
  }
  return op->getResult(slot);
}
