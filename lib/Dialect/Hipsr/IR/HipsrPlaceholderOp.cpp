/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::hipsr;

LogicalResult PlaceholderOp::verify() {
  if (getNumResults() == 0) {
    return emitOpError("must produce at least one tensor DPS init");
  }

  Operation *consumer = nullptr;
  for (Value result : getResults()) {
    if (!result.hasOneUse()) {
      return emitOpError("requires each result to have exactly one use");
    }

    OpOperand &use = *result.getUses().begin();
    Operation *owner = use.getOwner();
    auto dpsOp = dyn_cast<DestinationStyleOpInterface>(owner);
    if (!dpsOp || owner->getDialect() != getOperation()->getDialect() ||
        !dpsOp.isDpsInit(&use)) {
      return emitOpError(
          "requires each result to be used as a DPS init of a hipsr operation");
    }
    if (consumer && consumer != owner) {
      return emitOpError(
          "requires all results to initialize the same hipsr operation");
    }
    consumer = owner;
  }

  auto dpsConsumer = cast<DestinationStyleOpInterface>(consumer);
  SmallVector<Value> consumerInputs = dpsConsumer.getDpsInputs();
  if (getNumOperands() != consumerInputs.size()) {
    return emitOpError("operand count must match consumer DPS input count; "
                       "expected ")
           << consumerInputs.size() << ", got " << getNumOperands();
  }
  for (auto [index, input] : llvm::enumerate(consumerInputs)) {
    if (getOperand(index) != input) {
      return emitOpError("operand ")
             << index << " must match consumer DPS input " << index;
    }
  }

  if (getBodyRegion().empty()) {
    return success();
  }

  Block &block = *getBody();
  if (block.getNumArguments() != getNumOperands()) {
    return emitOpError("shape region block argument count must match operand "
                       "count; expected ")
           << getNumOperands() << ", got " << block.getNumArguments();
  }

  if (block.empty() || !isa<ShapeYieldOp>(block.back())) {
    return emitOpError("shape region must terminate with hipsr.shape_yield");
  }

  return success();
}
