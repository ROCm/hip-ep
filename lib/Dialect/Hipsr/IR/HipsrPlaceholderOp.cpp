/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::hipsr;

Operation *PlaceholderOp::getDpsConsumer() {
  for (Value result : getResults()) {
    for (OpOperand &use : result.getUses()) {
      Operation *owner = use.getOwner();
      auto dpsOp = dyn_cast<DestinationStyleOpInterface>(owner);
      if (dpsOp && owner->getDialect() == getOperation()->getDialect() &&
          dpsOp.isDpsInit(&use)) {
        return owner;
      }
    }
  }
  return nullptr;
}

LogicalResult PlaceholderOp::verify() {
  if (getNumResults() == 0) {
    return emitOpError("must produce at least one tensor DPS init");
  }

  Operation *consumer = nullptr;
  for (auto [resultIndex, result] : llvm::enumerate(getResults())) {
    OpOperand *dpsInitUse = nullptr;
    for (OpOperand &use : result.getUses()) {
      Operation *owner = use.getOwner();
      if (isa<PlaceholderOp>(owner)) {
        continue;
      }

      auto dpsOp = dyn_cast<DestinationStyleOpInterface>(owner);
      if (!dpsOp || owner->getDialect() != getOperation()->getDialect() ||
          !dpsOp.isDpsInit(&use)) {
        return emitOpError("requires each result use to be a placeholder input "
                           "or a DPS init of a hipsr operation");
      }
      if (dpsInitUse) {
        return emitOpError(
            "requires each result to initialize exactly one hipsr operation");
      }
      dpsInitUse = &use;
    }
    if (!dpsInitUse) {
      return emitOpError(
          "requires each result to initialize exactly one hipsr operation");
    }

    Operation *owner = dpsInitUse->getOwner();
    auto dpsOp = cast<DestinationStyleOpInterface>(owner);
    OpResult consumerResult = dpsOp.getTiedOpResult(dpsInitUse);
    if (result.getType() != consumerResult.getType()) {
      return emitOpError("result ")
             << resultIndex << " type " << result.getType()
             << " must match consumer result type " << consumerResult.getType();
    }

    if (consumer && consumer != owner) {
      return emitOpError(
          "requires all results to initialize the same hipsr operation");
    }
    consumer = owner;
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
