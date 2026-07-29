/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Interfaces/DestinationStyleOpInterface.h"

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

  return success();
}
