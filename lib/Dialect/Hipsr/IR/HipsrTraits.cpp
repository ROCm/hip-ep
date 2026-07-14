/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrTraits.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>

using namespace mlir;

// Same structure as MLIR's OpTrait::impl::verifyIsIsolatedFromAbove
// (mlir/lib/IR/Operation.cpp), with one added rule: a region may also use the
// op's own operands, not just values defined inside its regions.
LogicalResult
mlir::hipsr::OpTrait::impl::verifyIsolatedFromAboveButAllowOperands(
    Operation *isolatedOp) {
  assert(isolatedOp->hasTrait<
             mlir::hipsr::OpTrait::IsolatedFromAboveButAllowOperands>() &&
         "Intended to check IsolatedFromAboveButAllowOperands ops");

  llvm::DenseSet<Value> allowedOperands;
  for (Value operand : isolatedOp->getOperands())
    allowedOperands.insert(operand);

  llvm::SmallVector<Region *, 8> pendingRegions;
  for (Region &region : isolatedOp->getRegions()) {
    pendingRegions.push_back(&region);

    while (!pendingRegions.empty()) {
      for (Operation &op : pendingRegions.pop_back_val()->getOps()) {
        for (Value operand : op.getOperands()) {
          Region *operandRegion = operand.getParentRegion();
          if (!operandRegion)
            return op.emitError("operation's operand is unlinked");

          // OK if the value is defined in this region or a nested one
          // (isAncestor covers block arguments too), or is an op operand.
          if (!region.isAncestor(operandRegion) &&
              !allowedOperands.contains(operand)) {
            return op.emitOpError("using value defined outside the region")
                       .attachNote(isolatedOp->getLoc())
                   << "may only use values defined in its regions or the op's "
                      "operands";
          }
        }

        // Descend, but skip ops that isolate themselves -- they run their own
        // check.
        if (op.getNumRegions() &&
            !op.hasTrait<mlir::OpTrait::IsIsolatedFromAbove>() &&
            !op.hasTrait<
                mlir::hipsr::OpTrait::IsolatedFromAboveButAllowOperands>()) {
          for (Region &subRegion : op.getRegions())
            pendingRegions.push_back(&subRegion);
        }
      }
    }
  }

  return success();
}
