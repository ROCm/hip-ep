/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::hipsr;

bool PlaceholderOp::isAllowedShapeGraphInput(Value value) {
  if (isa<BlockArgument>(value)) {
    return true;
  }

  Operation *definingOp = value.getDefiningOp();
  return isa_and_nonnull<PlaceholderOp, ConstantOp, arith::ConstantOp>(
      definingOp);
}

// Uniqueness is a rule the verifier imposes:
// every result must fill one outs slot of the same operation, so
// the first outs use found identifies the consumer.
Operation *PlaceholderOp::getConsumer() {
  for (Value result : getResults()) {
    for (OpOperand &use : result.getUses()) {
      if (isHipsrDestinationOperand(use)) {
        return use.getOwner();
      }
    }
  }
  return nullptr;
}

SmallVector<Type> PlaceholderOp::getShapeRegionArgumentTypes() {
  SmallVector<Type> types;
  if (getPlaceholderType() == PlaceholderType::Normal) {
    types.assign(getInputs().size(), shape::ShapeType::get(getContext()));
    return types;
  }

  types.reserve(getNumOperands());
  types.push_back(getCtx().getType());
  llvm::append_range(types, getInputs().getTypes());
  return types;
}

LogicalResult PlaceholderOp::verify() {
  if (getNumResults() == 0) {
    return emitOpError("must produce at least one tensor outs operand");
  }

  for (auto [inputIndex, input] : llvm::enumerate(getInputs())) {
    if (!isAllowedShapeGraphInput(input)) {
      return emitOpError("input ")
             << inputIndex
             << " must be a block argument or a result of hipsr.placeholder, "
                "arith.constant, or hipsr.constant; got result of '"
             << input.getDefiningOp()->getName() << "'";
    }
  }

  Operation *consumer = nullptr;
  for (auto [resultIndex, result] : llvm::enumerate(getResults())) {
    OpOperand *outsUse = nullptr;
    for (OpOperand &use : result.getUses()) {
      if (isa<PlaceholderOp, PoolDomainYieldOp>(use.getOwner())) {
        continue;
      }

      if (!isHipsrDestinationOperand(use)) {
        return emitOpError("requires each result use to be a placeholder "
                           "input, pool-domain yield, or an outs operand of a "
                           "hipsr operation");
      }
      if (outsUse) {
        return emitOpError(
            "requires each result to initialize exactly one hipsr operation");
      }
      outsUse = &use;
    }
    if (!outsUse) {
      return emitOpError(
          "requires each result to initialize exactly one hipsr operation");
    }

    Operation *owner = outsUse->getOwner();
    if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(owner)) {
      OpResult consumerResult = dpsOp.getTiedOpResult(outsUse);
      if (result.getType() != consumerResult.getType()) {
        return emitOpError("result ")
               << resultIndex << " type " << result.getType()
               << " must match consumer result type "
               << consumerResult.getType();
      }
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
  SmallVector<Type> expectedTypes = getShapeRegionArgumentTypes();
  if (block.getNumArguments() != expectedTypes.size()) {
    return emitOpError("shape region block argument count does not match the "
                       "placeholder type layout; expected ")
           << expectedTypes.size() << ", got " << block.getNumArguments();
  }
  for (auto [index, expectedType] : llvm::enumerate(expectedTypes)) {
    Type actualType = block.getArgument(index).getType();
    if (actualType != expectedType) {
      return emitOpError("shape region block argument ")
             << index << " type " << actualType
             << " does not match expected type " << expectedType;
    }
  }

  return success();
}
