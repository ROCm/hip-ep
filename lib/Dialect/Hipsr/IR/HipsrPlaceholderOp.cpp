/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::hipsr;

namespace {

// Keep the shape graph apart from the data graph: no data results as inputs.
LogicalResult verifyShapeGraphInputs(PlaceholderOp op) {
  for (auto [index, input] : llvm::enumerate(op.getInputs())) {
    if (!PlaceholderOp::isAllowedShapeGraphInput(input)) {
      return op.emitOpError("input ")
             << index
             << " must be a block argument or a result of hipsr.placeholder, "
                "arith.constant, or hipsr.constant; got result of '"
             << input.getDefiningOp()->getName() << "'";
    }
  }
  return success();
}

// A placeholder holds the outs values of one op, one slot per result.
LogicalResult verifyResultUses(PlaceholderOp op) {
  SmallVector<Operation *> consumers;
  for (OpResult result : op.getResults()) {
    auto initUses =
        llvm::make_filter_range(result.getUses(), [](OpOperand &use) {
          return !isa<PlaceholderOp, PoolDomainYieldOp>(use.getOwner());
        });

    if (!llvm::all_of(initUses, isHipsrDestinationOperand)) {
      return op.emitOpError("requires each result use to be a placeholder "
                            "input, pool-domain yield, or an outs operand of a "
                            "hipsr operation");
    }
    if (!llvm::hasSingleElement(initUses)) {
      return op.emitOpError(
          "requires each result to initialize exactly one hipsr operation");
    }
    consumers.push_back(initUses.begin()->getOwner());
  }

  if (!llvm::all_equal(consumers)) {
    return op.emitOpError(
        "requires all results to initialize the same hipsr operation");
  }
  return success();
}

// A placeholder and its consumer share one destination buffer, so both must
// land in the same pool domain. That holds when the two read the same values,
// the placeholder on the shape-graph side and the consumer on the data side.
//
// Only a value another placeholder holds is checked. A block argument or a
// constant cannot reach a later domain, and inside a pool domain a data value
// and its counterpart are two block arguments that no longer name each other.
LogicalResult verifyConsumerTopology(PlaceholderOp op) {
  Operation *consumer = op.getConsumer();
  if (!consumer) {
    return success();
  }

  SmallVector<Value> counterparts = llvm::map_to_vector(
      getHipsrInputOperands(consumer), getShapeGraphCounterpart);

  for (auto [index, counterpart] : llvm::enumerate(counterparts)) {
    if (!counterpart.getDefiningOp<PlaceholderOp>()) {
      continue;
    }
    if (!llvm::is_contained(op.getInputs(), counterpart)) {
      return op.emitOpError("must read the shape-graph value of input ")
             << index << " of its consumer '" << consumer->getName() << "'";
    }
  }

  for (auto [index, input] : llvm::enumerate(op.getInputs())) {
    if (!input.getDefiningOp<PlaceholderOp>()) {
      continue;
    }
    if (!llvm::is_contained(counterparts, input)) {
      return op.emitOpError("input ")
             << index << " has no matching operand in its consumer '"
             << consumer->getName() << "'";
    }
  }
  return success();
}

// The placeholder type sets the shape region's block arguments.
LogicalResult verifyShapeRegionSignature(PlaceholderOp op) {
  if (op.getBodyRegion().empty()) {
    return success();
  }

  Block &block = *op.getBody();
  SmallVector<Type> expectedTypes = op.getShapeRegionArgumentTypes();
  if (block.getNumArguments() != expectedTypes.size()) {
    return op.emitOpError("shape region block argument count does not match "
                          "the placeholder type layout; expected ")
           << expectedTypes.size() << ", got " << block.getNumArguments();
  }
  for (auto [index, actualType, expectedType] :
       llvm::enumerate(block.getArgumentTypes(), expectedTypes)) {
    if (actualType != expectedType) {
      return op.emitOpError("shape region block argument ")
             << index << " type " << actualType
             << " does not match expected type " << expectedType;
    }
  }
  return success();
}

} // namespace

bool PlaceholderOp::isAllowedShapeGraphInput(Value value) {
  if (isa<BlockArgument>(value)) {
    return true;
  }

  Operation *definingOp = value.getDefiningOp();
  return isa_and_nonnull<PlaceholderOp, ConstantOp, arith::ConstantOp>(
      definingOp);
}

// All results go to the same op, so the first outs use names the consumer.
Operation *PlaceholderOp::getConsumer() {
  for (OpResult result : getResults()) {
    for (OpOperand &use : result.getUses()) {
      if (isHipsrDestinationOperand(use)) {
        return use.getOwner();
      }
    }
  }
  return nullptr;
}

SmallVector<Type> PlaceholderOp::getShapeRegionArgumentTypes() {
  if (getPlaceholderType() == PlaceholderType::Normal) {
    return llvm::map_to_vector(getInputs(), [](Value input) -> Type {
      return getExtentTensorTypeOf(input);
    });
  }

  SmallVector<Type> types;
  types.reserve(getNumOperands());
  types.push_back(getCtx().getType());
  llvm::append_range(types, getInputs().getTypes());
  return types;
}

LogicalResult PlaceholderOp::verify() {
  if (getNumResults() == 0) {
    return emitOpError("must produce at least one tensor outs operand");
  }
  if (failed(verifyShapeGraphInputs(*this))) {
    return failure();
  }
  if (failed(verifyResultUses(*this))) {
    return failure();
  }
  if (failed(verifyConsumerTopology(*this))) {
    return failure();
  }
  if (failed(verifyShapeRegionSignature(*this))) {
    return failure();
  }
  return success();
}
