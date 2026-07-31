/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionPopulation.h"

#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <utility>

namespace mlir {
namespace hipsr {

FailureOr<ShapeRegionPopulationMatch>
ShapeRegionPopulationPatternSet::match(PlaceholderOp placeholder,
                                       Operation *consumer) const {
  llvm::StringRef operationName = consumer->getName().getStringRef();
  for (const ShapeRegionPopulationPattern &pattern : patterns) {
    if (pattern.operationName == operationName) {
      return ShapeRegionPopulationMatch{&pattern,
                                        pattern.getPlaceholderType(consumer)};
    }
  }
  placeholder.emitOpError("shape-region population does not support consumer ")
      << consumer->getName();
  return failure();
}

unsigned getShapedDpsInputCount(Operation *consumer) {
  auto dps = cast<DestinationStyleOpInterface>(consumer);
  return static_cast<unsigned>(
      llvm::count_if(dps.getDpsInputOperands(), [](OpOperand *operand) {
        return isa<ShapedType>(operand->get().getType());
      }));
}

FailureOr<SmallVector<Value>>
getConsumerShapeGraphInputs(PlaceholderOp placeholder, Operation *consumer) {
  auto dps = cast<DestinationStyleOpInterface>(consumer);
  SmallVector<Value> inputs;
  for (OpOperand *operand : dps.getDpsInputOperands()) {
    Value input = operand->get();
    if (!isa<ShapedType>(input.getType())) {
      continue;
    }
    if (PlaceholderOp::isAllowedShapeGraphInput(input)) {
      inputs.push_back(input);
      continue;
    }

    auto result = dyn_cast<OpResult>(input);
    if (!result) {
      return failure();
    }
    auto producer = dyn_cast<DestinationStyleOpInterface>(result.getOwner());
    if (!producer || producer->getDialect() != placeholder->getDialect()) {
      return failure();
    }
    OpOperand *init = producer.getTiedOpOperand(result);
    if (!init || !init->get().getDefiningOp<PlaceholderOp>()) {
      return failure();
    }
    inputs.push_back(init->get());
  }
  return inputs;
}

FailureOr<ShapeRegionPopulationPlan>
planShapeRegionPopulation(PlaceholderOp placeholder,
                          const ShapeRegionPopulationPatternSet &patterns) {
  if (!placeholder.getShapeRegion().empty()) {
    placeholder.emitOpError(
        "cannot plan population for a non-empty shape region");
    return failure();
  }

  Operation *consumer = placeholder.getDpsConsumer();
  if (!consumer) {
    placeholder.emitOpError(
        "cannot populate shape region without a HIPSR DPS consumer");
    return failure();
  }

  FailureOr<ShapeRegionPopulationMatch> match =
      patterns.match(placeholder, consumer);
  if (failed(match)) {
    return failure();
  }

  unsigned expectedInputCount = getShapedDpsInputCount(consumer);
  SmallVector<Value> inputs(placeholder.getInputs());
  if (inputs.size() > expectedInputCount) {
    FailureOr<SmallVector<Value>> consumerInputs =
        getConsumerShapeGraphInputs(placeholder, consumer);
    if (failed(consumerInputs)) {
      llvm::report_fatal_error(
          "verified placeholder has unreconcilable shape-graph inputs");
    }
    inputs = std::move(*consumerInputs);
  }

  return ShapeRegionPopulationPlan{placeholder, consumer,
                                   match->placeholderType, std::move(inputs),
                                   match->pattern->populate};
}

LogicalResult populateShapeRegion(const ShapeRegionPopulationPlan &plan,
                                  OpBuilder &builder) {
  PlaceholderOp placeholder = plan.placeholder;
  if (!placeholder.getShapeRegion().empty()) {
    return placeholder.emitOpError("cannot populate a non-empty shape region");
  }
  if (!plan.populate) {
    return placeholder.emitOpError(
        "shape-region population plan has no callback");
  }

  if (!llvm::equal(placeholder.getInputs(), plan.inputs)) {
    placeholder.getInputsMutable().assign(plan.inputs);
  }
  placeholder.setPlaceholderType(plan.placeholderType);

  Region &shapeRegion = placeholder.getShapeRegion();
  Block *block = builder.createBlock(&shapeRegion);
  for (Type type : placeholder.getShapeRegionArgumentTypes()) {
    block->addArgument(type, placeholder.getLoc());
  }

  plan.populate(builder, *block, plan.consumer, plan.placeholderType);
  return success();
}

void populateHipsrShapeRegionPatterns(
    ShapeRegionPopulationPatternSet &patterns) {
  populateCastShapeRegionPatterns(patterns);
  populateAddShapeRegionPatterns(patterns);
  populateMatMulShapeRegionPatterns(patterns);
  populateExpandShapeRegionPatterns(patterns);
}

} // namespace hipsr
} // namespace mlir
