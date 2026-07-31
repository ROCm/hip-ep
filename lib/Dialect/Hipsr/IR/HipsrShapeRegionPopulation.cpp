/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionPopulation.h"

#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

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
  return ShapeRegionPopulationPlan{
      placeholder, consumer, match->placeholderType,
      getShapedDpsInputCount(consumer), match->pattern->populate};
}

LogicalResult populateShapeRegion(const ShapeRegionPopulationPlan &plan,
                                  OpBuilder &builder) {
  PlaceholderOp placeholder = plan.placeholder;
  if (!placeholder.getShapeRegion().empty()) {
    return placeholder.emitOpError("cannot populate a non-empty shape region");
  }
  if (placeholder.getInputs().size() < plan.inputCount) {
    return placeholder.emitOpError("cannot populate shape region: expected ")
           << plan.inputCount << " shape-graph input(s), got "
           << placeholder.getInputs().size();
  }
  if (!plan.populate) {
    return placeholder.emitOpError(
        "shape-region population plan has no callback");
  }

  unsigned currentInputCount =
      static_cast<unsigned>(placeholder.getInputs().size());
  unsigned trailingInputCount = currentInputCount - plan.inputCount;
  if (trailingInputCount != 0) {
    placeholder.getInputsMutable().erase(plan.inputCount, trailingInputCount);
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
