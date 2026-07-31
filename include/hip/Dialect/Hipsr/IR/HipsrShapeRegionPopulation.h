/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_SHAPE_REGION_POPULATION_H
#define HIPSR_SHAPE_REGION_POPULATION_H

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir {
namespace hipsr {

/// A type-erased pair of category selection and population callbacks.
struct ShapeRegionPopulationPattern {
  using GetPlaceholderTypeFunction = PlaceholderType (*)(Operation *consumer);
  using PopulateFunction = void (*)(OpBuilder &builder, Block &block,
                                    Operation *consumer,
                                    PlaceholderType placeholderType);

  llvm::StringRef operationName;
  GetPlaceholderTypeFunction getPlaceholderType;
  PopulateFunction populate;
};

struct ShapeRegionPopulationMatch {
  const ShapeRegionPopulationPattern *pattern;
  PlaceholderType placeholderType;
};

/// Explicit registration for op-local population callbacks. Matching selects
/// callbacks without changing IR, so the pass can plan the whole function
/// before populating any region.
class ShapeRegionPopulationPatternSet {
public:
  template <typename OpTy, PlaceholderType (*GetPlaceholderType)(OpTy),
            void (*Populate)(OpBuilder &, Block &, OpTy, PlaceholderType)>
  void add() {
    for (const ShapeRegionPopulationPattern &pattern : patterns) {
      if (pattern.operationName == OpTy::getOperationName()) {
        llvm::report_fatal_error(
            "duplicate shape-region population pattern for " +
            OpTy::getOperationName());
      }
    }
    patterns.push_back(
        {OpTy::getOperationName(),
         [](Operation *consumer) {
           return GetPlaceholderType(cast<OpTy>(consumer));
         },
         [](OpBuilder &builder, Block &block, Operation *consumer,
            PlaceholderType placeholderType) {
           Populate(builder, block, cast<OpTy>(consumer), placeholderType);
         }});
  }

  FailureOr<ShapeRegionPopulationMatch> match(PlaceholderOp placeholder,
                                              Operation *consumer) const;

private:
  llvm::SmallVector<ShapeRegionPopulationPattern, 8> patterns;
};

struct ShapeRegionPopulationPlan {
  PlaceholderOp placeholder;
  Operation *consumer;
  PlaceholderType placeholderType;
  unsigned inputCount;
  ShapeRegionPopulationPattern::PopulateFunction populate;
};

/// Bounds-checked access to shape-region arguments. Input indices follow the
/// relative order of the consumer's shaped DPS inputs.
template <typename OpTy> class ShapeRegionArgs {
public:
  ShapeRegionArgs(PlaceholderType placeholderType, Block &block)
      : placeholderType(placeholderType), block(block) {
    if (!isa_and_nonnull<PlaceholderOp>(block.getParentOp())) {
      llvm::report_fatal_error(
          "shape-region argument block must belong to a placeholder");
    }
  }

  Value ctx() const {
    if (placeholderType != PlaceholderType::Barrier) {
      llvm::report_fatal_error(
          "only barrier placeholder shape regions receive context");
    }
    return arg(0);
  }

  Value in(unsigned index) const {
    unsigned offset = placeholderType == PlaceholderType::Barrier ? 1 : 0;
    return arg(offset + index);
  }

private:
  Value arg(unsigned index) const {
    if (index >= block.getNumArguments()) {
      std::string message;
      llvm::raw_string_ostream(message)
          << OpTy::getOperationName()
          << " shape region is missing block argument " << index
          << " (block has " << block.getNumArguments() << ")";
      llvm::report_fatal_error(llvm::StringRef(message));
    }
    return block.getArgument(index);
  }

  PlaceholderType placeholderType;
  Block &block;
};

/// Returns the number of shaped DPS input operands on a consumer.
unsigned getShapedDpsInputCount(Operation *consumer);

/// Creates a population plan for a verified empty placeholder.
FailureOr<ShapeRegionPopulationPlan>
planShapeRegionPopulation(PlaceholderOp placeholder,
                          const ShapeRegionPopulationPatternSet &patterns);

/// Applies a collected plan and emits the operation-specific recipe.
LogicalResult populateShapeRegion(const ShapeRegionPopulationPlan &plan,
                                  OpBuilder &builder);

void populateCastShapeRegionPatterns(ShapeRegionPopulationPatternSet &patterns);
void populateAddShapeRegionPatterns(ShapeRegionPopulationPatternSet &patterns);
void populateMatMulShapeRegionPatterns(
    ShapeRegionPopulationPatternSet &patterns);
void populateExpandShapeRegionPatterns(
    ShapeRegionPopulationPatternSet &patterns);
void populateHipsrShapeRegionPatterns(
    ShapeRegionPopulationPatternSet &patterns);

} // namespace hipsr
} // namespace mlir

#endif // HIPSR_SHAPE_REGION_POPULATION_H
