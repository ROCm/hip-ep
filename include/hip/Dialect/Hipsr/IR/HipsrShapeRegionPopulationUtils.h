/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_SHAPE_REGION_POPULATION_UTILS_H
#define HIPSR_SHAPE_REGION_POPULATION_UTILS_H

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace hipsr {

/// Accesses block arguments of a placeholder-owned shape region.
struct ShapeRegionArgs {
  explicit ShapeRegionArgs(Block &block) : block(block) {}

  FailureOr<Value> ctx() const {
    if (numCtxArgs() == 0) {
      return failure();
    }
    return arg(0);
  }

  FailureOr<Value> in(unsigned i) const { return arg(numCtxArgs() + i); }

private:
  Block &block;

  PlaceholderType getPlaceholderType() const {
    return cast<PlaceholderOp>(block.getParentOp()).getPlaceholderType();
  }

  unsigned numCtxArgs() const {
    return getPlaceholderType() == PlaceholderType::Barrier ? 1u : 0u;
  }

  FailureOr<Value> arg(unsigned index) const {
    if (index >= block.getNumArguments()) {
      return failure();
    }
    return block.getArgument(index);
  }
};

} // namespace hipsr
} // namespace mlir

#endif // HIPSR_SHAPE_REGION_POPULATION_UTILS_H
