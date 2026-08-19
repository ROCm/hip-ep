/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_SHAPE_REGION_POPULATION_UTILS_H
#define HIPSR_SHAPE_REGION_POPULATION_UTILS_H

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir {
namespace hipsr {

/// Creates the entry block of a placeholder's shape region, one argument per
/// entry in `getShapeRegionArgumentTypes`. Shared by
/// hipsr-populate-shape-region and the conversions that fill a region
/// themselves.
inline Block &createPlaceholderShapeBlock(OpBuilder &builder,
                                          PlaceholderOp placeholder) {
  Region &shapeRegion = placeholder.getShapeRegion();
  Block &block = *builder.createBlock(&shapeRegion);
  for (Type type : placeholder.getShapeRegionArgumentTypes()) {
    block.addArgument(type, placeholder.getLoc());
  }
  return block;
}

/// Accesses block arguments of a placeholder-owned shape region.
/// Missing arguments are fatal because createPlaceholderShapeBlock adds one per
/// entry in `getShapeRegionArgumentTypes`.
struct PlaceholderShapeRegionArgs {
  PlaceholderShapeRegionArgs(Block &block) : block(block) {}

  Value ctx() const {
    if (numCtxArgs() == 0) {
      std::string msg;
      llvm::raw_string_ostream(msg)
          << PlaceholderOp::getOperationName()
          << " shape region has no context block argument";
      llvm::report_fatal_error(llvm::StringRef(msg));
    }
    return arg(0);
  }

  Value in(unsigned i) const { return arg(numCtxArgs() + i); }

private:
  Block &block;

  PlaceholderOp getPlaceholder() const {
    return cast<PlaceholderOp>(block.getParentOp());
  }

  PlaceholderType getPlaceholderType() const {
    return getPlaceholder().getPlaceholderType();
  }

  unsigned numCtxArgs() const {
    return getPlaceholderType() == PlaceholderType::Barrier ? 1u : 0u;
  }

  Value arg(unsigned index) const {
    if (index >= block.getNumArguments()) {
      std::string msg;
      llvm::raw_string_ostream(msg)
          << PlaceholderOp::getOperationName()
          << " shape region is missing block arg " << index << " (block has "
          << block.getNumArguments() << ")";
      llvm::report_fatal_error(llvm::StringRef(msg));
    }
    return block.getArgument(index);
  }
};

} // namespace hipsr
} // namespace mlir

#endif // HIPSR_SHAPE_REGION_POPULATION_UTILS_H
