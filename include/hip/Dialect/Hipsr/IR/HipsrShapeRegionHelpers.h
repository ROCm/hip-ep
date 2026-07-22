/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_SHAPE_REGION_HELPERS_H
#define HIPSR_SHAPE_REGION_HELPERS_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace hipsr {

/// Bounds-checked block-argument accessor. Each op derives this to name its
/// args (getInput(), getA()/getB(), ...) on top of `ins()`. OpTy names the op
/// in the out-of-range diagnostic.
template <typename OpTy> struct ShapeRegionArgs {
  Block &block;

  explicit ShapeRegionArgs(Block &b) : block(b) {}

protected:
  Value ins(unsigned index) const {
    if (index >= block.getNumArguments()) {
      std::string msg;
      llvm::raw_string_ostream(msg)
          << OpTy::getOperationName() << " shape region is missing block arg "
          << index << " (block has " << block.getNumArguments() << ")";
      llvm::report_fatal_error(llvm::StringRef(msg));
    }
    return block.getArgument(index);
  }
};

} // namespace hipsr
} // namespace mlir

#endif // HIPSR_SHAPE_REGION_HELPERS_H
