/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/HipsrToLLVM/HipsrToLLVM.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::hipsr;

// PreserveShapeOp does not modify memory, but it must report a side effect
// to avoid being eliminated as a trivially dead operation.
void PreserveShapeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

// An extent tensor or memref holds one entry per dimension of $data, so its
// length must equal the data rank. An opaque !shape.shape carries no length.
LogicalResult PreserveShapeOp::verify() {
  auto shapeType = dyn_cast<ShapedType>(getShape().getType());
  auto dataType = dyn_cast<ShapedType>(getData().getType());
  if (!shapeType || !dataType || !dataType.hasRank() ||
      shapeType.isDynamicDim(0)) {
    return success();
  }
  if (shapeType.getDimSize(0) != dataType.getRank()) {
    return emitOpError() << "extent count " << shapeType.getDimSize(0)
                         << " does not match data rank " << dataType.getRank();
  }
  return success();
}

namespace {

// Erase the op.
struct PreserveShapeLowering : OpRewritePattern<PreserveShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PreserveShapeOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hipsr::populateHipsrPreserveShapeLoweringPatterns(
    const LLVMTypeConverter &, RewritePatternSet &patterns) {
  patterns.add<PreserveShapeLowering>(patterns.getContext());
}
