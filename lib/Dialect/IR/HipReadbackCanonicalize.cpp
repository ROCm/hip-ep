/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/APInt.h"

using namespace mlir;
using namespace mlir::hip;

namespace {

// Recover a host constant or tensor dimension without a tensor-payload load.
//
// Before: readback(cast_i32(from_elements(index_cast_i64(tensor.dim))))
// After:  arith.trunci(index_cast_i64(tensor.dim))
//
// The tensor cast can still have GPU consumers; replace only the readback.
struct ReadbackOfDimensionCast : OpRewritePattern<ReadbackScalarOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReadbackScalarOp op,
                                PatternRewriter &rewriter) const override {
    auto i32 = rewriter.getI32Type();
    auto scalarI32 = RankedTensorType::get({}, i32);
    auto scalarI64 = RankedTensorType::get({}, rewriter.getI64Type());
    if (op.getScalar().getType() != scalarI32 || op.getValue().getType() != i32)
      return failure();

    auto castOp = op.getScalar().getDefiningOp<hip::CastOp>();
    if (!castOp || castOp.getCtx() != op.getCtx() ||
        castOp.getInput().getType() != scalarI64)
      return failure();

    // A static dimension may already have folded through from_elements.
    DenseIntElementsAttr constant;
    if (matchPattern(castOp.getInput(), m_Constant(&constant))) {
      APInt narrowed = constant.getSplatValue<APInt>().trunc(32);
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, i32, rewriter.getIntegerAttr(i32, narrowed));
      return success();
    }

    auto packed = castOp.getInput().getDefiningOp<tensor::FromElementsOp>();
    if (!packed)
      return failure();
    Value host = packed.getElements().front();
    auto indexCast = host.getDefiningOp<arith::IndexCastOp>();
    // Arbitrary scalar producers may load device payload; require metadata.
    if (!indexCast || !indexCast.getIn().getDefiningOp<tensor::DimOp>())
      return failure();

    // Keep the original index-to-i64 conversion and its narrowing semantics.
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, i32, host);
    return success();
  }
};

} // namespace

void ReadbackScalarOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *context) {
  patterns.add<ReadbackOfDimensionCast>(context);
}
